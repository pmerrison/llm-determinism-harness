#!/usr/bin/env python3
"""
Basel Pillar III batch-variance acceptance test (Modal H200 + vLLM 0.23.0)

Method (He et al., "Defeating Nondeterminism in LLM Inference"): hold ONE Basel
Pillar III filing fixed at batch index 0, vary how many other requests share the
batch (1, 2, 4, 8, 16) using real filler filings, and hash the target's output at
each size. Repeat each size to separate batch-composition drift from run-to-run
noise. Run the whole matrix with VLLM_BATCH_INVARIANT OFF, then ON.

A configuration passes when the target produces ONE hash across every batch size.
When it drifts, the distinct outputs are parsed and compared against ground truth,
so you can see whether the drift reached the extracted figures or stayed in prose.
This test detects drift; it does not remove it.

Method invariants (held identical across every cell):
  temperature=0, top_p=1, seed=42, enforce_eager=True,
  enable_prefix_caching=False, enable_chunked_prefill=False
  (chunked prefill would make prefill reductions batch-dependent -> a confound).

Prerequisite: the filings listed in GROUND_TRUTH uploaded to /pdfs/ on the Modal
volume (see README).

Usage (defaults reproduce the published HSBC Q3 2024 runs):
    modal run --detach basel_batch_variance.py --smoke                 # cheap check
    modal run --detach basel_batch_variance.py --model qwen3           # Qwen3-32B
    modal run --detach basel_batch_variance.py --model qwen3 --json-only
    modal run --detach basel_batch_variance.py --model qwen3 --max-tokens 2560
    BV_GPU="H200:2" modal run --detach basel_batch_variance.py --model qwen3 --tp 2
"""

import modal
import os
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# CUDA *devel* base (includes nvcc) is required: FlashInfer JIT-compiles its
# prefill kernel at warmup. debian_slim ships only the CUDA runtime -> build fails
# with "Could not find nvcc".
CUDA_BASE = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])
# Latest vLLM (0.23.0) — needed for clean Gemma 3 / Gemma 4 interleaved-attention
# support. transformers unpinned so vLLM pulls a Gemma3-capable version.
VLLM_VERSION = "0.23.0"
image = CUDA_BASE.pip_install(f"vllm=={VLLM_VERSION}", "transformers", "pymupdf")

app = modal.App("basel-batch-variance")
# Account-specific names come from the environment so nothing here is tied to one
# Modal workspace. HF_SECRET_NAME must hold HF_TOKEN (Gemma weights are gated).
volume = modal.Volume.from_name(os.environ.get("HARNESS_VOLUME", "llm-determinism"),
                                create_if_missing=True)
model_cache = modal.Volume.from_name(os.environ.get("HARNESS_MODEL_CACHE", "model-cache"),
                                     create_if_missing=True)
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "huggingface-secret")

# GPU spec is read at import time in the launching process, so tensor-parallel runs
# are launched as e.g.:  BV_GPU="H200:2" modal run --detach basel_batch_variance.py --tp 2
GPU_SPEC = os.environ.get("BV_GPU", "H200")

MODELS = {
    "qwen3": {"name": "Qwen3-32B", "hf_id": "Qwen/Qwen3-32B", "arch": "dense", "ctx": 32768},
    "gemma2": {"name": "Gemma2-9B", "hf_id": "google/gemma-2-9b-it", "arch": "dense", "ctx": 8192},
    # Gemma 3: 128k context (vs Gemma2's 8k), interleaved attention, no soft-capping.
    # Use FLASH_ATTN or default backend — NOT FLASHINFER (it can't serve interleaved
    # attention and silently caps context to the 1024-token sliding window).
    "gemma3": {"name": "Gemma3-27B", "hf_id": "google/gemma-3-27b-it", "arch": "dense"},
    "gemma3-12b": {"name": "Gemma3-12B", "hf_id": "google/gemma-3-12b-it", "arch": "dense"},
    "gpt-oss": {"name": "gpt-oss-120b", "hf_id": "openai/gpt-oss-120b", "arch": "MoE"},
    "qwen2.5": {"name": "Qwen2.5-7B", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "arch": "dense", "ctx": 32768},
    "qwen35": {"name": "Qwen3.5-27B", "hf_id": "Qwen/Qwen3.5-27B", "arch": "gated-linear-attn", "ctx": 32768},
    # chat=True: render prompts through the model's chat template. gpt-oss models are
    # trained on the harmony format and never terminate their analysis channel under a
    # raw completion prompt (observed 2026-08-18: no parseable JSON even at 2560
    # tokens). Chat-template results are a DIFFERENT decode surface; they are recorded
    # as such and must not be compared against raw-prompt runs of the same model.
    "gpt-oss-20b": {"name": "gpt-oss-20b", "hf_id": "openai/gpt-oss-20b", "arch": "MoE", "ctx": 32768, "chat": True},
}

# Human-verified ground truth (mirrors data/ground_truth.csv).
GROUND_TRUTH = {
    "barclays_plc_fy_2024.pdf": {"cet1_capital": {"amount": 48558, "currency": "GBP"}, "cet1_ratio": 13.6, "total_capital_ratio": 19.6, "rwa": {"amount": 358127, "currency": "GBP"}, "leverage_ratio": 5.0, "lcr": 172.4},
    "barclays_q3_2024.pdf": {"cet1_capital": {"amount": 47030, "currency": "GBP"}, "cet1_ratio": 13.8, "total_capital_ratio": 19.9, "rwa": {"amount": 340401, "currency": "GBP"}, "leverage_ratio": 4.9, "lcr": 170.1},
    "goldman_sachs-2Q25.pdf": {"cet1_capital": {"amount": 102586, "currency": "USD"}, "cet1_ratio": 15.3, "total_capital_ratio": 19.0, "rwa": {"amount": 671816, "currency": "USD"}, "leverage_ratio": 5.3, "lcr": None},
    "goldman_sachs-3Q25.pdf": {"cet1_capital": {"amount": 103769, "currency": "USD"}, "cet1_ratio": 15.1, "total_capital_ratio": 18.7, "rwa": {"amount": 687426, "currency": "USD"}, "leverage_ratio": 5.2, "lcr": None},
    # HSBC amounts corrected 2026-08-18: the old values (131,400 / 863,900 and
    # 124,900 / 838,300) were the narrative-section ROUNDINGS ("$131.4bn"), not the
    # own-funds/OV1 table figures. Exact transitional table values below (hsbc_q3:
    # own funds row 29 p6, OV1 total p8; hsbc_q4 equivalents). Result files written
    # before this date carry gt_comparison against the rounded values.
    "hsbc_q3_2024.pdf": {"cet1_capital": {"amount": 131428, "currency": "USD"}, "cet1_ratio": 15.2, "total_capital_ratio": 20.8, "rwa": {"amount": 863923, "currency": "USD"}, "leverage_ratio": 5.7, "lcr": 137},
    "hsbc_q4_2024.pdf": {"cet1_capital": {"amount": 124911, "currency": "USD"}, "cet1_ratio": 14.9, "total_capital_ratio": 20.6, "rwa": {"amount": 838254, "currency": "USD"}, "leverage_ratio": 5.6, "lcr": 138},
    "jpm_q1_2024.pdf": {"cet1_capital": {"amount": 257569, "currency": "USD"}, "cet1_ratio": 15.3, "total_capital_ratio": 17.8, "rwa": {"amount": 1540501, "currency": "USD"}, "leverage_ratio": 6.6, "lcr": None},
    "pnc_q3_2024.pdf": {"cet1_capital": {"amount": 43741, "currency": "USD"}, "cet1_ratio": 10.3, "total_capital_ratio": 13.6, "rwa": {"amount": 432212, "currency": "USD"}, "leverage_ratio": 7.4, "lcr": None},
    "pnc_q4_2024.pdf": {"cet1_capital": {"amount": 44467, "currency": "USD"}, "cet1_ratio": 10.5, "total_capital_ratio": 13.6, "rwa": {"amount": 422399, "currency": "USD"}, "leverage_ratio": 7.5, "lcr": None},
    "stanchart_fy_2024.pdf": {"cet1_capital": {"amount": 35190, "currency": "USD"}, "cet1_ratio": 14.2, "total_capital_ratio": 21.5, "rwa": {"amount": 247065, "currency": "USD"}, "leverage_ratio": 4.4, "lcr": 150.3},
    "stanchart_q3_2024.pdf": {"cet1_capital": {"amount": 35425, "currency": "USD"}, "cet1_ratio": 14.2, "total_capital_ratio": 21.6, "rwa": {"amount": 248924, "currency": "USD"}, "leverage_ratio": 4.2, "lcr": 152.6},
}

# All docs available as targets / fillers (must exist at /data/pdfs/ in the volume).
ALL_DOCS = sorted(GROUND_TRUTH.keys())

EXTRACTION_PROMPT = """You are a financial data extraction assistant. Extract the following Basel III capital metrics from this regulatory disclosure document.

Extract these 6 metrics:
1. CET1 Capital (Common Equity Tier 1) - the absolute amount in millions
2. CET1 Ratio (%) - the percentage ratio
3. Total Capital Ratio (%) - the percentage ratio
4. Total RWA (Risk-Weighted Assets) - the absolute amount in millions
5. Leverage Ratio (%) - also known as SLR (Supplementary Leverage Ratio)
6. LCR (%) - Liquidity Coverage Ratio (if available)

IMPORTANT:
- Extract the MOST RECENT figures (latest reporting date in the document)
- Use the TRANSITIONAL figures if both transitional and fully-loaded are shown
- Report amounts in MILLIONS (not billions or thousands)
- If a metric is not found, use null

Output ONLY valid JSON in this exact format, no other text:
{{
  "cet1_capital": {{"amount": <number>, "currency": "<USD/GBP/EUR>"}},
  "cet1_ratio": <number>,
  "total_capital_ratio": <number>,
  "rwa": {{"amount": <number>, "currency": "<USD/GBP/EUR>"}},
  "leverage_ratio": <number>,
  "lcr": <number or null>
}}

Document text:
---
{document_text}
---

Output ONLY the JSON:"""

# The output-length confound. Qwen3-32B emits a free-text reasoning preamble
# (where all observed drift lives); Gemma emits terse JSON. json_only=True forces
# every model onto the same output surface via structured outputs, so a drift
# comparison is at matched output length. NOTE: constrained decoding masks logits,
# so this measures "does drift survive when the output surface is held fixed",
# not the unconstrained margin distribution.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "cet1_capital": {
            "type": "object",
            "properties": {"amount": {"type": ["number", "null"]},
                           "currency": {"type": "string", "enum": ["USD", "GBP", "EUR"]}},
            "required": ["amount", "currency"],
        },
        "cet1_ratio": {"type": ["number", "null"]},
        "total_capital_ratio": {"type": ["number", "null"]},
        "rwa": {
            "type": "object",
            "properties": {"amount": {"type": ["number", "null"]},
                           "currency": {"type": "string", "enum": ["USD", "GBP", "EUR"]}},
            "required": ["amount", "currency"],
        },
        "leverage_ratio": {"type": ["number", "null"]},
        "lcr": {"type": ["number", "null"]},
    },
    "required": ["cet1_capital", "cet1_ratio", "total_capital_ratio", "rwa",
                 "leverage_ratio", "lcr"],
}


def _make_sampling_params(SamplingParams, max_tokens: int, json_only: bool):
    """Greedy params, optionally with JSON-schema-constrained output. The structured
    output API moved across vLLM versions, so try the current name then fall back."""
    base = dict(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=42)
    if not json_only:
        return SamplingParams(**base)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(**base,
                              structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
    except (ImportError, TypeError):
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(**base,
                              guided_decoding=GuidedDecodingParams(json=JSON_SCHEMA))


def _detect_attn_backend(llm) -> str:
    """Best-effort read of the attention backend vLLM actually selected (0.23 v1).
    Falls back to 'auto(see-logs)'; the engine log line 'Using <X> attention backend'
    is the authoritative record either way."""
    def rget(obj, path):
        for p in path.split("."):
            obj = getattr(obj, p)
        return obj
    group_paths = [
        "llm_engine.model_executor.driver_worker.model_runner.attn_groups",
        "llm_engine.model_executor.driver_worker.worker.model_runner.attn_groups",
    ]
    for path in group_paths:
        try:
            g = rget(llm, path)[0][0]
            for attr in ("backend", "attn_backend"):
                b = getattr(g, attr, None)
                if b is not None:
                    return getattr(b, "__name__", str(b))
        except Exception:
            pass
    direct_paths = [
        "llm_engine.model_executor.driver_worker.model_runner.attn_backend",
        "llm_engine.model_executor.driver_worker.worker.model_runner.attn_backend",
    ]
    for path in direct_paths:
        try:
            b = rget(llm, path)
            return getattr(b, "__name__", str(b))
        except Exception:
            pass
    return "auto(see-logs)"


def extract_text_from_pdf(pdf_path: str, char_cap: int) -> str:
    """Extract text from a PDF, truncated to char_cap characters."""
    import fitz
    doc = fitz.open(pdf_path)
    parts = []
    total = 0
    for page_num in range(len(doc)):
        t = f"--- Page {page_num + 1} ---\n{doc[page_num].get_text()}"
        parts.append(t)
        total += len(t)
        if total >= char_cap:
            break
    doc.close()
    return "\n\n".join(parts)[:char_cap]


def parse_metrics(output_text: str):
    """Best-effort parse of the JSON metrics block from model output.
    Reasoning models often restate the schema template (with <number> placeholders)
    before the real answer, so try candidate blocks last-to-first and take the last
    one that parses."""
    try:
        return json.loads(output_text), True
    except json.JSONDecodeError:
        for m in reversed(re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', output_text)):
            try:
                return json.loads(m), True
            except json.JSONDecodeError:
                continue
        return None, False


def compare_to_ground_truth(parsed: dict, gt: dict):
    """Return per-metric {value, gt, match} comparison. Tolerance: ratios +-0.1, amounts exact-ish."""
    if not parsed or not gt:
        return None

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def amt(d):
        return num(d.get("amount")) if isinstance(d, dict) else num(d)

    out = {}
    checks = {
        "cet1_ratio": (num(parsed.get("cet1_ratio")), num(gt.get("cet1_ratio")), 0.1),
        "total_capital_ratio": (num(parsed.get("total_capital_ratio")), num(gt.get("total_capital_ratio")), 0.1),
        "leverage_ratio": (num(parsed.get("leverage_ratio")), num(gt.get("leverage_ratio")), 0.1),
        "lcr": (num(parsed.get("lcr")), num(gt.get("lcr")), 0.5),
        "cet1_capital_amount": (amt(parsed.get("cet1_capital")), amt(gt.get("cet1_capital")), 1.0),
        "rwa_amount": (amt(parsed.get("rwa")), amt(gt.get("rwa")), 1.0),
    }
    for k, (v, g, tol) in checks.items():
        if v is None and g is None:
            match = True
        elif v is None or g is None:
            match = False
        else:
            match = abs(v - g) <= tol
        out[k] = {"value": v, "gt": g, "match": match}
    return out


@app.function(
    gpu=GPU_SPEC,
    image=image,
    volumes={"/data": volume, "/model-cache": model_cache},
    timeout=7200,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def run_basel_batch_variance(
    model_key: str,
    target_doc: str,
    filler_docs: List[str],
    batch_sizes: List[int],
    repeats: int,
    batch_invariant: bool,
    target_char_cap: int,
    filler_char_cap: int,
    max_tokens: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    attn_backend: str = "FLASHINFER",
    json_only: bool = False,
    tp: int = 1,
    reasoning_effort: str = "",
) -> Dict[str, Any]:
    import time
    import vllm
    from vllm import LLM, SamplingParams

    cfg = MODELS[model_key]
    # Make nvcc visible so FlashInfer's JIT can build its kernels.
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["VLLM_BATCH_INVARIANT"] = "1" if batch_invariant else "0"
    os.environ["HF_HOME"] = "/model-cache"
    # NOTE: VLLM_ATTENTION_BACKEND env var was dropped in vLLM 0.13+ ("Unknown env
    # variable" warning on 0.23). Backend is now selected via the LLM() kwarg below.
    mode = "ON" if batch_invariant else "OFF"

    print(f"{'='*80}")
    print(f"🔬 {cfg['name']} | vLLM {vllm.__version__} | VLLM_BATCH_INVARIANT={mode} | backend={attn_backend or 'auto'} | target={target_doc}")
    print(f"   batch sizes {batch_sizes} x {repeats} repeats")
    print(f"{'='*80}")

    # Build the fixed target prompt (long, ground-truthed).
    target_text = extract_text_from_pdf(f"/data/pdfs/{target_doc}", target_char_cap)
    target_prompt = EXTRACTION_PROMPT.format(document_text=target_text)
    print(f"   target doc chars={len(target_text):,}  prompt chars={len(target_prompt):,}")

    # Build filler prompts. Cycle through available docs to reach the largest
    # batch size (real docs are distinct up to len(filler_docs); beyond that we
    # recycle — duplicates still change batch shape/token count, which is what
    # drives the kernel's reduction choice).
    need = max(batch_sizes) - 1
    filler_prompts = []
    cap_texts = [extract_text_from_pdf(f"/data/pdfs/{d}", filler_char_cap) for d in filler_docs]
    for i in range(need):
        ft = cap_texts[i % len(cap_texts)]
        filler_prompts.append(EXTRACTION_PROMPT.format(document_text=ft))

    use_chat = bool(cfg.get("chat"))
    if use_chat:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
        # harmony's chat template accepts reasoning_effort; "low" shrinks
        # the analysis channel, the free-text region the JSON grammar cannot reach.
        tmpl_kwargs = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        def _render(p):
            try:
                return tok.apply_chat_template([{"role": "user", "content": p}],
                                               tokenize=False, add_generation_prompt=True,
                                               **tmpl_kwargs)
            except TypeError:
                return tok.apply_chat_template([{"role": "user", "content": p}],
                                               tokenize=False, add_generation_prompt=True)
        target_prompt = _render(target_prompt)
        filler_prompts = [_render(p) for p in filler_prompts]
        print(f"   CHAT-TEMPLATE mode: prompts rendered via the model's chat template"
              + (f" (reasoning_effort={reasoning_effort})" if reasoning_effort else ""))
    print(f"   {len(filler_prompts)} filler prompts ({len(cap_texts)} distinct docs, "
          f"cap {filler_char_cap:,} chars each, cycled to reach batch {max(batch_sizes)})")

    # Clamp to the model's native window (Gemma2 is 8k) so a shared default
    # max_model_len never over-sets a small-context model. Mirrors the probe script.
    eff_max_len = min(max_model_len, cfg.get("ctx", max_model_len))
    llm_kwargs = dict(
        model=cfg["hf_id"],
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=eff_max_len,
        tensor_parallel_size=tp,
        seed=42,
        gpu_memory_utilization=0.93,
        trust_remote_code=True,
    )
    if attn_backend and attn_backend.lower() != "auto":
        # vLLM 0.13+ engine arg (replaces the removed VLLM_ATTENTION_BACKEND env var).
        llm_kwargs["attention_backend"] = attn_backend
    try:
        llm = LLM(**llm_kwargs)
    except TypeError as e:
        print(f"   (LLM() rejected attention_backend kwarg: {e}; retrying with auto-select)")
        llm_kwargs.pop("attention_backend", None)
        llm = LLM(**llm_kwargs)
    actual_backend = _detect_attn_backend(llm)
    print(f"   ACTUAL attention backend (vLLM-selected): {actual_backend}")
    try:
        vc = llm.llm_engine.vllm_config
        print(f"   ACTIVE: chunked_prefill={vc.scheduler_config.enable_chunked_prefill} "
              f"max_num_batched_tokens={vc.scheduler_config.max_num_batched_tokens}")
    except Exception as e:
        print(f"   (could not read engine config: {e})")
    model_cache.commit()

    sampling_params = _make_sampling_params(SamplingParams, max_tokens, json_only)
    if json_only:
        print("   JSON-ONLY mode: structured outputs constrain every completion to the schema")

    # per_size[b] = list of {hash, text, batch_latency_s} for the TARGET (index 0) across repeats
    per_size: Dict[int, List[Dict[str, Any]]] = {}
    for b in batch_sizes:
        if b - 1 > len(filler_prompts):
            print(f"  ⚠️  not enough fillers for batch {b} (have {len(filler_prompts)}), skipping")
            continue
        rows = []
        for r in range(repeats):
            batch = [target_prompt] + filler_prompts[: b - 1]
            t0 = time.time()
            outputs = llm.generate(batch, sampling_params)
            dt = time.time() - t0
            target_out = outputs[0].outputs[0].text  # index 0 == target, always
            h = hashlib.sha256(target_out.encode("utf-8")).hexdigest()
            rows.append({"hash": h, "text": target_out, "batch_latency_s": round(dt, 2)})
        per_size[b] = rows
        within = len(set(row["hash"] for row in rows))
        print(f"  batch={b:2d}: target {rows[0]['hash'][:12]}  within-size unique={within}/{repeats}  ({rows[0]['batch_latency_s']}s)")

    # Cross-size determinism for the target.
    all_hashes = [row["hash"] for rows in per_size.values() for row in rows]
    unique_across = len(set(all_hashes))
    invariant = unique_across == 1
    size_to_hash = {b: per_size[b][0]["hash"] for b in per_size}

    # Ground-truth comparison for each DISTINCT target output.
    gt = GROUND_TRUTH.get(target_doc)
    distinct = {}
    for rows in per_size.values():
        for row in rows:
            if row["hash"] not in distinct:
                parsed, ok = parse_metrics(row["text"])
                distinct[row["hash"]] = {
                    "parse_success": ok,
                    "parsed": parsed,
                    "gt_comparison": compare_to_ground_truth(parsed, gt) if ok else None,
                    "text_head": row["text"][:500],
                }

    print(f"\n  → unique target outputs across all batch sizes: {unique_across}")
    print(f"  → batch-invariant for this filing: {'YES ✅' if invariant else 'NO ❌'}")
    if not invariant:
        print(f"  ⚠️  {unique_across} distinct extractions for the SAME filing/seed — inspecting metrics:")
        for h, info in distinct.items():
            if info["gt_comparison"]:
                vals = {k: v["value"] for k, v in info["gt_comparison"].items()}
                print(f"     {h[:12]}: {vals}")

    result = {
        "model": cfg["name"],
        "model_id": cfg["hf_id"],
        "arch": cfg["arch"],
        "batch_invariant": batch_invariant,
        "mode": mode,
        "attn_backend_requested": attn_backend or "auto",
        "attn_backend_actual": actual_backend,
        "target_doc": target_doc,
        "filler_docs": filler_docs,
        "batch_sizes": batch_sizes,
        "repeats": repeats,
        "target_char_cap": target_char_cap,
        "filler_char_cap": filler_char_cap,
        "max_tokens": max_tokens,
        "json_only": json_only,
        "chat_template": use_chat,
        "reasoning_effort": reasoning_effort or None,
        "tensor_parallel_size": tp,
        "unique_target_outputs_across_sizes": unique_across,
        "is_batch_invariant": invariant,
        "size_to_hash": {str(k): v for k, v in size_to_hash.items()},
        "per_size": {
            str(b): [{"hash": row["hash"], "batch_latency_s": row["batch_latency_s"]} for row in rows]
            for b, rows in per_size.items()
        },
        "distinct_outputs": distinct,
        "ground_truth": gt,
        "infrastructure": f"Modal H200 + vLLM {vllm.__version__}",
        "vllm_version": vllm.__version__,
        "timestamp": datetime.now().isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bk = actual_backend.replace("(see-logs)", "").replace("Backend", "") or "auto"
    jo = (("_JSONONLY" if json_only else "") + ("_CHAT" if use_chat else "")
          + (f"_RE{reasoning_effort.upper()}" if reasoning_effort else "")
          + (f"_TP{tp}" if tp > 1 else ""))
    fname = f"/data/basel_bv_{cfg['name']}_{target_doc.replace('.pdf','')}_{bk}_{mode}{jo}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2, default=str)
    volume.commit()
    print(f"  💾 saved {fname}")
    return result


@app.local_entrypoint()
def main(
    model: str = "qwen3",
    target: str = "hsbc_q3_2024.pdf",
    batch_sizes: str = "1,2,4,8,16",
    repeats: int = 3,
    backends: str = "auto",          # comma list; "auto" = let vLLM pick (all published 0.23.0 runs)
    target_char_cap: int = 18000,    # ~6k tokens; the published HSBC runs
    filler_char_cap: int = 9000,     # per filler; too small and drift does not trigger
    max_tokens: int = 1024,          # room for reasoning preamble + JSON; 2560 for matched-length runs
    max_model_len: int = 32768,
    max_num_batched_tokens: int = 98304,
    smoke: bool = False,
    json_only: bool = False,
    tp: int = 1,
    reasoning_effort: str = "",
):
    if model not in MODELS:
        print(f"Unknown model {model}. Options: {list(MODELS)}")
        return
    if target not in GROUND_TRUTH:
        print(f"Unknown target {target}. Options: {ALL_DOCS}")
        return

    if smoke:
        batch_sizes = "1,2"
        repeats = 1
        target_char_cap = 24000
        filler_char_cap = 12000
        max_tokens = 512

    sizes = [int(x) for x in batch_sizes.split(",")]
    backend_list = [b.strip() for b in backends.split(",")] if backends else [""]
    # Distinct filler docs; the container cycles them to reach the largest batch.
    fillers = [d for d in ALL_DOCS if d != target]

    print(f"\n📊 BASEL BATCH-VARIANCE  model={MODELS[model]['name']}  target={target}")
    print(f"   sizes={sizes} x {repeats} repeats | backends={backend_list} | OFF then ON")
    print(f"   target_cap={target_char_cap:,}c  filler_cap={filler_char_cap:,}c  max_tokens={max_tokens}")
    print(f"   distinct filler docs ({len(fillers)}): {fillers}\n")

    # matrix[backend][mode] = result
    matrix: Dict[str, Dict[str, Any]] = {}
    for backend in backend_list:
        bk = backend or "default"
        matrix[bk] = {}
        for bi in (False, True):
            mode_key = "ON" if bi else "OFF"
            try:
                matrix[bk][mode_key] = run_basel_batch_variance.remote(
                    model_key=model, target_doc=target, filler_docs=fillers,
                    batch_sizes=sizes, repeats=repeats, batch_invariant=bi,
                    target_char_cap=target_char_cap, filler_char_cap=filler_char_cap,
                    max_tokens=max_tokens, max_model_len=max_model_len,
                    max_num_batched_tokens=max_num_batched_tokens, attn_backend=backend,
                    json_only=json_only, tp=tp, reasoning_effort=reasoning_effort,
                )
            except Exception as e:
                # e.g. Qwen3.5-27B refuses to boot with the flag ON (GDN_ATTN).
                # Keep the completed arm rather than losing the whole matrix.
                print(f"  ⚠️  {mode_key} arm failed: {e}")
                matrix[bk][mode_key] = {"error": str(e)}

    out_dir = Path(__file__).parent.parent / "results" / "modal-h200"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jo = ("_jsononly" if json_only else "") + (f"_tp{tp}" if tp > 1 else "")
    out_path = out_dir / f"basel_bv_{MODELS[model]['name']}_{target.replace('.pdf','')}_matrix{jo}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2, default=str)

    print(f"\n{'='*80}\n📊 VERDICT — {MODELS[model]['name']}, target {target}")
    for bk, modes in matrix.items():
        print(f"\n  backend={bk}")
        for mode in ("OFF", "ON"):
            r = modes.get(mode)
            if not r or "error" in r:
                if r:
                    print(f"    {mode:3s}: FAILED — {r['error'][:120]}")
                continue
            u = r["unique_target_outputs_across_sizes"]
            verdict = "INVARIANT ✅" if r["is_batch_invariant"] else f"DRIFTS ❌ ({u} distinct outputs)"
            print(f"    {mode:3s} (BATCH_INVARIANT={'1' if mode=='ON' else '0'}): {verdict}")
            print(f"         size→hash: " + "  ".join(f"{b}:{h[:8]}" for b, h in r["size_to_hash"].items()))
            # surface the drifting metrics
            if not r["is_batch_invariant"]:
                for h, info in r["distinct_outputs"].items():
                    gc = info.get("gt_comparison")
                    if gc:
                        vals = {k: v["value"] for k, v in gc.items()}
                        print(f"         {h[:8]} metrics: {vals}")
    print(f"\n   saved: {out_path}\n{'='*80}")
    return matrix
