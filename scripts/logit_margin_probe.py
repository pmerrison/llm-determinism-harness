#!/usr/bin/env python3
"""
Logit-margin probe: measure how susceptible a decode path is to batch-induced drift.

At temperature=0 decoding is pure argmax. Batch composition changes floating-point
reduction order, which jitters logits by roughly 0.01-0.1. An output can only change
if that jitter flips the argmax at some step, which needs a tight top-1 vs top-2 gap.
Drift = susceptibility (how many tight steps) x trigger (batch-dependent jitter).

This probe measures susceptibility in one batch-1 greedy generate with logprobs on.
vLLM returns post-softmax log probabilities, and for the top-2 tokens
log p1 - log p2 == logit1 - logit2 exactly (the logsumexp cancels), so the logprob
gap IS the raw logit margin. The statistic that separated drifting from invariant
runs in our data is the fraction of decode steps with a margin below 2.0 logits;
the minimum margin does not discriminate.

Susceptibility belongs to the decode path the task uses (model x document x output
format x engine version), so probe your production prompt and output format.

Usage:
    modal run --detach logit_margin_probe.py                          # gemma3,qwen3,gemma2
    modal run --detach logit_margin_probe.py --models qwen3 --max-tokens 1024
    modal run --detach logit_margin_probe.py --models qwen3 --json-only
"""

import modal
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

CUDA_BASE = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11"
).entrypoint([])
image = CUDA_BASE.pip_install("vllm==0.23.0", "transformers", "pymupdf")

app = modal.App("logit-margin-probe")
volume = modal.Volume.from_name(os.environ.get("HARNESS_VOLUME", "llm-determinism"),
                                create_if_missing=True)
model_cache = modal.Volume.from_name(os.environ.get("HARNESS_MODEL_CACHE", "model-cache"),
                                     create_if_missing=True)
HF_SECRET_NAME = os.environ.get("HF_SECRET_NAME", "huggingface-secret")

# ctx = the model's native max context (Gemma2 is 8k; the others are large).
# max_model_len is clamped to this so a probe never over-sets a small window.
MODELS = {
    "gemma2": {"name": "Gemma2-9B", "hf_id": "google/gemma-2-9b-it", "ctx": 8192},
    "gemma3": {"name": "Gemma3-27B", "hf_id": "google/gemma-3-27b-it", "ctx": 32768},
    "gemma3-12b": {"name": "Gemma3-12B", "hf_id": "google/gemma-3-12b-it", "ctx": 32768},
    "qwen3": {"name": "Qwen3-32B", "hf_id": "Qwen/Qwen3-32B", "ctx": 32768},
    "qwen2.5": {"name": "Qwen2.5-7B", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "ctx": 32768},
    "qwen35": {"name": "Qwen3.5-27B", "hf_id": "Qwen/Qwen3.5-27B", "ctx": 32768},
    # chat=True: render the prompt through the model's chat template (harmony for
    # gpt-oss). A different decode surface; recorded in the result and never paired
    # with raw-prompt runs.
    "gpt-oss-20b": {"name": "gpt-oss-20b", "hf_id": "openai/gpt-oss-20b", "ctx": 32768, "chat": True},
}

# Thresholds in nats (== logit units). A step whose top1-top2 margin is below the
# threshold is "vulnerable": batch-induced FP jitter of that magnitude could flip it.
TIGHT_THRESHOLDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

EXTRACTION_PROMPT = """You are a financial data extraction assistant. Extract the following Basel III capital metrics from this regulatory disclosure document.

Extract these 6 metrics:
1. CET1 Capital (Common Equity Tier 1) - the absolute amount in millions
2. CET1 Ratio (%) - the percentage ratio
3. Total Capital Ratio (%) - the percentage ratio
4. Total RWA (Risk-Weighted Assets) - the absolute amount in millions
5. Leverage Ratio (%) - also known as SLR (Supplementary Leverage Ratio)
6. LCR (%) - Liquidity Coverage Ratio (if available)

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

# Same schema as basel_batch_variance.py's --json-only arm. Probing with the
# constraint on measures the margins of the decode path the constrained task
# actually uses — the unconstrained probe measures preamble prose Qwen would
# otherwise emit. Expectation: constrained Qwen's tight-step density
# collapses toward Gemma's.
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


def _make_sampling_params(SamplingParams, max_tokens: int, logprobs_k: int, json_only: bool):
    base = dict(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=42,
                logprobs=logprobs_k)
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


def extract_text_from_pdf(pdf_path: str, char_cap: int) -> str:
    import fitz
    doc = fitz.open(pdf_path)
    parts, total = [], 0
    for i in range(len(doc)):
        t = f"--- Page {i + 1} ---\n{doc[i].get_text()}"
        parts.append(t)
        total += len(t)
        if total >= char_cap:
            break
    doc.close()
    return "\n\n".join(parts)[:char_cap]


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@app.function(
    gpu="H200",
    image=image,
    volumes={"/data": volume, "/model-cache": model_cache},
    timeout=3600,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
)
def run_margin_probe(
    model_key: str,
    target_doc: str,
    target_char_cap: int,
    max_tokens: int,
    logprobs_k: int,
    max_model_len: int,
    json_only: bool = False,
) -> Dict[str, Any]:
    import vllm
    from vllm import LLM, SamplingParams

    cfg = MODELS[model_key]
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["HF_HOME"] = "/model-cache"

    print(f"{'='*80}\n🔬 MARGIN PROBE | {cfg['name']} | vLLM {vllm.__version__} | doc={target_doc}\n{'='*80}")

    target_text = extract_text_from_pdf(f"/data/pdfs/{target_doc}", target_char_cap)
    prompt = EXTRACTION_PROMPT.format(document_text=target_text)
    use_chat = bool(cfg.get("chat"))
    if use_chat:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["hf_id"])
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
        print("   CHAT-TEMPLATE mode: prompt rendered via the model's chat template")
    print(f"   prompt chars={len(prompt):,}")

    eff_max_len = min(max_model_len, cfg.get("ctx", max_model_len))  # don't exceed the model's window
    llm = LLM(
        model=cfg["hf_id"],
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_model_len=eff_max_len,
        seed=42,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    model_cache.commit()

    # Greedy decode (temperature=0), logprobs so we see top-k per position.
    sp = _make_sampling_params(SamplingParams, max_tokens, logprobs_k, json_only)
    if json_only:
        print("   JSON-ONLY mode: margins measured on the grammar-constrained decode path")
    out = llm.generate([prompt], sp)[0].outputs[0]
    text = out.text
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Per-step top1-top2 margin (== logit gap, since vLLM returns log-probs).
    margins: List[float] = []
    tight_examples: List[Dict[str, Any]] = []
    n_forced = 0  # grammar-masked steps with <2 finite candidates: structurally can't flip
    for pos, lp_dict in enumerate(out.logprobs or []):
        # lp_dict: {token_id: Logprob(logprob, rank, decoded_token)}
        ranked = sorted(lp_dict.values(), key=lambda x: x.logprob, reverse=True)
        if len(ranked) < 2 or ranked[1].logprob == float("-inf"):
            n_forced += 1
            continue
        margin = ranked[0].logprob - ranked[1].logprob
        if margin != margin or margin == float("inf"):  # NaN / inf guard
            n_forced += 1
            continue
        margins.append(margin)
        if margin < 0.25:  # capture the vulnerable steps for illustration
            tight_examples.append({
                "pos": pos,
                "margin": round(margin, 4),
                "top1": getattr(ranked[0], "decoded_token", None),
                "top2": getattr(ranked[1], "decoded_token", None),
            })

    margins_sorted = sorted(margins)
    n = len(margins_sorted)
    stats = {
        "n_steps": n,
        "min": round(margins_sorted[0], 4) if n else None,
        "p1": round(_percentile(margins_sorted, 0.01), 4) if n else None,
        "p5": round(_percentile(margins_sorted, 0.05), 4) if n else None,
        "p10": round(_percentile(margins_sorted, 0.10), 4) if n else None,
        "p25": round(_percentile(margins_sorted, 0.25), 4) if n else None,
        "median": round(_percentile(margins_sorted, 0.50), 4) if n else None,
        "mean": round(sum(margins_sorted) / n, 4) if n else None,
    }
    # Fraction of steps below each "vulnerable" threshold.
    tight_fraction = {
        str(t): round(sum(1 for m in margins_sorted if m < t) / n, 4) if n else None
        for t in TIGHT_THRESHOLDS
    }

    print(f"   steps={n}  min={stats['min']}  p5={stats['p5']}  median={stats['median']}  mean={stats['mean']}")
    print(f"   tight-step fraction: " + "  ".join(f"<{t}:{tight_fraction[str(t)]}" for t in TIGHT_THRESHOLDS))
    print(f"   #steps with margin<0.1: {sum(1 for m in margins if m < 0.1)}")

    result = {
        "model": cfg["name"],
        "model_key": model_key,
        "hf_id": cfg["hf_id"],
        "vllm_version": vllm.__version__,
        "target_doc": target_doc,
        "max_tokens": max_tokens,
        "logprobs_k": logprobs_k,
        "json_only": json_only,
        "chat_template": use_chat,
        "n_forced_steps": n_forced,
        "output_hash": text_hash,
        "output_head": text[:300],
        "margin_stats": stats,
        "tight_fraction": tight_fraction,
        "n_tight_below_0p1": sum(1 for m in margins if m < 0.1),
        "tightest_steps": sorted(tight_examples, key=lambda x: x["margin"])[:15],
        "all_margins": [round(m, 4) for m in margins],  # full series for plotting
        "timestamp": datetime.now().isoformat(),
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jo = ("_JSONONLY" if json_only else "") + ("_CHAT" if use_chat else "")
    fname = f"/data/margin_probe_{cfg['name']}_{target_doc.replace('.pdf','')}{jo}_{ts}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2, default=str)
    volume.commit()
    print(f"   💾 saved {fname}")
    return result


@app.local_entrypoint()
def main(
    models: str = "gemma3,qwen3,gemma2",
    target: str = "hsbc_q3_2024.pdf",
    target_char_cap: int = 18000,   # ~6k tokens, matches the matched batch-variance run
    max_tokens: int = 512,
    logprobs_k: int = 5,
    max_model_len: int = 32768,
    json_only: bool = False,
):
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    print(f"\n📊 LOGIT-MARGIN PROBE  models={model_list}  doc={target}  max_tokens={max_tokens}\n")

    results = []
    for mk in model_list:
        if mk not in MODELS:
            print(f"   skip unknown model {mk}")
            continue
        results.append(run_margin_probe.remote(
            model_key=mk, target_doc=target, target_char_cap=target_char_cap,
            max_tokens=max_tokens, logprobs_k=logprobs_k, max_model_len=max_model_len,
            json_only=json_only,
        ))

    out_dir = Path(__file__).parent.parent / "results" / "modal-h200"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"margin_probe_compare_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}\n📊 MARGIN COMPARISON — top1-top2 logit gap per decode step ({target})")
    print(f"{'Model':<14}{'steps':>7}{'min':>8}{'p5':>8}{'median':>9}{'mean':>9}{'<0.1':>8}{'<0.5':>8}")
    for r in results:
        if not r:
            continue
        s = r["margin_stats"]; tf = r["tight_fraction"]
        print(f"{r['model']:<14}{s['n_steps']:>7}{s['min']:>8}{s['p5']:>8}{s['median']:>9}"
              f"{s['mean']:>9}{tf['0.1']:>8}{tf['0.5']:>8}")
    print(f"\n   (<0.1 / <0.5 = fraction of decode steps with margin below that many logit units —")
    print(f"    the 'tight races' vulnerable to batch-induced FP jitter. See tight_fraction['2.0'].)")
    print(f"\n   saved: {out_path}\n{'='*80}")
    return results
