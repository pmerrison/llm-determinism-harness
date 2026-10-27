#!/usr/bin/env python3
"""
Probe your own endpoint: client-side batch-variance test against any served
OpenAI-compatible chat endpoint (vLLM, SGLang, a managed provider).

There is no batch control from the client, which is the point: this measures the
serving surface a team actually uses, where the "batch" is whatever concurrent
traffic lands in the same scheduling tick. Method: send the target prompt alongside
varying amounts of concurrent filler traffic, hash the target's completion, repeat.

If the endpoint returns top_logprobs, the margin probe also runs through it and the
tight-step density is reported. If it does not (for example SGLang with speculative
decoding), the script reports drift only: probe margins offline on batch-1 vLLM with
logit_margin_probe.py, then acceptance-test the served endpoint with this script.

Auth: set ENDPOINT_API_KEY for a bearer token, and/or MODAL_PROXY_TOKEN_ID +
MODAL_PROXY_TOKEN_SECRET for a Modal proxy-auth endpoint.

Usage:
    python3 endpoint_drift_probe.py --base-url https://<host>/v1 --model <model-id> \
        --pdf-dir ../pdfs
    python3 endpoint_drift_probe.py --base-url ... --model ... --check-logprobs-only

Results land in ../results/endpoint/ (the base URL is not recorded unless
--record-url is passed).
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(os.environ.get("PDF_DIR", REPO_ROOT / "pdfs"))
OUT_DIR = REPO_ROOT / "results" / "endpoint"

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


def pdf_text(name: str, cap: int) -> str:
    import fitz
    doc = fitz.open(str(PDF_DIR / name))
    parts, total = [], 0
    for page in doc:
        t = page.get_text()
        parts.append(t)
        total += len(t)
        if total >= cap:
            break
    doc.close()
    return "\n\n".join(parts)[:cap]


def _auth_headers():
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ENDPOINT_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['ENDPOINT_API_KEY']}"
    if os.environ.get("MODAL_PROXY_TOKEN_ID"):
        headers["Modal-Key"] = os.environ["MODAL_PROXY_TOKEN_ID"]
        headers["Modal-Secret"] = os.environ.get("MODAL_PROXY_TOKEN_SECRET", "")
    return headers


def call_endpoint(base_url, model, prompt, max_tokens, timeout, logprobs=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": max_tokens,
    }
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 5
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=_auth_headers(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def measure_margins(base_url, model, prompt, max_tokens, timeout):
    """One greedy generate with top_logprobs through the endpoint; compute the
    tight-step density client-side. This is the margin probe, cross-engine."""
    resp = call_endpoint(base_url, model, prompt, max_tokens, timeout, logprobs=True)
    content = resp["choices"][0].get("logprobs", {}).get("content") or []
    margins = []
    for step in content:
        tops = step.get("top_logprobs") or []
        if len(tops) >= 2:
            m = tops[0]["logprob"] - tops[1]["logprob"]
            if m == m and m != float("inf"):
                margins.append(m)
    n = len(margins)
    if not n:
        return {"n_steps": 0}
    ms = sorted(margins)
    out = {"n_steps": n, "min": round(ms[0], 4),
           "median": round(ms[n // 2], 4)}
    for t in (0.1, 0.5, 1.0, 2.0):
        out[f"frac<{t}"] = round(sum(1 for m in ms if m < t) / n, 4)
    print(f"margin probe via endpoint: {out}")
    return out


def check_logprobs(base_url, model, timeout):
    resp = call_endpoint(base_url, model, "Reply with the single word: yes",
                         max_tokens=8, timeout=timeout, logprobs=True)
    choice = resp["choices"][0]
    lp = choice.get("logprobs")
    ok = bool(lp and lp.get("content"))
    detail = {}
    if ok:
        first = lp["content"][0]
        tops = first.get("top_logprobs") or []
        detail = {"n_top_logprobs": len(tops),
                  "example_margin": (round(tops[0]["logprob"] - tops[1]["logprob"], 4)
                                     if len(tops) >= 2 else None)}
    print(f"logprobs supported: {ok}  {detail}")
    return {"logprobs_supported": ok, **detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="hsbc_q3_2024.pdf")
    ap.add_argument("--target-char-cap", type=int, default=18000)
    ap.add_argument("--filler-char-cap", type=int, default=9000)
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--check-logprobs-only", action="store_true")
    ap.add_argument("--pdf-dir", default=None, help="directory of filings (default: $PDF_DIR or ../pdfs)")
    ap.add_argument("--engine-label", default="unspecified",
                    help="free-text description of the serving stack, recorded in the result")
    ap.add_argument("--record-url", action="store_true", help="record --base-url in the result file")
    args = ap.parse_args()
    global PDF_DIR
    if args.pdf_dir:
        PDF_DIR = Path(args.pdf_dir)

    try:
        lp_info = check_logprobs(args.base_url, args.model, args.timeout)
    except Exception as e:
        # e.g. SGLang: "DFLASH speculative decoding does not support return_logprob yet."
        body = getattr(e, "read", lambda: b"")()
        detail = (body[:200].decode(errors="replace") if body else str(e))
        print(f"logprobs unsupported on this endpoint: {detail}")
        lp_info = {"logprobs_supported": False, "error": detail}
    if args.check_logprobs_only:
        return 0

    fillers = [f for f in sorted(os.listdir(PDF_DIR))
               if f.endswith(".pdf") and f != args.target]
    target_prompt = EXTRACTION_PROMPT.format(
        document_text=pdf_text(args.target, args.target_char_cap))
    if lp_info.get("logprobs_supported"):
        margin_info = measure_margins(args.base_url, args.model, target_prompt,
                                      args.max_tokens, args.timeout)
    else:
        margin_info = {}
        print("no logprobs from this endpoint: reporting drift only "
              "(run logit_margin_probe.py offline for the margin screen)")
    filler_prompts = [EXTRACTION_PROMPT.format(
        document_text=pdf_text(f, args.filler_char_cap)) for f in fillers]

    sizes = [int(x) for x in args.concurrency.split(",")]
    per_size = {}
    for c in sizes:
        rows = []
        for rep in range(args.repeats):
            prompts = [target_prompt] + [filler_prompts[i % len(filler_prompts)]
                                         for i in range(c - 1)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
                futs = [ex.submit(call_endpoint, args.base_url, args.model, p,
                                  args.max_tokens, args.timeout) for p in prompts]
                results = [f.result() for f in futs]
            msg = results[0]["choices"][0]["message"]
            final = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            h = hashlib.sha256((reasoning + "\x00" + final).encode()).hexdigest()
            hf = hashlib.sha256(final.encode()).hexdigest()
            rows.append({"hash": h, "hash_final": hf,
                         "text_head": final[:500], "reasoning_head": reasoning[:200]})
        per_size[c] = rows
        within = len(set(r["hash"] for r in rows))
        within_f = len(set(r["hash_final"] for r in rows))
        print(f"concurrency={c:2d}: {rows[0]['hash'][:12]}  within unique={within}/{args.repeats}"
              f"  final-channel unique={within_f}/{args.repeats}")

    all_hashes = [r["hash"] for rows in per_size.values() for r in rows]
    unique = len(set(all_hashes))
    unique_final = len(set(r["hash_final"] for rows in per_size.values() for r in rows))
    print(f"\nunique target outputs across all concurrency levels: {unique}"
          f" (final channel only: {unique_final})")
    print("verdict:", "INVARIANT" if unique == 1 else f"DRIFTS ({unique} distinct)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"endpoint_bv_{args.model.replace('/', '_')}_{args.target.replace('.pdf', '')}_{ts}.json"
    out.write_text(json.dumps({
        "method": "client-side concurrent traffic against served endpoint (no batch pinning)",
        "engine": args.engine_label,
        **({"base_url": args.base_url} if args.record_url else {}),
        "model": args.model,
        "target_doc": args.target, "concurrency": sizes, "repeats": args.repeats,
        "max_tokens": args.max_tokens, "temperature": 0.0, "seed": 42,
        "logprobs_check": lp_info,
        "margin_probe_via_endpoint": margin_info,
        "unique_target_outputs": unique,
        "unique_final_channel_outputs": unique_final,
        "per_size": {str(k): v for k, v in per_size.items()},
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
