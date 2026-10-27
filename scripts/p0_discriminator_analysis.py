#!/usr/bin/env python3
"""
Choose the discriminating margin statistic. Local, no GPU.

Loads every margin probe result on disk (both per-model margin_probe_*.json files
and the margin_probe_compare_*.json lists), recomputes candidate
statistics from the raw `all_margins` series, pairs each probe with the observed
batch-variance verdict for the same model (basel_bv_*_matrix_*.json, OFF mode),
and reports which statistic cleanly separates invariant from drifting models.

Candidates: min, p1, p5, and the fraction of decode steps with margin below
0.01 / 0.1 / 0.5 / 1.0 / 2.0 logits.

Usage:
    python3 p0_discriminator_analysis.py            # scans ../results/modal-h200
    python3 p0_discriminator_analysis.py <dir> ...  # additional result dirs
"""

import json
import sys
from pathlib import Path

THRESHOLDS = [0.01, 0.1, 0.5, 1.0, 2.0]


def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def load_probes(dirs):
    """Yield per-model probe dicts from margin_probe_*.json (single-dict or list)."""
    seen = {}
    for d in dirs:
        for f in sorted(Path(d).glob("margin_probe_*.json")):
            data = json.loads(f.read_text())
            entries = data if isinstance(data, list) else [data]
            for e in entries:
                if not e or "all_margins" not in e:
                    continue
                key = (e["model"], e.get("target_doc"), e.get("output_hash"))
                # later files win only if this exact probe wasn't already loaded
                if key not in seen:
                    e["_source"] = f.name
                    seen[key] = e
    return list(seen.values())


def load_verdicts(dirs):
    """(model, doc, json_only) -> unique OFF-mode output count from basel_bv matrix files.
    Pairing includes the output surface: a constrained probe must only pair with a
    constrained batch-variance verdict, and vice versa."""
    verdicts = {}
    for d in dirs:
        for f in sorted(Path(d).glob("basel_bv_*_matrix_*.json")):
            data = json.loads(f.read_text())
            for backend, modes in data.items():
                off = modes.get("OFF")
                if not off or "error" in off:
                    continue
                # Engine version is part of the identity: batch-invariance behaviour is
                # version-specific, so a 0.13-era verdict must never pair with a 0.23 probe.
                # Files predating the vllm_version field pair with nothing.
                key = (off["model"], off["target_doc"], bool(off.get("json_only", False)),
                       bool(off.get("chat_template", False)), off.get("vllm_version"))
                u = off["unique_target_outputs_across_sizes"]
                # keep the worst (max) drift seen for that pairing
                if key not in verdicts or u > verdicts[key]:
                    verdicts[key] = u
    return verdicts


def stats_for(margins):
    ms = sorted(margins)
    n = len(ms)
    row = {
        "n_steps": n,
        "min": ms[0],
        "p1": percentile(ms, 0.01),
        "p5": percentile(ms, 0.05),
        "median": percentile(ms, 0.50),
    }
    for t in THRESHOLDS:
        row[f"frac<{t}"] = sum(1 for m in ms if m < t) / n
    return row


def main():
    base = Path(__file__).parent.parent / "results" / "modal-h200"
    dirs = [base] + [Path(a) for a in sys.argv[1:]]
    probes = load_probes(dirs)
    verdicts = load_verdicts(dirs)

    if not probes:
        print("No margin_probe_*.json files found in:", *dirs, sep="\n  ")
        return 1

    rows = []
    for p in probes:
        row = stats_for([float(m) for m in p["all_margins"]])
        row["model"] = p["model"]
        jo = bool(p.get("json_only", False))
        ct = bool(p.get("chat_template", False))
        row["doc"] = p.get("target_doc", "?") + (" [json]" if jo else "") + (" [chat]" if ct else "")
        u = verdicts.get((p["model"], p.get("target_doc", "?"), jo, ct, p.get("vllm_version")))
        row["unique_outputs_OFF"] = u
        row["observed"] = ("?" if u is None else "invariant" if u == 1 else f"drifts({u})")
        row["source"] = p["_source"]
        rows.append(row)

    rows.sort(key=lambda r: (r["unique_outputs_OFF"] is None, r["unique_outputs_OFF"] or 0))

    cols = ["min", "p1", "p5", "median"] + [f"frac<{t}" for t in THRESHOLDS]
    print(f"\n{'Model':<14}{'doc':<22}{'steps':>6}", end="")
    for c in cols:
        print(f"{c:>10}", end="")
    print(f"{'observed':>14}")
    for r in rows:
        print(f"{r['model']:<14}{r['doc']:<22}{r['n_steps']:>6}", end="")
        for c in cols:
            print(f"{r[c]:>10.4f}", end="")
        print(f"{r['observed']:>14}")

    # Separation check: for each candidate, does every drifting model score
    # strictly worse (higher frac / lower margin) than every invariant model?
    inv = [r for r in rows if r["unique_outputs_OFF"] == 1]
    drf = [r for r in rows if (r["unique_outputs_OFF"] or 0) > 1]
    print(f"\nSeparation check ({len(inv)} invariant vs {len(drf)} drifting, "
          f"{sum(1 for r in rows if r['unique_outputs_OFF'] is None)} unmatched):")
    if not inv or not drf:
        print("  Not enough matched probes on both sides to test separation yet.")
        for r in rows:
            if r["unique_outputs_OFF"] is None:
                print(f"  unmatched: {r['model']} / {r['doc']} ({r['source']})")
        return 0

    for c in cols:
        lower_is_risky = not c.startswith("frac")  # margins: low=risky; fractions: high=risky
        if lower_is_risky:
            worst_inv, best_drf = min(r[c] for r in inv), max(r[c] for r in drf)
            sep = all(r[c] > d[c] for r in inv for d in drf)
            gap = worst_inv - best_drf
        else:
            worst_inv, best_drf = max(r[c] for r in inv), min(r[c] for r in drf)
            sep = all(r[c] < d[c] for r in inv for d in drf)
            gap = best_drf - worst_inv
        print(f"  {c:>10}: {'SEPARATES' if sep else 'overlaps '}  "
              f"invariant-side extreme={worst_inv:.4f}  drift-side extreme={best_drf:.4f}  "
              f"gap={gap:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
