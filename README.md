# llm-determinism-harness

A small harness for testing whether an LLM extraction pipeline gives the same answer
when the only thing that changes is what else is sharing the GPU batch.

It contains three tools and the results that back the findings below:

- **Batch-variance acceptance test** (`scripts/basel_batch_variance.py`): pins one Basel
  III Pillar 3 filing at batch index 0, varies the batch around it (1, 2, 4, 8, 16, three
  repeats each), and hashes the target's output. One hash across all batch sizes passes.
  Anything else fails, and the distinct outputs are parsed and compared to ground truth.
- **Logit-margin screen** (`scripts/logit_margin_probe.py` +
  `scripts/p0_discriminator_analysis.py`): one batch-1 greedy generate with logprobs,
  recording the top-1 vs top-2 logit gap at every decode step. The fraction of steps below
  2.0 logits ("tight-step density") separated drifting from invariant runs in our data.
- **Probe your own endpoint** (`scripts/endpoint_drift_probe.py`): the same test run
  client-side against any OpenAI-compatible endpoint, varying concurrent traffic instead
  of pinned batches.

## Why temperature zero is not enough

At temperature zero, decoding is pure argmax, so there is no sampling left to remove.
But batch composition changes floating-point reduction order, which jitters logits by
roughly 0.01–0.1. When the top two candidates at some step are closer than that, the
winner can flip, and the generation diverges from that point. So drift needs two things:
**susceptibility**, meaning tight races in the decode path, and a **trigger**, meaning
batch-dependent jitter. Batch-invariant kernels (`VLLM_BATCH_INVARIANT=1`) go after the
trigger. The margin screen measures susceptibility.

## Headline results

All on Modal H200, vLLM 0.23.0, temperature 0, seed 42, chunked prefill and prefix caching
off. Target: HSBC Q3 2024 Pillar 3 unless stated. These are findings **in our data, on
this engine**. They are not constants, and verdicts don't carry over between engine
versions.

### Batch composition changed a reported capital figure

Qwen3-32B, HSBC Q3 2024, flag off: at batch size 1 the extracted CET1 capital was
**$131,428m**. At batch sizes 2, 4, 8 and 16 it was **$131,400m**. Each size was stable
across its three repeats. Both figures are in the filing: $131,428m is the own-funds
table value, and $131,400m is the narrative's "$131.4bn" rounding. Batch composition
decided which line of the filing the pipeline quoted.
(`results/modal-h200/basel_bv_Qwen3-32B_hsbc_q3_2024_matrix_20260622_192027.json`)

### Batch-variance verdicts

Distinct target outputs across batch sizes 1–16 × 3 repeats (1 = byte-invariant).

| Model | Document / surface | Flag OFF | Flag ON |
|---|---|---|---|
| Gemma2-9B | HSBC Q3 2024 | 1 | 1 |
| Gemma2-9B | HSBC Q3 2024, tensor parallel 2 | 1 | 1 |
| Gemma2-9B | Barclays FY 2024 | 1 | 1 |
| Gemma3-27B | HSBC Q3 2024 | 1 | 1 |
| Gemma3-12B | Barclays FY 2024, full filing, batch 1–8 | 1 | 1 |
| Qwen2.5-7B | HSBC Q3 2024 | 1 | 1 |
| Qwen3-32B | Barclays FY 2024 | 1 | 1 |
| Qwen3-32B | HSBC Q3 2024 | 6 | 2 |
| Qwen3-32B | HSBC Q3 2024, tensor parallel 2 | 5 | 1 |
| Qwen3-32B | HSBC Q3 2024, JSON-schema constrained | 1 | 1 |
| Qwen3.5-27B | HSBC Q3 2024 | 5 | engine fails to start |
| Qwen3.5-27B | HSBC Q3 2024, JSON-schema constrained | 1 | engine fails to start |
| gpt-oss-20b | HSBC Q3 2024, raw prompt | 5 | engine fails to start |
| gpt-oss-20b | HSBC Q3 2024, chat template | 5 | engine fails to start |
| gpt-oss-20b | HSBC Q3 2024, chat template + JSON schema | 5 | engine fails to start |
| gpt-oss-20b | as above + `reasoning_effort=low` | 2 | engine fails to start |

What this shows:

- **The flag is a partial control.** It depends on the model and configuration, it has
  no effect on models that were already invariant, and on this engine vLLM refuses to
  start with it for Qwen3.5-27B and gpt-oss-20b.
- **It's not about parameter count.** Gemma 9B, 12B and 27B are invariant while a 32B
  drifts.
- **Constraining the output was the control that worked on both counts.** Under a JSON
  schema, Qwen3-32B and Qwen3.5-27B are byte-invariant with the flag off and produce the
  same hash as each other. Both return the exact table figure ($131,428m) at every batch
  size.
- **The constraint doesn't reach a reasoning channel.** vLLM's structured outputs exempt
  gpt-oss-20b's analysis channel, so it still drifts. Its extracted values are identical
  across all its drifting outputs, which makes the decision layer the only guarantee
  available for that model on this stack.

### Tight-step density

Fraction of decode steps whose top-1 vs top-2 margin is below 2.0 logits, as reported by
`python3 scripts/p0_discriminator_analysis.py` over `results/modal-h200/`. Drifting
models are probed at a generation length matched to their batch-variance run.

| Model | Document / surface | Steps | Density < 2.0 | Verdict (flag OFF) |
|---|---|---|---|---|
| Gemma3-27B | HSBC Q3 2024 | 110 | 0.0% | invariant |
| Gemma3-12B | Barclays FY 2024 | 119 | 0.8% | invariant |
| Qwen2.5-7B | HSBC Q3 2024 | 197 | 1.0% | invariant |
| Qwen3-32B | Barclays FY 2024 | 512 | 2.3% | invariant |
| Gemma2-9B | HSBC Q3 2024 | 118 | 4.2% | invariant |
| Gemma2-9B | Barclays FY 2024 | 115 | 6.1% | invariant |
| Qwen3.5-27B | HSBC Q3 2024 | 1934 | 12.3% | drifts (5) |
| Qwen3-32B | HSBC Q3 2024 | 1024 | 14.3% | drifts (6) |
| gpt-oss-20b | HSBC Q3 2024 | 906 | 25.9% | drifts (5) |
| Qwen3-32B | HSBC Q3 2024, JSON schema | 86 | 1.2% | invariant |
| Qwen3.5-27B | HSBC Q3 2024, JSON schema | 84 | 2.4% | invariant |
| gpt-oss-20b | HSBC Q3 2024, chat + JSON schema | 479 | 20.7% | drifts (5) |

In this data, invariant runs sit at or below 6.1% and drifting runs at or above 12.3%.
The minimum margin does **not** discriminate: Gemma2-9B has an exact 0.0 tie and never
flips. The same model can land on either side of the line depending on the document
(Qwen3-32B: HSBC vs Barclays) and the output format (free text vs schema), so probe your
own prompt and output format, and re-probe whenever the engine changes. A point measured
through a served endpoint on a different engine (gemma-3-1b, 7.3%, drifts) falls between
the two groups. Read the screen as: low is safe, high is dangerous, and anything in the
middle needs the acceptance test.

Per-probe files also record `tight_fraction` at probe time, computed from unrounded
margins. Those figures run slightly higher (for example 14.4% / 12.7% / 26.4% for the
three drifters) because `all_margins` is stored rounded to 4 decimal places, which shifts
steps that sit right at the 2.0 boundary. The verdicts and the separation are the same
either way. The table above uses the recomputed figures so that it reproduces exactly
from this repo.

### On a served endpoint (Modal Auto Endpoint: tuned SGLang, speculative decoding)

- Qwen3.5-27B-FP8: 15 distinct outputs in 15 requests across concurrency levels 1–16.
  The endpoint can't return logprobs at all ("DFLASH speculative decoding does not support
  return_logprob yet"), so the margin screen can't run there.
- gemma-3-1b-it: stable within each concurrency level, three distinct outputs across
  levels, and the extracted CET1 ratio changed from 11.5 (concurrency 1) to 13.3
  (concurrency 2–4). Read this as engine behaviour, not a model claim: a 1B model is far
  below what this task needs.

(`results/modal-endpoint/`)

## Using it as a control

1. **Screen** candidate models offline on your documents and output format with the margin
   probe.
2. **Constrain** the output to a schema wherever the task allows it.
3. **Acceptance-test** by varying the batch (or concurrency, against an endpoint). Never
   test by re-running the same prompt, which passes on systems that aren't reproducible.
   The test **detects** drift; it doesn't remove it.
4. **Re-run** the test on every model, engine version, GPU, tensor-parallel or
   serving-stack change.

## Running it

> **Note:** the result files were produced by the research versions of these scripts.
> The published scripts have since been cleaned up (paths, auth, defaults) and compile,
> but have not yet been re-run end to end. Treat a first run as a check, and open an
> issue if your results differ from the files here.

### Setup

- A [Modal](https://modal.com) account, with the CLI installed and logged in.
- A Modal secret holding `HF_TOKEN`, named `huggingface-secret` (or set `HF_SECRET_NAME`).
  The Gemma weights are gated on Hugging Face.
- A Modal volume for inputs and outputs, named `llm-determinism` (or set `HARNESS_VOLUME`).
  Model weights are cached in a second volume, `model-cache` (or set `HARNESS_MODEL_CACHE`).
- The Pillar 3 filings, uploaded to `/pdfs/` on that volume. They are not included here.
  Download them from each bank's investor-relations site and save them under these names:

  | File name | Filing |
  |---|---|
  | `hsbc_q3_2024.pdf` | HSBC Holdings plc, Pillar 3 disclosure, 30 September 2024 |
  | `hsbc_q4_2024.pdf` | HSBC Holdings plc, Pillar 3 disclosure, 31 December 2024 |
  | `barclays_plc_fy_2024.pdf` | Barclays PLC, Pillar 3 disclosure, full year 2024 |
  | `barclays_q3_2024.pdf` | Barclays PLC, Pillar 3 disclosure, Q3 2024 |
  | `stanchart_fy_2024.pdf` | Standard Chartered PLC, Pillar 3 disclosure, full year 2024 |
  | `stanchart_q3_2024.pdf` | Standard Chartered PLC, Pillar 3 disclosure, Q3 2024 |
  | `goldman_sachs-2Q25.pdf` | The Goldman Sachs Group, Pillar 3 disclosure, 2Q 2025 |
  | `goldman_sachs-3Q25.pdf` | The Goldman Sachs Group, Pillar 3 disclosure, 3Q 2025 |
  | `jpm_q1_2024.pdf` | JPMorgan Chase & Co., Pillar 3 disclosure, Q1 2024 |
  | `pnc_q3_2024.pdf` | The PNC Financial Services Group, Pillar 3 disclosure, Q3 2024 |
  | `pnc_q4_2024.pdf` | The PNC Financial Services Group, Pillar 3 disclosure, Q4 2024 |

  ```
  modal volume put llm-determinism ./pdfs /pdfs
  ```

  Every filing other than the target is used as batch filler, so the full set is needed to
  reproduce the published batches exactly. `data/ground_truth.csv` has the human-verified
  figures. Its `jpm_q3_2024` row isn't used by the scripts.

The stack is pinned: H200, CUDA 12.8 devel image, vLLM 0.23.0. The engine version is part
of what a result *means*, so change it and the verdicts no longer apply.

### Batch-variance acceptance test

```
cd scripts
modal run --detach basel_batch_variance.py --smoke                       # cheap check
modal run --detach basel_batch_variance.py --model qwen3                 # HSBC Q3 2024, OFF then ON
modal run --detach basel_batch_variance.py --model qwen3 --json-only     # schema-constrained
modal run --detach basel_batch_variance.py --model qwen35 --max-tokens 2560
modal run --detach basel_batch_variance.py --model qwen3 --target barclays_plc_fy_2024.pdf
BV_GPU="H200:2" modal run --detach basel_batch_variance.py --model qwen3 --tp 2
```

Models: `qwen3`, `qwen35`, `qwen2.5`, `gemma2`, `gemma3`, `gemma3-12b`, `gpt-oss-20b`.
The matrix is written to `results/modal-h200/`, and each arm also goes to the volume.

### Margin probe and analysis

```
cd scripts
modal run --detach logit_margin_probe.py --models gemma2,qwen3
modal run --detach logit_margin_probe.py --models qwen3 --max-tokens 1024
modal run --detach logit_margin_probe.py --models qwen3 --json-only
python3 p0_discriminator_analysis.py        # local, no GPU
```

The analysis pairs each probe with the batch-variance verdict for the same model,
document, output surface and engine version, and reports which statistic separates them.

### Probe your own endpoint

```
export ENDPOINT_API_KEY=...       # bearer token, if your endpoint needs one
python3 scripts/endpoint_drift_probe.py --base-url https://<host>/v1 --model <model-id> \
    --pdf-dir ./pdfs --engine-label "vLLM 0.23.0, 1xH200"
```

The script first checks whether the endpoint returns `top_logprobs`. If it does, it runs
the margin screen through the endpoint as well. If not, it reports drift only. In that
case, screen the model offline with `logit_margin_probe.py`, then acceptance-test the
endpoint with this script. Results go to `results/endpoint/`, and the base URL is left
out unless you pass `--record-url`.

## Licence

Apache-2.0, copyright 2026 Paul Merrison. See `LICENSE` and `NOTICE`.
