---
license: cc-by-sa-4.0
pretty_name: llms-xx-to-en-translation
task_categories:
  - translation
tags:
  - machine-translation
  - benchmark
  - x-to-en
---

# llms-xx-to-en-translation

Open models translating GeoReSeT corpus languages into English.
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated 2026-11-13 14:26 CET

## Progress

| Model | FLORES+ | NTREX-128 | WMT22 | WMT23 | Tatoeba |
|---|---|---|---|---|---|
| gemma-3-27b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-26b-a4b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-31b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-e4b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| hy-mt2-30b-a3b | ░░░░░░ 0/5 | ░░░░░░ 0/4 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/4 |
| hy-mt2-7b | ░░░░░░ 0/5 | ░░░░░░ 0/4 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/4 |
| lfm3-1.7b | — | — | — | — | — |
| madlad400-10b-mt | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| madlad400-3b-mt | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| nllb-200-3.3b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| plamo-3-translate-2b | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| plamo-3-translate-8b | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| qwen3.8-27b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| tiny-aya-global | ░░░░░░ 0/8 | ░░░░░░ 0/8 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/6 |
| translategemma-12b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| translategemma-27b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| translategemma-4b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |

Each cell: shards generated / total. A shard is 8 languages of one benchmark.

## Results

_No scores yet. Hypotheses are generated first; scores follow on bf16 GPUs._



## Files

- `results/<model>/<benchmark>/<shard>.parquet`: hypotheses and token counts
- `scores/<model>/<benchmark>/<shard>.parquet`: per-segment scores and failure flags
- `manifests/<model>/job-<id>.json`: full model config and run settings for each job

Each row carries its own provenance: `model_revision`, `prompt_template`, `prompt_version`, `prompt_sha256`,
`decoding`, `seed`, `budget_*`, `max_new_tokens`, `dtype`, `backend`, `software`, `gpu`, `node`, `job_id`,
`code_commit`, `generated_at`. Generated text is in `hypothesis`; the untouched model output is in
`raw_hypothesis`, with `finish_reason` and `truncated`.

Source and reference text are included in every row, so each shard stands alone.
Benchmark licences apply to those texts: FLORES+ CC-BY-SA-4.0, Tatoeba CC-BY,
NTREX and WMT per their dataset cards.

## Method

X→EN · greedy decoding · seed 0 · thinking off · output budget 3× source tokens.
COMET-22 `Unbabel/wmt22-comet-da` · MetricX-24 hybrid XL (reference-based) · GlotLID v4 · chrF++ (sacrebleu).
Volume-weighted scores weight each language by its corpus rows.
