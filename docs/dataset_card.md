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
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated 2026-11-13 13:41 CET

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
| madlad400-3b-mt | █░░░░░ 1/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| nllb-200-3.3b | █░░░░░ 2/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| plamo-3-translate-2b | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| plamo-3-translate-8b | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| qwen3.8-27b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| tiny-aya-global | ██░░░░ 3/8 | ░░░░░░ 0/8 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/6 |
| translategemma-12b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| translategemma-27b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| translategemma-4b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |

Each cell: shards generated / total. A shard is 8 languages of one benchmark.

## Results

_No scores yet. Hypotheses are generated first; scores follow on bf16 GPUs._



## Files

- `results/<model>/<benchmark>/<shard>.jsonl`: hypotheses and token counts
- `scores/<model>/<benchmark>/<shard>.jsonl`: per-segment scores and failure flags

Source and reference text are not copied: join on `(benchmark, bench_lang, segment_id)`
with the public benchmark.

## Method

X→EN · greedy decoding · seed 0 · thinking off · output budget 3× source tokens.
COMET-22 `Unbabel/wmt22-comet-da` · MetricX-24 hybrid XL (reference-based) · GlotLID v4 · chrF++ (sacrebleu).
Volume-weighted scores weight each language by its corpus rows.
