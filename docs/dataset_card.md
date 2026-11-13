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
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated 2026-11-13 15:18 CET

## Progress

| Model | FLORES+ | NTREX-128 | WMT22 | WMT23 | Tatoeba |
|---|---|---|---|---|---|
| gemma-3-27b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-26b-a4b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-31b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-e4b-it | █████░ 10/11 | ███░░░ 5/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| hy-mt2-30b-a3b | ░░░░░░ 0/5 | ░░░░░░ 0/4 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/4 |
| hy-mt2-7b | ░░░░░░ 0/5 | ░░░░░░ 0/4 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/4 |
| lfm3-1.7b | — | — | — | — | — |
| madlad400-10b-mt | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| madlad400-3b-mt | ██░░░░ 4/11 | ░░░░░░ 0/10 | ██████ 1/1 | ░░░░░░ 0/1 | █░░░░░ 1/8 |
| nllb-200-3.3b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| plamo-3-translate-2b | ████░░ 4/6 | █░░░░░ 1/6 | ██████ 1/1 | ██████ 1/1 | ██░░░░ 2/5 |
| plamo-3-translate-8b | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| qwen3.8-27b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| tiny-aya-global | ░░░░░░ 0/8 | ░░░░░░ 0/8 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/6 |
| translategemma-12b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| translategemma-27b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |
| translategemma-4b-it | ░░░░░░ 0/6 | ░░░░░░ 0/6 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/5 |

Each cell: shards generated / total. A shard is 8 languages of one benchmark.

## Results

Overall chrF++ uses the 32 languages every scored model covers.

| Model | chrF++ ↑ | Volume-weighted chrF++ ↑ | COMET-22 ↑ | MetricX-24 ↓ | Failures | Segments |
|---|---:|---:|---:|---:|---:|---:|
| madlad400-3b-mt | 60.5 | 64.3 | 0.859 | 3.46 | 3.3% | 32,384 |

<details><summary>Best model per language</summary>

| Language | Model | chrF++ |
|---|---|---:|
| afr_Latn | madlad400-3b-mt | 75.1 |
| als_Latn | madlad400-3b-mt | 64.7 |
| amh_Ethi | madlad400-3b-mt | 55.7 |
| arb_Arab | madlad400-3b-mt | 64.3 |
| azj_Latn | madlad400-3b-mt | 40.4 |
| bel_Cyrl | madlad400-3b-mt | 39.4 |
| ben_Beng | madlad400-3b-mt | 58.3 |
| bul_Cyrl | madlad400-3b-mt | 65.3 |
| cat_Latn | madlad400-3b-mt | 69.2 |
| ceb_Latn | madlad400-3b-mt | 50.3 |
| ces_Latn | madlad400-3b-mt | 64.1 |
| cmn_Hani | madlad400-3b-mt | 55.6 |
| cym_Latn | madlad400-3b-mt | 74.3 |
| dan_Latn | madlad400-3b-mt | 70.3 |
| deu_Latn | madlad400-3b-mt | 67.4 |
| ekk_Latn | madlad400-3b-mt | 62.0 |
| ell_Grek | madlad400-3b-mt | 61.0 |
| epo_Latn | madlad400-3b-mt | 65.5 |
| eus_Latn | madlad400-3b-mt | 54.2 |
| fas_Arab | madlad400-3b-mt | 60.0 |
| fin_Latn | madlad400-3b-mt | 59.8 |
| fra_Latn | madlad400-3b-mt | 67.9 |
| gla_Latn | madlad400-3b-mt | 57.0 |
| gle_Latn | madlad400-3b-mt | 65.0 |
| glg_Latn | madlad400-3b-mt | 66.5 |
| guj_Gujr | madlad400-3b-mt | 62.3 |
| hau_Latn | madlad400-3b-mt | 52.3 |
| heb_Hebr | madlad400-3b-mt | 65.0 |
| hin_Deva | madlad400-3b-mt | 62.7 |
| hun_Latn | madlad400-3b-mt | 61.2 |
| hye_Armn | madlad400-3b-mt | 52.0 |
| ibo_Latn | madlad400-3b-mt | 46.0 |

</details>

## Files

- `results/<model>/<benchmark>/<shard>.parquet`: hypotheses and token counts
- `scores/<model>/<benchmark>/<shard>.parquet`: per-segment scores and failure flags
- `manifests/<model>/job-<id>.json`: full model config and run settings for each job

Columns: `source` is the corpus sentence, `reference` is the ground-truth English from the benchmark, and
`hypothesis` is the generated English. Each row also carries its provenance: `model_revision`,
`prompt_template`, `prompt_version`, `prompt_sha256`,
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
