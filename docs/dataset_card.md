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
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated 2026-11-13 17:37 CET

## Progress

| Model | FLORES+ | NTREX-128 | WMT22 | WMT23 | Tatoeba |
|---|---|---|---|---|---|
| gemma-3-27b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-26b-a4b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-31b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-e4b-it | █████░ 10/11 | ███░░░ 5/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| hy-mt2-30b-a3b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| hy-mt2-7b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| lfm3-1.7b | — | — | — | — | — |
| madlad400-10b-mt | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| madlad400-3b-mt | █████░ 9/11 | █░░░░░ 2/10 | ██████ 1/1 | ██████ 1/1 | ███░░░ 4/8 |
| nllb-200-3.3b | █░░░░░ 1/11 | ░░░░░░ 0/10 | ██████ 1/1 | ██████ 1/1 | ░░░░░░ 0/8 |
| plamo-3-translate-2b | ████░░ 7/11 | ██░░░░ 3/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ███░░░ 4/8 |
| plamo-3-translate-8b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| qwen3.8-27b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| tiny-aya-global | ██████ 11/11 | █████░ 8/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | █░░░░░ 1/8 |
| translategemma-12b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| translategemma-27b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| translategemma-4b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |

Each cell: shards generated / total. A shard is 8 languages of one benchmark.

## Results

Overall chrF++ uses the 0 languages every scored model covers.

| Model | chrF++ ↑ | Volume-weighted chrF++ ↑ | COMET-22 ↑ | MetricX-24 ↓ | Failures | Segments |
|---|---:|---:|---:|---:|---:|---:|
| madlad400-3b-mt | 58.8 | 0.0 | 0.855 | 3.74 | 3.9% | 88,840 |
| nllb-200-3.3b | 44.7 | 0.0 | 0.710 | 5.25 | 21.9% | 8,323 |
| tiny-aya-global | 51.1 | 0.0 | 0.821 | 5.45 | 1.6% | 15,976 |

<details><summary>Best model per language</summary>

| Language | Model | chrF++ |
|---|---|---:|
| afr_Latn | madlad400-3b-mt | 73.8 |
| als_Latn | madlad400-3b-mt | 63.5 |
| amh_Ethi | madlad400-3b-mt | 50.0 |
| arb_Arab | madlad400-3b-mt | 60.1 |
| azj_Latn | tiny-aya-global | 43.9 |
| bel_Cyrl | tiny-aya-global | 49.8 |
| ben_Beng | madlad400-3b-mt | 57.1 |
| bul_Cyrl | madlad400-3b-mt | 61.3 |
| cat_Latn | madlad400-3b-mt | 69.2 |
| ceb_Latn | madlad400-3b-mt | 50.4 |
| ces_Latn | nllb-200-3.3b | 65.6 |
| ckb_Arab | madlad400-3b-mt | 49.7 |
| cmn_Hani | madlad400-3b-mt | 55.7 |
| cym_Latn | madlad400-3b-mt | 74.4 |
| dan_Latn | madlad400-3b-mt | 70.3 |
| deu_Latn | madlad400-3b-mt | 67.4 |
| ekk_Latn | madlad400-3b-mt | 62.0 |
| ell_Grek | madlad400-3b-mt | 61.1 |
| epo_Latn | madlad400-3b-mt | 65.4 |
| eus_Latn | madlad400-3b-mt | 54.3 |
| fas_Arab | madlad400-3b-mt | 60.1 |
| fin_Latn | madlad400-3b-mt | 59.9 |
| fra_Latn | madlad400-3b-mt | 67.9 |
| gla_Latn | madlad400-3b-mt | 57.1 |
| gle_Latn | madlad400-3b-mt | 65.0 |
| glg_Latn | madlad400-3b-mt | 66.5 |
| guj_Gujr | madlad400-3b-mt | 62.4 |
| hau_Latn | madlad400-3b-mt | 52.3 |
| heb_Hebr | madlad400-3b-mt | 65.0 |
| hin_Deva | madlad400-3b-mt | 62.7 |
| hun_Latn | madlad400-3b-mt | 61.2 |
| hye_Armn | madlad400-3b-mt | 52.1 |
| ibo_Latn | madlad400-3b-mt | 46.0 |
| ind_Latn | madlad400-3b-mt | 65.4 |
| isl_Latn | madlad400-3b-mt | 58.3 |
| ita_Latn | madlad400-3b-mt | 60.8 |
| jav_Latn | madlad400-3b-mt | 42.7 |
| jpn_Jpan | madlad400-3b-mt | 54.2 |
| kan_Knda | madlad400-3b-mt | 56.3 |
| kat_Geor | madlad400-3b-mt | 38.9 |
| kaz_Cyrl | madlad400-3b-mt | 54.2 |
| khk_Cyrl | madlad400-3b-mt | 52.2 |
| khm_Khmr | madlad400-3b-mt | 57.6 |
| kir_Cyrl | madlad400-3b-mt | 40.3 |
| kmr_Latn | madlad400-3b-mt | 51.2 |
| kor_Hang | madlad400-3b-mt | 56.0 |
| lit_Latn | madlad400-3b-mt | 58.5 |
| lvs_Latn | madlad400-3b-mt | 62.4 |
| mal_Mlym | madlad400-3b-mt | 54.3 |
| mar_Deva | madlad400-3b-mt | 58.1 |
| mkd_Cyrl | madlad400-3b-mt | 66.0 |
| mlt_Latn | madlad400-3b-mt | 75.1 |
| mya_Mymr | madlad400-3b-mt | 53.3 |
| nld_Latn | madlad400-3b-mt | 58.7 |
| nno_Latn | madlad400-3b-mt | 67.5 |
| nob_Latn | madlad400-3b-mt | 74.3 |
| npi_Deva | madlad400-3b-mt | 61.1 |
| pan_Guru | madlad400-3b-mt | 61.4 |
| pbt_Arab | madlad400-3b-mt | 54.5 |
| plt_Latn | madlad400-3b-mt | 52.8 |
| pol_Latn | madlad400-3b-mt | 56.1 |
| por_Latn | madlad400-3b-mt | 71.0 |
| ron_Latn | madlad400-3b-mt | 67.5 |
| rus_Cyrl | madlad400-3b-mt | 61.4 |
| sin_Sinh | madlad400-3b-mt | 56.2 |
| slk_Latn | madlad400-3b-mt | 64.1 |
| slv_Latn | madlad400-3b-mt | 60.6 |
| spa_Latn | madlad400-3b-mt | 58.6 |
| srp_Cyrl | madlad400-3b-mt | 65.1 |
| swe_Latn | madlad400-3b-mt | 69.4 |
| tam_Taml | madlad400-3b-mt | 53.7 |
| tel_Telu | madlad400-3b-mt | 60.2 |

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
