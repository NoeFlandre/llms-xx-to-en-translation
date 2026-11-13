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
[Code](https://github.com/NoeFlandre/llms-xx-to-en-translation) · updated 2026-11-13 20:59 CET

## Progress

| Model | FLORES+ | NTREX-128 | WMT22 | WMT23 | Tatoeba |
|---|---|---|---|---|---|
| gemma-3-27b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-26b-a4b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-31b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| gemma-4-e4b-it | ██████ 11/11 | ██████ 10/10 | ██████ 1/1 | ██████ 1/1 | ███░░░ 4/8 |
| hy-mt2-30b-a3b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| hy-mt2-7b | ██████ 11/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ██████ 1/1 | ░░░░░░ 0/8 |
| lfm3-1.7b | — | — | — | — | — |
| madlad400-10b-mt | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| madlad400-3b-mt | ██████ 11/11 | ██░░░░ 4/10 | ██████ 1/1 | ██████ 1/1 | ████░░ 6/8 |
| nllb-200-3.3b | ████░░ 8/11 | █░░░░░ 2/10 | ██████ 1/1 | ██████ 1/1 | █░░░░░ 1/8 |
| plamo-3-translate-2b | ██████ 11/11 | █████░ 8/10 | ██████ 1/1 | ██████ 1/1 | ██████ 8/8 |
| plamo-3-translate-8b | ███░░░ 5/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ██████ 1/1 | ░░░░░░ 0/8 |
| qwen3.8-27b | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| tiny-aya-global | ██████ 11/11 | ██████ 10/10 | ██████ 1/1 | ██████ 1/1 | ██████ 8/8 |
| translategemma-12b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| translategemma-27b-it | ░░░░░░ 0/11 | ░░░░░░ 0/10 | ░░░░░░ 0/1 | ░░░░░░ 0/1 | ░░░░░░ 0/8 |
| translategemma-4b-it | █░░░░░ 1/11 | █░░░░░ 1/10 | ██████ 1/1 | ██████ 1/1 | ░░░░░░ 0/8 |

Each cell: shards generated / total. A shard is 8 languages of one benchmark.

## Results

Overall chrF++ uses the 4 languages every scored model covers.

| Model | chrF++ ↑ | Volume-weighted chrF++ ↑ | COMET-22 ↑ | MetricX-24 ↓ | Failures | Segments |
|---|---:|---:|---:|---:|---:|---:|
| tiny-aya-global | 55.7 | 61.6 | 0.840 | 4.51 | 8.5% | 327,389 |
| gemma-4-e4b-it | 55.2 | 61.5 | 0.836 | 4.59 | 4.6% | 136,768 |
| hy-mt2-7b | 54.3 | 60.0 | 0.841 | 2.97 | 32.7% | 10,976 |
| plamo-3-translate-8b | 48.8 | 59.5 | 0.788 | 5.24 | 26.3% | 19,072 |
| madlad400-3b-mt | 57.7 | 58.7 | 0.851 | 3.70 | 8.4% | 137,271 |
| plamo-3-translate-2b | 47.2 | 57.2 | 0.769 | 6.60 | 21.8% | 32,611 |
| translategemma-4b-it | 51.2 | 52.1 | 0.833 | 2.94 | 28.1% | 19,299 |
| nllb-200-3.3b | 43.6 | 48.4 | 0.711 | 5.40 | 28.9% | 19,299 |

<details><summary>Best model per language</summary>

| Language | Model | chrF++ |
|---|---|---:|
| afr_Latn | madlad400-3b-mt | 73.8 |
| als_Latn | madlad400-3b-mt | 63.5 |
| amh_Ethi | madlad400-3b-mt | 50.0 |
| arb_Arab | plamo-3-translate-8b | 64.6 |
| azj_Latn | plamo-3-translate-8b | 51.1 |
| bel_Cyrl | tiny-aya-global | 52.4 |
| ben_Beng | tiny-aya-global | 58.6 |
| bul_Cyrl | tiny-aya-global | 65.3 |
| cat_Latn | plamo-3-translate-2b | 66.4 |
| ceb_Latn | plamo-3-translate-2b | 63.8 |
| ces_Latn | nllb-200-3.3b | 65.6 |
| ckb_Arab | madlad400-3b-mt | 48.7 |
| cmn_Hani | tiny-aya-global | 53.3 |
| cym_Latn | madlad400-3b-mt | 67.0 |
| dan_Latn | tiny-aya-global | 69.4 |
| deu_Latn | tiny-aya-global | 64.1 |
| ekk_Latn | madlad400-3b-mt | 59.3 |
| ell_Grek | tiny-aya-global | 74.1 |
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
| heb_Hebr | madlad400-3b-mt | 63.5 |
| hin_Deva | madlad400-3b-mt | 62.7 |
| hun_Latn | madlad400-3b-mt | 61.2 |
| hye_Armn | gemma-4-e4b-it | 54.5 |
| ibo_Latn | madlad400-3b-mt | 46.0 |
| ind_Latn | madlad400-3b-mt | 65.4 |
| isl_Latn | madlad400-3b-mt | 58.5 |
| ita_Latn | madlad400-3b-mt | 60.8 |
| jav_Latn | gemma-4-e4b-it | 54.6 |
| jpn_Jpan | gemma-4-e4b-it | 55.2 |
| kan_Knda | gemma-4-e4b-it | 56.5 |
| kat_Geor | gemma-4-e4b-it | 53.7 |
| kaz_Cyrl | gemma-4-e4b-it | 55.1 |
| khk_Cyrl | madlad400-3b-mt | 52.2 |
| khm_Khmr | madlad400-3b-mt | 57.6 |
| kir_Cyrl | gemma-4-e4b-it | 46.2 |
| kmr_Latn | madlad400-3b-mt | 51.2 |
| kor_Hang | gemma-4-e4b-it | 57.0 |
| lit_Latn | madlad400-3b-mt | 58.5 |
| lvs_Latn | madlad400-3b-mt | 62.4 |
| mal_Mlym | gemma-4-e4b-it | 59.0 |
| mar_Deva | gemma-4-e4b-it | 59.2 |
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
| rus_Cyrl | gemma-4-e4b-it | 61.1 |
| sin_Sinh | gemma-4-e4b-it | 56.8 |
| slk_Latn | madlad400-3b-mt | 64.1 |
| slv_Latn | madlad400-3b-mt | 60.6 |
| spa_Latn | madlad400-3b-mt | 58.6 |
| srp_Cyrl | madlad400-3b-mt | 65.1 |
| swe_Latn | madlad400-3b-mt | 69.4 |
| tam_Taml | gemma-4-e4b-it | 56.3 |
| tel_Telu | gemma-4-e4b-it | 60.2 |
| tgk_Cyrl | madlad400-3b-mt | 32.8 |
| tha_Thai | madlad400-3b-mt | 56.3 |
| tur_Latn | madlad400-3b-mt | 62.4 |
| ukr_Cyrl | hy-mt2-7b | 61.5 |
| urd_Arab | madlad400-3b-mt | 57.1 |
| uzn_Latn | madlad400-3b-mt | 55.7 |
| vie_Latn | madlad400-3b-mt | 60.6 |
| xho_Latn | madlad400-3b-mt | 52.4 |
| ydd_Hebr | madlad400-3b-mt | 51.2 |
| yor_Latn | tiny-aya-global | 36.2 |
| yue_Hani | madlad400-3b-mt | 53.8 |
| zsm_Latn | madlad400-3b-mt | 65.7 |
| zul_Latn | madlad400-3b-mt | 54.3 |

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
