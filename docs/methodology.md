# Methodology

## Question

Which open model should translate the GeoReSeT corpus into English? The corpus is about 71.5M non-English sentences
(about 13 words each) that are dense with place names and proper nouns.

## Benchmarks

Benchmarks are only the existing public test sets listed in the README. No new test sets are created.

- Native-origin sources: WMT22 and WMT23 X→EN, and Tatoeba (mixed origin).
- English-origin sources: FLORES+ devtest and NTREX-128. These are reported separately.

## Protocol (planned)

Every model runs every corpus language covered by a benchmark, so per-language results compare directly; the official language lists in the configs are reference only.
- Official prompt format for each model family; greedy decoding; thinking disabled.
- Fixed seeds; `max_new_tokens` tied to source length (see `xx_to_en.lengths`).
- Every output is cached under a content hash of its inputs (model, prompt, decoding, source text).
- Contamination risk is noted where a model card reports training on a benchmark.

## Metrics

- chrF++ (implemented, `xx_to_en.metrics`).
- COMET-22 and MetricX-24 (planned).
- Paired bootstrap 95% CIs on per-segment scores (`xx_to_en.stats`).
- Failure rates (`xx_to_en.failures`): wrong output language (GlotLID, planned), empty output, repetition loops,
  length-ratio outliers, untranslated copies.

## Aggregation

- Per-language and per-benchmark tables.
- Volume-weighted overall score, weighted by corpus rows per language (`xx_to_en.report`).
- Per-language winner.

## Code quality gates

- Tests: pytest with coverage.
- Lint and types: ruff and ty.
- CRAP score below 6 for every function (`scripts/crap.py`).
- Mutation testing with mutmut (configured; not yet run).
