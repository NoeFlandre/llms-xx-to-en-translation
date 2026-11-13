# llms-xx-to-en-translation

Reproducible benchmark that ranks open translation models on X→English, to find the best model for the
GeoReSeT corpus (about 71.5M non-English sentences of short, place-name-dense text from Wikipedia, websites,
Wikivoyage and OpenStreetMap descriptions).

The corpus is used only to understand what the text looks like. It is never translated.

## Status

| Part | State |
|---|---|
| Config schema and validation (17 models, 5 benchmarks) | done |
| Language × benchmark coverage matrix | done, see [docs/coverage_matrix.md](docs/coverage_matrix.md) |
| Failure checks, paired bootstrap CIs, chrF++, volume-weighted aggregation | done, unit-tested |
| Output cache (content-addressed) and generation budget | done |
| Model runner on Grid5000 (vLLM and transformers backends) | not started |
| COMET-22, MetricX-24, GlotLID wiring | not started |
| Official prompt formats and model language lists | pending (read from model cards) |
| HF results dataset and dataset card | not started |

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Core dependencies only; GPU and neural-metric dependencies are
installed on Grid5000 and are not needed here.

```bash
uv sync
uv run pytest --cov --cov-report=json:coverage.json
uv run ruff check .
uv run ty check src tests
uv run python scripts/crap.py --threshold 6
uv run xx2en check-configs
uv run xx2en coverage
```

Rebuild the benchmark inventory (metadata only, needs a Hugging Face token in `~/.cache/huggingface/token`
or `HF_TOKEN`):

```bash
python3 scripts/snapshot_benchmarks.py
```

## Benchmarks (X→EN)

| Benchmark | Origin | Used for | Notes |
|---|---|---|---|
| WMT22 X→EN | native | native-source check | cs, de, is, ru, zh |
| WMT23 X→EN | native | native-source check | de, he, is, ja, ru, uk, zh |
| Tatoeba Challenge | mixed | long tail | 100 languages; community-written pairs, direction of authorship not recorded |
| FLORES+ devtest | English-origin | core Wikipedia-domain set | sources written in English, translated out |
| NTREX-128 | English-origin | news | same caveat as FLORES+ |

WMT24 is not used at all. WMT24++ is English-source and excluded.

FLORES+ and NTREX are English-origin, so for X→EN the source is a human translation. Results on them should be
reported with and without those sets.

## Models

Seventeen candidates, one YAML each in [configs/models](configs/models). Selection criteria: stated support in the
requested families, runnable on one 96 GB GPU in bf16 (two-GPU runs excluded for now), and an accessible license.
Gated repositories are included only where the account has access. The most recent dedicated MT model found is
Hy-MT2 (2026); older MT models (Tower-Plus, Seed-X, Hunyuan-MT, EuroLLM) were not added.

Official language lists are still `pending`; the runner will refuse any model whose list is pending.

## Layout

```
configs/                one YAML per model and per benchmark, plus languages.yaml
data/                   committed benchmark inventory (metadata only)
docs/                   MkDocs site and generated coverage matrix
scripts/                snapshot_benchmarks.py (HF metadata), crap.py (CRAP score)
src/xx_to_en/           config, languages, inventory, coverage, failures, stats, metrics, cache, lengths, report, cli
tests/                  pytest suite
```

## Data and compute policy

- Corpus: read only its metadata (configs, row counts, tags). Corpus rows are never downloaded or translated.
- Throughput for the ~71.5M-sentence estimate is measured on benchmark sentences and extrapolated.
- Compute: Grid5000. Frontends are used only for job submission; jobs run on nodes. Model weights are downloaded
  per job to node-local `/tmp`, never to home directories.
- Generated outputs and metrics will be published to a Hugging Face dataset named like this repository, with a
  dataset card that records GPUs, software versions, seeds, decoding settings and failure rates.
- Nothing generated is kept on the development machine beyond small caches.

## Citing Grid'5000

Publications with results from Grid'5000 must acknowledge it, following the
[Grid'5000 acknowledgment statement](https://www.grid5000.fr/w/Grid5000:UsagePolicy).

## Licence

Not yet chosen. Outputs derived from FLORES+ and NTREX inherit CC-BY-SA-4.0. Model licenses (Gemma terms, PLaMo and
LFM "other", CC-BY-NC for NLLB and Tiny Aya) apply to the weights, not to the outputs in this repository.
