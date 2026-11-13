"""Plan and execute translation runs for one model on one or more benchmarks.

Planning is pure and runs anywhere. Execution needs a backend, so it runs on Grid5000 nodes.
Outputs are cached by content hash and written as JSON lines; failure flags are computed here.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from xx_to_en.backends import Backend, GenerationResult, PromptInput
from xx_to_en.benchmarks import Segment
from xx_to_en.cache import OutputCache
from xx_to_en.config import ModelConfig
from xx_to_en.failures import detect
from xx_to_en.inventory import Inventory
from xx_to_en.languages import LanguageMap
from xx_to_en.lengths import max_new_tokens
from xx_to_en.prompts import PROMPT_VERSION, IsoLookup, make_prompt


@dataclass(frozen=True)
class WorkItem:
    benchmark: str
    corpus_lang: str
    bench_lang: str


@dataclass(frozen=True)
class Decoding:
    """Greedy decoding; the seed is passed to the engine. Budget = ratio x source tokens, clamped."""

    seed: int = 0
    ratio: float = 3.0
    floor: int = 32
    ceiling: int = 2048


@dataclass(frozen=True)
class Translation:
    segment: Segment
    hypothesis: str
    output_tokens: int
    cached: bool
    raw_text: str = ""
    finish_reason: str = ""
    prompt_sha256: str = ""
    max_new_tokens: int = 0


def plan_work(
    benchmarks: Iterable[str],
    corpus_langs: Iterable[str],
    inventory: Inventory,
    language_map: LanguageMap,
    supported: frozenset[str] | None = None,
) -> list[WorkItem]:
    """Benchmark x language pairs that exist, are mapped, and are supported by the model.

    ``supported`` is the model's official language set in corpus codes. ``None`` means no restriction
    and is only used for planning, never for real runs (the caller must pass the official list).
    """
    items = []
    languages = sorted(set(corpus_langs))
    for benchmark in benchmarks:
        available = inventory.available(benchmark)
        for corpus_lang in languages:
            if supported is not None and corpus_lang not in supported:
                continue
            bench_lang = language_map.benchmark_code(corpus_lang, benchmark)
            if bench_lang and bench_lang in available:
                items.append(WorkItem(benchmark, corpus_lang, bench_lang))
    return items


def _budget(backend: Backend, source: str, decoding: Decoding) -> int:
    return max_new_tokens(
        backend.count_tokens(source), ratio=decoding.ratio, floor=decoding.floor, ceiling=decoding.ceiling
    )


def _cache_key(model: ModelConfig, template: str, decoding: Decoding, segment: Segment, budget: int) -> str:
    return OutputCache.key(
        model=model.hf_repo,
        prompt_template=template,
        prompt_version=PROMPT_VERSION,
        seed=decoding.seed,
        greedy=True,
        budget=budget,
        benchmark=segment.benchmark,
        bench_lang=segment.bench_lang,
        segment_id=segment.segment_id,
        source=segment.source,
    )


def translate(
    segments: Sequence[Segment],
    *,
    model: ModelConfig,
    backend: Backend,
    cache: OutputCache,
    decoding: Decoding = Decoding(),
    iso: IsoLookup | None = None,
) -> list[Translation]:
    """Translate segments, reusing cached outputs and generating the rest in one batch."""
    plan = []
    for segment in segments:
        budget = _budget(backend, segment.source, decoding)
        key = _cache_key(model, model.prompt, decoding, segment, budget)
        prompt = make_prompt(model.prompt, segment, backend, iso)
        plan.append((segment, budget, key, prompt, cache.get(key)))
    pending = [(segment, budget, key, prompt) for segment, budget, key, prompt, hit in plan if hit is None]
    generated = _generate(pending, backend)
    results = []
    for segment, budget, key, prompt, hit in plan:
        prompt_sha = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        if hit is not None:
            results.append(
                Translation(
                    segment,
                    hit,
                    output_tokens=0,
                    cached=True,
                    prompt_sha256=prompt_sha,
                    max_new_tokens=budget,
                )
            )
            continue
        output = generated[key]
        cache.put(key, output.text)
        results.append(
            Translation(
                segment,
                output.text,
                output_tokens=output.output_tokens,
                cached=False,
                raw_text=output.raw_text,
                finish_reason=output.finish_reason,
                prompt_sha256=prompt_sha,
                max_new_tokens=budget,
            )
        )
    return results


def _generate(
    pending: Sequence[tuple[Segment, int, str, PromptInput]], backend: Backend
) -> dict[str, GenerationResult]:
    if not pending:
        return {}
    outputs = backend.generate(
        [prompt for _, _, _, prompt in pending], [budget for _, budget, _, _ in pending]
    )
    return {key: output for (_, _, key, _), output in zip(pending, outputs, strict=True)}


def failure_counts(translations: Sequence[Translation]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in translations:
        flags = detect(item.segment.source, item.hypothesis)
        for name in ("empty", "repetition_loop", "length_ratio_outlier", "untranslated_copy"):
            if getattr(flags, name):
                counts[name] += 1
    return dict(counts)


def write_jsonl(path: Path, translations: Sequence[Translation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in translations:
            record = {
                "benchmark": item.segment.benchmark,
                "corpus_lang": item.segment.corpus_lang,
                "bench_lang": item.segment.bench_lang,
                "segment_id": item.segment.segment_id,
                "source": item.segment.source,
                "reference": item.segment.reference,
                "hypothesis": item.hypothesis,
                "output_tokens": item.output_tokens,
                "cached": item.cached,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def throughput(translations: Sequence[Translation], seconds: float) -> dict[str, float]:
    tokens = sum(item.output_tokens for item in translations)
    return {
        "segments": float(len(translations)),
        "output_tokens": float(tokens),
        "seconds": seconds,
        "output_tokens_per_second": tokens / seconds if seconds > 0 else 0.0,
    }


def run_work_item(
    item: WorkItem,
    segments: Sequence[Segment],
    *,
    model: ModelConfig,
    backend: Backend,
    cache: OutputCache,
    out_dir: Path,
    decoding: Decoding = Decoding(),
    iso: IsoLookup | None = None,
) -> dict[str, object]:
    """Translate one benchmark x language pair, write its JSONL, and return a summary record."""
    started = time.perf_counter()
    translations = translate(segments, model=model, backend=backend, cache=cache, decoding=decoding, iso=iso)
    elapsed = time.perf_counter() - started
    path = out_dir / model.name / item.benchmark / f"{item.corpus_lang}.jsonl"
    write_jsonl(path, translations)
    summary: dict[str, object] = {"item": asdict(item), "path": str(path)}
    summary.update(throughput(translations, elapsed))
    summary["failures"] = failure_counts(translations)
    return summary
