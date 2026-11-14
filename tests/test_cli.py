import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from xx_to_en import backends, cli, metric_runtime, results_store, runner, sources, worker
from xx_to_en.cli import check_configs, main
from xx_to_en.languages import corpus_languages
from xx_to_en.plan import Shard, build_shards
from xx_to_en.runner import WorkItem

ROOT = Path(__file__).resolve().parents[1]
MODEL = "translategemma-4b-it"
BENCHMARK = "flores_plus_devtest"


def test_check_configs_reports_counts():
    message = check_configs(ROOT / "configs")
    assert message == "ok: 20 models, 5 benchmarks, languages.yaml"


def test_coverage_command_writes_matrix(tmp_path, capsys):
    out = tmp_path / "coverage.md"
    code = main(
        [
            "coverage",
            "--configs",
            str(ROOT / "configs"),
            "--inventory",
            str(ROOT / "data" / "benchmark_inventory.json"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("# Language x benchmark coverage")
    assert "wrote" in capsys.readouterr().out


def test_check_configs_command_returns_zero(capsys):
    assert main(["check-configs", "--root", str(ROOT / "configs")]) == 0
    assert "ok:" in capsys.readouterr().out


def test_unknown_command_exits(capsys):
    with pytest.raises(SystemExit):
        main(["nope"])


def test_module_entry_point_runs_check_configs(monkeypatch):
    import runpy

    monkeypatch.setattr("sys.argv", ["xx2en", "check-configs", "--root", str(ROOT / "configs")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("xx_to_en.__main__", run_name="__main__", alter_sys=True)
    assert exit_info.value.code == 0


def test_all_value_means_every_corpus_language():
    from xx_to_en.cli import Project

    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    gemma = project.model("gemma-4-31b-it")
    supported = project.supported_languages(gemma)
    assert "fra_Latn" in supported and "eng_Latn" not in supported
    assert len(supported) == 94


def test_every_model_runs_every_corpus_language():
    from xx_to_en.cli import Project

    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    corpus = set(corpus_languages(project.inventory.corpus_rows))
    assert project.supported_languages(project.model("translategemma-4b-it")) == corpus
    assert project.supported_languages(project.model("nllb-200-3.3b")) == corpus


def test_disabled_model_is_refused():
    from xx_to_en.cli import Project

    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    with pytest.raises(SystemExit):
        project.supported_languages(project.model("lfm3-1.7b"))


def test_every_corpus_language_has_a_prompt_code_except_cebuano():
    from xx_to_en.cli import Project

    project = Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")
    corpus = set(corpus_languages(project.inventory.corpus_rows))
    missing = {code for code in corpus if project.language_map.prompt_code(code) is None}
    assert missing == {"ceb_Latn"}


@pytest.fixture(scope="module")
def project() -> cli.Project:
    return cli.Project(ROOT / "configs", ROOT / "data" / "benchmark_inventory.json")


def _parse(*argv: str) -> argparse.Namespace:
    return cli.build_parser().parse_args(list(argv))


def _keys(shards: list[Shard]) -> list[tuple[str, int]]:
    return [(shard.benchmark, shard.index) for shard in shards]


class _FakeHfApi:
    def model_info(self, repo_id: str) -> SimpleNamespace:
        return SimpleNamespace(sha="abc123")


class _EveryFilePresent:
    def __contains__(self, path: object) -> bool:
        return True


def test_select_shards_filters_benchmark_and_job_residue():
    shards = [Shard("m", "a", index, ("fra_Latn",)) for index in range(4)]
    shards += [Shard("m", "b", index, ("fra_Latn",)) for index in range(2)]
    assert _keys(cli.select_shards(shards, "", 0, 2)) == [("a", 0), ("a", 2), ("b", 0)]
    assert _keys(cli.select_shards(shards, "a", 1, 2)) == [("a", 1), ("a", 3)]
    assert cli.select_shards(shards, "missing", 0, 1) == []


def test_remaining_budget_is_clamped_at_zero():
    assert cli.remaining_budget_s(1.0, time.monotonic() - 3600) == 0.0
    assert 59.0 < cli.remaining_budget_s(1.0, time.monotonic()) <= 60.0


@pytest.mark.parametrize(
    ("langs", "expected"),
    [(None, ["fra_Latn", "deu_Latn"]), (["deu_Latn"], ["deu_Latn"]), (["xxx_Nope"], [])],
)
def test_select_langs_keeps_requested_languages(langs, expected):
    items = [WorkItem(BENCHMARK, "fra_Latn", "fr"), WorkItem(BENCHMARK, "deu_Latn", "de")]
    assert [item.corpus_lang for item in cli._select_langs(items, langs)] == expected


def test_make_backend_builds_the_named_backend(project, monkeypatch):
    def fake_vllm(model, *, seed, dtype):
        return ("vllm", model.name, seed, dtype)

    def fake_transformers(model, *, seed, force_target_lang, dtype):
        return ("transformers", model.name, seed, force_target_lang, dtype)

    monkeypatch.setattr(backends, "VllmBackend", fake_vllm)
    monkeypatch.setattr(backends, "TransformersBackend", fake_transformers)
    assert cli.make_backend(project.model(MODEL), "vllm", seed=3, dtype="float16") == (
        "vllm",
        MODEL,
        3,
        "float16",
    )
    assert cli.make_backend(project.model("nllb-200-3.3b"), "transformers", seed=0) == (
        "transformers",
        "nllb-200-3.3b",
        0,
        True,
        None,
    )
    assert cli.make_backend(project.model("madlad400-3b-mt"), "transformers", seed=0)[3] is False
    with pytest.raises(SystemExit, match="unknown backend"):
        cli.make_backend(project.model(MODEL), "tensorrt", seed=0)


def test_run_command_translates_selected_items_and_prints_summaries(project, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_backend", lambda model, kind, *, seed: f"{kind}-backend")
    monkeypatch.setattr(sources, "load_segments", lambda benchmark, corpus, bench: list(range(5)))
    seen: list[dict] = []

    def fake_run_work_item(item, segments, *, model, backend, cache, out_dir, decoding, iso):
        seen.append(
            {
                "lang": item.corpus_lang,
                "segments": list(segments),
                "backend": backend,
                "out_dir": out_dir,
                "cache": cache.root,
            }
        )
        return {"corpus_lang": item.corpus_lang, "segments": len(segments)}

    monkeypatch.setattr(runner, "run_work_item", fake_run_work_item)
    argv = ["run", "--model", MODEL, "--benchmarks", BENCHMARK, "--langs", "fra_Latn", "--limit", "2"]
    argv += ["--backend", "vllm", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")]

    assert cli.run_command(project, _parse(*argv)) == f"{len(seen)} work items done"
    assert seen
    assert all(entry["lang"] == "fra_Latn" for entry in seen)
    assert all(entry["segments"] == [0, 1] for entry in seen)
    assert all(entry["backend"] == "vllm-backend" for entry in seen)
    assert all(entry["out_dir"] == tmp_path / "out" for entry in seen)
    assert all(entry["cache"] == tmp_path / "cache" for entry in seen)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert printed == [{"corpus_lang": "fra_Latn", "segments": 2}] * len(seen)


def test_run_command_refuses_an_empty_selection(project):
    args = _parse("run", "--model", MODEL, "--benchmarks", BENCHMARK, "--langs", "xxx_Nope")
    with pytest.raises(SystemExit, match="nothing to run"):
        cli.run_command(project, args)


def test_run_and_work_refuse_models_not_accepted(project, tmp_path, monkeypatch):
    model = project.model(MODEL)
    monkeypatch.setitem(project.models, MODEL, model.model_copy(update={"access": "not_accepted"}))
    with pytest.raises(SystemExit, match="access not accepted"):
        cli.run_command(project, _parse("run", "--model", MODEL))
    with pytest.raises(SystemExit, match="access not accepted"):
        cli.work_command(
            project, _parse("work", "--model", MODEL, "--results-repo", "r", "--work", str(tmp_path))
        )


def _work_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    argv = ["work", "--model", MODEL, "--results-repo", "org/results", "--work", str(tmp_path)]
    return _parse(*argv, "--benchmark", BENCHMARK, *extra)


def _benchmark_shards(project: cli.Project) -> list[Shard]:
    shards = build_shards(MODEL, project.plan(project.model(MODEL)))
    return [shard for shard in shards if shard.benchmark == BENCHMARK]


def test_work_command_generates_uploads_each_shard_then_the_manifest(project, tmp_path, monkeypatch):
    monkeypatch.delenv("OAR_JOB_ID", raising=False)  # the manifest name must not depend on the caller's job
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(results_store, "repo_files", lambda repo_id: set())
    monkeypatch.setattr(cli, "make_backend", lambda model, kind, *, seed, dtype: "backend")
    generated: list[Shard] = []
    monkeypatch.setattr(worker, "generate_shard", lambda shard, **kwargs: generated.append(shard))
    uploads: list[str] = []
    monkeypatch.setattr(results_store, "upload", lambda repo_id, local, path: uploads.append(path))

    message = cli.work_command(project, _work_args(tmp_path))

    assert generated
    assert message == f"{MODEL}: generated {len(generated)} of {len(generated)} pending shards"
    assert uploads[:-1] == [results_store.shard_path(shard) for shard in generated]
    assert uploads[-1] == f"manifests/{MODEL}/job-local.json"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shards"] == uploads[:-1]


def test_work_command_without_budget_generates_nothing(project, tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHfApi)
    monkeypatch.setattr(results_store, "repo_files", lambda repo_id: set())
    monkeypatch.setattr(cli, "make_backend", lambda model, kind, *, seed, dtype: "backend")
    generated: list[Shard] = []
    monkeypatch.setattr(worker, "generate_shard", lambda shard, **kwargs: generated.append(shard))
    uploads: list[str] = []
    monkeypatch.setattr(results_store, "upload", lambda repo_id, local, path: uploads.append(path))

    total = len(_benchmark_shards(project))
    message = cli.work_command(project, _work_args(tmp_path, "--budget-min", "0"))

    assert message == f"{MODEL}: generated 0 of {total} pending shards"
    assert generated == [] and uploads == []
    assert not (tmp_path / "manifest.json").exists()


def test_work_command_reports_nothing_pending(project, tmp_path, monkeypatch):
    monkeypatch.setattr(results_store, "repo_files", lambda repo_id: _EveryFilePresent())
    assert cli.work_command(project, _work_args(tmp_path)) == f"{MODEL}: nothing pending"


def _score_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    argv = ["score", "--model", MODEL, "--results-repo", "org/results", "--work", str(tmp_path)]
    return _parse(*argv, "--benchmark", BENCHMARK, *extra)


def test_score_command_scores_shards_and_downloads_only_missing_hypotheses(project, tmp_path, monkeypatch):
    shards = _benchmark_shards(project)
    local = tmp_path / "hyp" / results_store.shard_path(shards[0])
    local.parent.mkdir(parents=True)
    local.write_text("", encoding="utf-8")
    files = {results_store.shard_path(shard) for shard in shards}
    monkeypatch.setattr(results_store, "repo_files", lambda repo_id: files)
    downloads: list[str] = []
    monkeypatch.setattr(results_store, "download", lambda repo_id, path, target: downloads.append(path))
    uploads: list[str] = []
    monkeypatch.setattr(results_store, "upload", lambda repo_id, local_file, path: uploads.append(path))
    scored: list[Shard] = []
    monkeypatch.setattr(worker, "score_shard", lambda hyp, out, **kwargs: scored.append(kwargs["shard"]))
    scorer_settings: dict = {}

    class FakeScorers:
        def __init__(self, **kwargs):
            scorer_settings.update(kwargs)

    monkeypatch.setattr(metric_runtime, "Scorers", FakeScorers)

    message = cli.score_command(project, _score_args(tmp_path, "--comet-batch", "7"))

    assert message == f"{MODEL}: scored {len(shards)} of {len(shards)} pending shards"
    assert scored == shards
    assert downloads == [results_store.shard_path(shard) for shard in shards[1:]]
    assert uploads == [results_store.score_path(shard) for shard in shards]
    assert scorer_settings["comet_batch"] == 7
    assert scorer_settings["work"] == tmp_path / "metricx"


def test_score_command_reports_nothing_to_score(project, tmp_path, monkeypatch):
    monkeypatch.setattr(results_store, "repo_files", lambda repo_id: set())
    assert cli.score_command(project, _score_args(tmp_path)) == f"{MODEL}: nothing to score"


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["check-configs", "--root", "configs"], "check_configs"),
        (["coverage"], "write_coverage"),
        (["plan", "--model", MODEL], "plan_command"),
        (["run", "--model", MODEL], "run_command"),
        (["work", "--model", MODEL, "--results-repo", "r", "--work", "w"], "work_command"),
        (["score", "--model", MODEL, "--results-repo", "r", "--work", "w"], "score_command"),
    ],
)
def test_main_routes_each_subcommand_to_its_handler(argv, handler, monkeypatch, capsys):
    calls: list[str] = []

    def recorder(name: str):
        def fake(*args, **kwargs) -> str:
            calls.append(name)
            return f"{name} done"

        return fake

    for name in ("check_configs", "write_coverage", "plan_command", "run_command", "work_command"):
        monkeypatch.setattr(cli, name, recorder(name))
    monkeypatch.setattr(cli, "score_command", recorder("score_command"))
    monkeypatch.setattr(cli, "Project", lambda configs, inventory: "project")

    assert main(argv) == 0
    assert calls == [handler]
    assert capsys.readouterr().out == f"{handler} done\n"
