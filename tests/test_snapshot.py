import json
import urllib.error
from email.message import Message

from xx_to_en import snapshot


class _Body:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_token_prefers_the_environment_then_the_token_file(tmp_path, monkeypatch):
    token_file = tmp_path / "token"
    token_file.write_text(" from-file \n", encoding="utf-8")
    monkeypatch.setattr(snapshot, "TOKEN_FILE", token_file)
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert snapshot._token() == "from-env"
    monkeypatch.delenv("HF_TOKEN")
    assert snapshot._token() == "from-file"
    monkeypatch.setattr(snapshot, "TOKEN_FILE", tmp_path / "missing")
    assert snapshot._token() is None


def test_get_json_sends_the_bearer_token_when_there_is_one(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["auth"] = request.get_header("Authorization")
        seen["timeout"] = timeout
        return _Body({"ok": True})

    monkeypatch.setattr(snapshot, "_token", lambda: "t")
    monkeypatch.setattr(snapshot.urllib.request, "urlopen", fake_urlopen)
    assert snapshot._get_json("https://example.test") == {"ok": True}
    assert seen == {"auth": "Bearer t", "timeout": 60}
    monkeypatch.setattr(snapshot, "_token", lambda: None)
    snapshot._get_json("https://example.test")
    assert seen["auth"] is None


def _route(monkeypatch, responses):
    def fake_get_json(url):
        for fragment, payload in responses.items():
            if fragment in url:
                return payload
        raise AssertionError(url)

    monkeypatch.setattr(snapshot, "_get_json", fake_get_json)


def test_listings_read_the_hub_responses(monkeypatch):
    _route(
        monkeypatch,
        {
            "tree/main/devtest": [{"path": "devtest/fra_Latn.jsonl"}, {"path": "devtest/README.md"}],
            "splits?dataset=": {"splits": [{"config": "fra-en"}, {"config": "deu-en"}, {"config": "fra-en"}]},
            "size?dataset=": {"size": {"configs": [{"config": "fra_Latn", "num_rows": "7"}]}},
        },
    )
    assert snapshot.repo_listing("r", "devtest", ".jsonl") == ["fra_Latn"]
    assert snapshot.dataset_configs("r") == ["deu-en", "fra-en"]
    assert snapshot.config_row_counts("r") == {"fra_Latn": 7}
    assert snapshot.wmt_source_codes("r") == ["deu", "fra"]


def test_tatoeba_codes_keep_only_languages_paired_with_english(monkeypatch):
    monkeypatch.setattr(
        snapshot,
        "repo_listing",
        lambda repo, folder, suffix: [
            "tatoeba-test.eng-fra",
            "tatoeba-test.deu-eng",
            "tatoeba-test.eng-eng",
            "tatoeba-test.fra-deu",
            "other",
        ],
    )
    assert snapshot.tatoeba_non_english_codes() == ["deu", "fra"]


def test_try_reports_http_errors_as_a_status():
    assert snapshot._try(lambda: ["a"]) == {"status": "ok", "languages": ["a"]}

    def missing():
        raise urllib.error.HTTPError("https://example.test", 404, "not found", Message(), None)

    assert snapshot._try(missing) == {"status": "http 404", "languages": []}


def test_build_inventory_collects_every_benchmark(monkeypatch):
    _route(
        monkeypatch,
        {
            "/size?": {"size": {"configs": [{"config": "fra_Latn", "num_rows": "3"}]}},
            "/splits?": {"splits": [{"config": "fra-en"}]},
            "/tree/main/data/test": [{"path": "data/test/tatoeba-test.eng-fra.tsv"}],
            "/tree/main/devtest": [{"path": "devtest/fra_Latn.jsonl"}],
        },
    )
    inventory = snapshot.build_inventory()
    assert inventory["corpus"] == {"repo": snapshot.CORPUS_REPO, "config_rows": {"fra_Latn": 3}}
    assert inventory["benchmarks"]["flores_plus_devtest"]["languages"] == ["fra_Latn"]
    assert inventory["benchmarks"]["ntrex_128"]["languages"] == ["fra-en"]
    assert inventory["benchmarks"]["wmt22_xen"]["languages"] == ["fra"]
    assert inventory["benchmarks"]["tatoeba_xen"]["languages"] == ["fra"]
    assert inventory["snapshot_date"]


def test_run_writes_the_inventory_file(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "ROOT", tmp_path)
    monkeypatch.setattr(snapshot, "OUT", tmp_path / "data" / "inventory.json")
    monkeypatch.setattr(snapshot, "build_inventory", lambda: {"snapshot_date": "2026-01-01"})
    assert snapshot.run() == "wrote data/inventory.json"
    written = json.loads((tmp_path / "data" / "inventory.json").read_text(encoding="utf-8"))
    assert written == {"snapshot_date": "2026-01-01"}
