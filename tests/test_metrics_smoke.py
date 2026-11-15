import json

from xx_to_en import metrics_smoke


def test_run_scores_the_two_toy_segments(monkeypatch, tmp_path):
    class FakeScorers:
        def __init__(self, metricx_repo, work):
            self.paths = (metricx_repo, work)

        def comet(self, rows):
            return [0.9 for _ in rows]

        def metricx(self, rows):
            return [1.5 for _ in rows]

        def glotlid(self, texts):
            return ["en" for _ in texts]

    monkeypatch.setattr(metrics_smoke, "Scorers", FakeScorers)
    monkeypatch.setenv("METRICX_REPO", str(tmp_path))
    monkeypatch.setenv("WORK", str(tmp_path / "work"))
    assert json.loads(metrics_smoke.run()) == {
        "comet": [0.9, 0.9],
        "metricx": [1.5, 1.5],
        "glotlid": ["en", "en"],
    }
