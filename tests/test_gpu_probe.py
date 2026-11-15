import io
import json
import sys

from xx_to_en import gpu_probe
from xx_to_en.gpu_probe import count_free

SAMPLE = """network_address: g-1.site
state: Alive
properties: cluster=granite, gpu_model=RTX PRO 6000, drain=NO, maintenance=NO

network_address: g-2.site
state: Alive
jobs: 123
properties: cluster=granite, gpu_model=RTX PRO 6000, drain=NO, maintenance=NO

network_address: g-3.site
state: Dead
properties: cluster=gres, gpu_model=L40S, drain=NO, maintenance=NO

network_address: g-4.site
state: Alive
properties: cluster=gres, gpu_model=L40S, drain=YES, maintenance=NO

network_address: g-5.site
state: Alive
properties: cluster=gres, gpu_model=L40S, drain=NO, maintenance=NO
"""


def test_counts_only_alive_idle_undrained_gpus():
    counts = count_free(SAMPLE)
    assert counts["granite|RTX PRO 6000"] == 1
    assert counts["gres|L40S"] == 1


def test_run_reports_free_gpus_read_from_stdin(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(SAMPLE))
    out = json.loads(gpu_probe.run("nancy"))
    assert out == {"site": "nancy", "free": dict(sorted(count_free(SAMPLE).items()))}
