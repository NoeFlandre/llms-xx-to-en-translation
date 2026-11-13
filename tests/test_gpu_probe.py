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
