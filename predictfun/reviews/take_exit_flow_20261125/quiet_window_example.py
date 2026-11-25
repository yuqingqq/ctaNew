"""Typical take 1 (flow page): when Binance crossed the UP edge on the stream vs the 50 ms tick that fired, and whether the 2 s quiet
window would pass if it looked before the market's first tick. The frozen rule (take_replay.triggers) counts quiet ticks from the
market's first grid tick only. Reads the v3 Binance change cache (allow-listed days; example day 20261102) and flow_data.json.
Output (write-once): quiet_window_example.json beside this file.
Usage: PF_SLOT=claude predictfun/research/run_light.sh flow_quiet_example -- python3 predictfun/reviews/take_exit_flow_20261125/quiet_window_example.py
"""
import json
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'take_exit_replay_20261125'))
import other_days as OD  # noqa: E402

OUT = H / 'quiet_window_example.json'
assert not OUT.exists(), f'{OUT.name} exists (write-once)'
d = json.loads((H / 'flow_data.json').read_text())
ex = d['examples'][0]
rule = d['rule']
assert ex['name'] == 'typical' and ex['day'] == '20261102', 'example changed'
t0, edge, b_in, quiet, step = ex['start_abs_ms'], rule['edge_up_bps'], rule['bn_in_ms'], rule['quiet_ms'], rule['step_ms']
trig = next(e for e in ex['events'] if e['kind'] == 'trigger')
take = next(e for e in ex['events'] if e['kind'] == 'take' and e['gid'] == trig['gid'])

k, m = OD.seg_arrays('B')
ta = (k + b_in).astype(float)                                    # time an update is usable (replay convention)


def r250(x):
    j0, j1 = np.searchsorted(ta, x, 'right') - 1, np.searchsorted(ta, x - 250, 'right') - 1
    return float(1e4 * np.log(m[j0] / m[j1]))


# r250 is piecewise constant; it can change only at an update's usable time or 250 ms after it
lo, hi = t0 - quiet - 300, t0 + trig['t']
brk = np.unique(np.r_[ta[(ta > lo) & (ta <= hi)], ta[(ta + 250 > lo) & (ta + 250 <= hi)] + 250])
cross = next(float(x) for x in brk if x > t0 and r250(x) >= edge)
win = [float(x) for x in brk if cross - quiet <= x < cross] + [cross - quiet]
on_before = sorted(x for x in win if r250(x) >= edge)
grid_on = [round(r250(t0 + g), 6) for g in np.arange(0, trig['t'] + step, step)]
res = dict(
    asof='2026-11-25', market=ex['mid'], day=ex['day'], edge_up_bps=edge, quiet_ms=quiet,
    stream_cross_ms=cross - t0, grid_fire_ms=trig['t'], earlier_by_ms=trig['t'] - (cross - t0),
    ioc_landed_ms=take['t'], take_delay_ms=rule['take_delay_ms'],
    grid_r250_up_from_first_tick=grid_on,
    stream_on_in_prior_2s_ms=[x - t0 for x in on_before[:3]], stream_on_max_bps=max((r250(x) for x in on_before), default=None),
)
res['readings'] = dict(
    grid_quiet_passes=all(v < edge for v in grid_on[:-1]) and grid_on[-1] >= edge,
    stream_quiet_passes=not on_before,
    stream_fires_before_grid=res['earlier_by_ms'] > 0,
)
OUT.write_text(json.dumps(res, indent=1) + '\n')
print(json.dumps(res))
