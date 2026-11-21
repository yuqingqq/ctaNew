"""Timing-only probe (no feature values printed): one builder call over every grid step of one complete Oct 24 market."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_ladder_rest as M
REPO = Path('/home/yuqing/ctaNew'); man = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text()); day = '20261024'; m = 3697902
x = [y for y in man['selected'] if y['day'] == day and int(y['m']) == m][0]
t0 = time.time(); prs, fair, depth, feat_fn, allw, opens = M.load_day(day, [x], man); tl = time.time() - t0
W_ = prs[m]; path, st, info = M.market_path(W_, float(x['ws']), opens[m][1]); t_abs0, end = float(info['t_start']), float(info['end_rel'])
ch = {}
for s in (0, 1):
    tg = np.asarray(W_[s]['tg'], float); lo = int(np.searchsorted(tg, t_abs0 - 1e-6)); hi = int(np.searchsorted(tg, t_abs0 + end + 1e-6))
    ch[s] = list(range(lo, hi))
t1 = time.time(); F = feat_fn(W_, fair.get(m), depth, ch); tf = time.time() - t1
print(json.dumps(dict(load_s=round(tl, 1), status=st, end_rel=end, steps={s: len(v) for s, v in ch.items()}, rows={s: len(F[s]) for s in (0, 1)}, builder_s=round(tf, 1),
                      ms_per_step=round(1e3 * tf / max(1, sum(len(v) for v in ch.values())), 3), grid_step_ms=float(np.median(np.diff(np.asarray(W_[0]['tg'], float)))))))
