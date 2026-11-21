"""A declared COVERAGE DIAGNOSTIC of the LADDER_REST_INVENTORY_v1 statuses (added after the pilot showed most full paths incomplete; it changes no
frozen code, status or reading and selects nothing). Per selected market of one day, from the same identity-checked inputs (pf_ladder_rest
.load_day): the start (market_start), every venue frame gap > 1 s inside [start, opening + 298 s] (count, largest, first position), the last
valid mark at or before opening + 298 s (the builder's AC.valid_marks rule) and its distance to the end, and the builder's status. Writes
logs/coverage_diag_<day>.json. Usage: coverage_diag.py <day>"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_ladder_rest as M                                # noqa: E402

day = sys.argv[1]; R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_rest_20261121'); out = R / 'logs' / f'coverage_diag_{day}.json'
M.C.require(not out.exists(), 'the diagnostic exists (write-once)'); M.C.require(M.frozen(), 'not the frozen run')
man = json.loads((R / 'manifest.json').read_text()); members = [x for x in man['selected'] if x['day'] == day]
prs, fair, depth, feat_fn, allw, opens = M.load_day(day, members, man); rows = []
for x in members:
    m, ws = int(x['m']), float(x['ws']); W = prs.get(m)
    if W is None:
        rows.append(dict(m=m, status='missing_market_tape')); continue
    t0, seen, none = M.market_start(W, ws); end = ws + M.SUPPORTED_END; tf = np.asarray(W[0]['tf'], float)
    lo = t0 if t0 is not None else ws; sel = tf[(tf >= lo) & (tf <= end)]; gaps = np.diff(np.r_[sel, end]) if len(sel) else np.array([end - lo])
    big = np.flatnonzero(gaps > 1000.)
    mt, mm = M.QP.AC.valid_marks(W[0], 0); ix = int(np.searchsorted(mt, end, side='right')) - 1
    p, st = M.QP.build_path(W, t0, end) if t0 is not None else (None, 'no_start')
    rows.append(dict(m=m, status=st, start_rel_s=None if t0 is None else (t0 - ws) / 1000., gaps_gt_1s=int(len(big)), max_gap_s=float(gaps.max()) / 1000. if len(gaps) else None,
                     first_gap_rel_s=float(sel[big[0]] - ws) / 1000. if len(big) else None, last_valid_mark_rel_s=float(mt[ix] - ws) / 1000. if ix >= 0 else None,
                     end_minus_last_mark_s=float(end - mt[ix]) / 1000. if ix >= 0 else None))
out.write_text(json.dumps(dict(day=day, asof=M.C.stamp(), markets=rows), indent=1) + '\n'); print(json.dumps(rows[:6], indent=1))
