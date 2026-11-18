"""Check of 9a114cc1 (the USER's settlement-lag fix): recompute the known-order readings with the corrected K6 basis from the SAVED known v4
case tables (no tape is read, nothing is written under data/), and show where S3's known false CANCEL_FIRST fills flip between offset bases.
Run via run_light (PF_SLOT=claude): python3 predictfun/reviews/settlement_lag_fix_check_20261118/verify.py <out_dir>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/yuqing/ctaNew')
sys.path.insert(0, str(ROOT / 'predictfun/research'))
import pf_keep_pull_exec_windows as X  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
d4 = Path(X.I.K.IO) / 'keep_pull_exec_windows_known_v4_20261118'
r4 = json.loads((d4 / 'known.json').read_text())
for n, h in r4['outputs'].items():                                  # the saved inputs are the ones the v4 receipt names
    assert sha(d4 / n) == h, f'known v4 output changed since its receipt: {n}'
K = pd.read_parquet(d4 / 'known_cases.parquet')
K = K.drop(columns=[c for c in K if c.startswith(('K4_', 'K8_', 'S1_', 'S2_', 'S3_', 'ST_', 'EV_', 'M50_')) or c in ('n_candidates', 'anchor_origin', 'anchor_own_row')])
KE = pd.read_parquet(d4 / 'known_case_rows.parquet')
R, read = X.known_checks(K, r4['R']['K0_ref'], KE)
L = []
# (1) K8 is untouched by the K6 basis change: its fields must equal v4's.
for f in ('status', 'consistent', 'clock_offset', 'offset_settlement_premise', 'lag_bound_ms', 'universal_lag_status', 'notice_delay_ms'):
    a, b = json.loads(json.dumps(R['K8'][f], default=float)), r4['R']['K8'][f]
    assert a == b, f'K8.{f} differs from v4: {a} vs {b}'
L.append('K8 fields identical to known v4: status, consistent, clock_offset, offset_settlement_premise, lag_bound_ms, universal_lag_status, notice_delay_ms')
# (2) the readings: what changed against v4, by key
v4r = r4['readings']; keys = list(dict.fromkeys(list(v4r) + list(read)))
L.append('READINGS v4 -> corrected (K6 declared at K7\'s offsets; K8 shared clock and settlement offsets as separate conditional readings):')
for k in keys:
    a, b = v4r.get(k, '(absent)'), read.get(k, '(absent)')
    L.append(f"  {'CHANGED ' if a != b else '        '}{k}: {a if a == b else f'{a}  ->  {b}'}")
# (3) false CANCEL_FIRST at some offset per basis, and the three S3 legs' flip thresholds (every candidate must be after the confirmation)
k7, clk, z = R['K7'], R['K8']['clock_offset'], R['K8']['offset_settlement_premise']
bases = {'K7 (declared)': k7, 'K8 shared clock (conditional)': clk, 'K7 lo .. K8 lo only': dict(lo=k7['lo'], hi=clk['lo'], exists=True), 'zero-lag settlement (conditional)': z}
fc = {}
for nm, off in bases.items():
    o, _ = X.window_checks(K, R['_KE'], dict(lo=off['lo'], hi=off['hi'], exists=True), ('S1', 'S2', 'S3', 'EV', 'M50'))
    fc[nm] = {k: o[k]['false_cancel_first']['some_offset'] for k in o}
    L.append(f"  known fills called CANCEL_FIRST at some offset, {nm} [{off['lo']:+.1f}, {off['hi']:+.1f}] ms: {fc[nm]}")
i = R['_KE'].i.values.astype(int); ack = K.ack.values[i].astype(float); ok = np.isfinite(ack) & (R['_KE'].S3_st.values < 2)
thr = pd.Series(np.where(ok, R['_KE'].S3_lo.values - ack, -np.inf)).groupby(i).min()
legs = [int(j) for j in np.flatnonzero(R['_legs']['S3_false_cf_some_offset'])]
for j in legs:
    r = R['_KE'][R['_KE'].i == j]
    L.append(f"  S3 leg {j} (run {K.run[j]}, cid {K.cid[j]}): called avoided when the lowest offset < {thr[j]:+.1f} ms; S3 code {sorted(set(X.SEM_CODES[c] for c in r.S3_code))}")
t_n = (K.t_set.values - K.notice.values).astype(float)
L.append(f"  K8 shared-clock lower end = max(t_set - first notice) = {t_n.max():+.1f} ms (next {np.sort(t_n)[-2]:+.1f}): it binds above K7's {k7['lo']:+.1f} ms only if the "
         'UUID clock is the clock of the S3 windows (REST status seconds for these legs)')
lag_lo = np.asarray(R['K8']['known_leg_lag_ms']['lo']); lag_hi = np.asarray(R['K8']['known_leg_lag_ms']['hi'])
L.append(f"  K8 per-leg creation lag: lower bound > 0 on {int((lag_lo > 0).sum())} of {len(lag_lo)} legs; upper bounds min {lag_hi.min():.1f} / median {np.median(lag_hi):.1f} / max {lag_hi.max():.1f} ms")
# the declared predicates, computed
pred = dict(s3_fails_at_k7=fc['K7 (declared)']['S3'] > 0, s3_passes_at_k8_clock=fc['K8 shared clock (conditional)']['S3'] == 0,
            flip_from_lower_end_only=fc['K7 lo .. K8 lo only'] == fc['K7 (declared)'], k6_gate_s3_reading=read['K6 GATE S3'],
            k6_gate_s3_k8_clock_reading=read.get('K6 GATE S3 (K8 shared clock)'), s3_legs=len(legs))
L.append('PREDICATES: ' + json.dumps(pred))
(out / 'known_readings_corrected.txt').write_text('\n'.join(L) + '\n')
(out / 'known_readings_corrected.json').write_text(json.dumps(dict(asof=X.C.stamp(), inputs={str(d4 / n): h for n, h in r4['outputs'].items()}, readings=read,
                                                                   false_cancel_first_some_offset=fc, predicates=pred, source_sha256=sha(X.__file__),
                                                                   raw_tape_read=False, data_dir_written=False), indent=1, default=float))
print('\n'.join(L))
