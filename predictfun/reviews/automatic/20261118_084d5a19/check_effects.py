"""Bounded synthetic propagation check for K8 -> K5S and fill-notice bounds."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET = '084d5a1921460a9a74ba47bef8bb41ff38921676'
root = Path(sys.argv[1]).resolve()
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == TARGET
sys.path.insert(0, str(root / 'predictfun/research'))
import pf_keep_pull_exec_windows as X
np, pd = X.np, X.pd
base = 1_800_000_000_000.
K = pd.DataFrame(dict(run=['r', 'r'], cid=['one', 'two'], place=[base, base],
                     upper=base + np.array([40., 900.]), t_set=base + np.array([30., 800.]),
                     M=[base / 1000, base / 1000], notice=[base + 1000, base + 1000],
                     signed=[base + 1, base + 1], accepted=[base + 30, base + 30],
                     req=[np.nan, base + 200], ack=[np.nan, base + 900],
                     ucancel=[np.nan, np.nan], cdone=[np.nan, np.nan], thr=[np.nan, np.nan]))
R, ml, mh, _ = X.settlement_checks(K, dict(lo=0., hi=0., exists=True))
assert R['consistent']
P, result = X.order_path(K, ml, mh)
p = P[P.cid == 'two'].iloc[0]
assert p.cancel_lo_set == 570 and p.cancel_hi == 700 and p.cancel_status_set == 'CONTRADICTED'
# Two possible worlds have the exact same observed placement, acceptance,
# cancellation request/acknowledgment, status second and UUID time.
# Each has nonnegative creation lag and effect-before-acknowledgment.
worlds = [dict(name='constant_lag_positive', match=780., uuid=800., lag=20., effect=800.),
          dict(name='variable_lag_counterexample', match=100., uuid=800., lag=700., effect=250.)]
for w in worlds:
    assert 1. <= 30. <= w['match'] < w['effect'] <= 900.
    assert 200. <= w['effect'] and w['uuid'] - w['match'] == w['lag'] >= 0
    assert np.floor(w['match'] / 1000.) == np.floor(w['uuid'] / 1000.) == 0
    w['cancel_latency_ms'] = w['effect'] - 200.
    w['notice_delay_ms'] = 1000. - w['match']
assert worlds[0]['cancel_latency_ms'] > 100 and worlds[1]['cancel_latency_ms'] <= 100
assert 190 <= worlds[0]['notice_delay_ms'] <= 230
assert worlds[1]['notice_delay_ms'] > 230
# On these same calibration rows, ST's no-false-cancel check is guaranteed
# at the minimum clock offset by the common-d fit: t_set - d_hi <= upper.
# For cancelled legs upper <= ack, so the earliest ST lower endpoint cannot
# certify a known fill was avoided; passing that check is not semantic proof.
assert np.all(K.t_set.values - R['d']['hi'] <= K.upper.values)
cancelled = np.isfinite(K.ack.values)
assert np.all(K.upper.values[cancelled] <= K.ack.values[cancelled])
evidence = dict(target=TARGET, asof=datetime.now(timezone.utc).isoformat(), synthetic=True,
                inferred_second_order=dict(cancel_interval_ms=[570, 700], status=p.cancel_status_set,
                                           notice_interval_ms=[190, 230]), compatible_worlds=worlds,
                positive_control=True, variable_lag_false_inference_reproduced=True,
                same_sample_st_gate_follows_from_common_fit=True)
(Path(__file__).parent / 'effects.json').write_text(json.dumps(evidence, indent=2) + '\n')
print(json.dumps(evidence, indent=2))
