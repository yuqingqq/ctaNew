"""Bounded checks of cause labels and acknowledgment-bound semantics; no tape replay."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'predictfun/research'))
import pf_keep_pull_exec_windows as X
np, pd = X.np, X.pd
prior = json.loads((out / 'receipt.json').read_text())
data = Path('/home/yuqing/ctaNew/data/predict_fun/research')
ledger = data / 'keep_pull_exec_windows_20261118_20261023'
for name in ('ledger_events.parquet', 'ledger_fills.parquet', 'ledger_fill_events.parquet'):
    assert hashlib.sha256((ledger / name).read_bytes()).hexdigest() == prior['verified_hashes'][str(ledger / name)]
EV = pd.read_parquet(ledger / 'ledger_events.parquet')
FL = pd.read_parquet(ledger / 'ledger_fills.parquet')
P = pd.read_parquet(ledger / 'ledger_fill_events.parquet')
J = P.merge(EV[['e', 'S1_st', 'S2_st']], on='e', validate='many_to_one')
res = dict(target=prior['target'], asof=datetime.now(timezone.utc).isoformat(),
           market_day='20261023', fills=len(FL), events=len(EV), causes={})
for k in ('S1', 'S2'):
    res['causes'][k] = {}
    for name, flag in [('any_contradiction', J[k + '_st'] >= 2),
                       ('evidence_cap_status', J[k + '_st'] == 2),
                       ('leg_intersection_status', J[k + '_st'] == 3)]:
        group = J.assign(flag=flag).groupby('fid').flag.agg(['all', 'any']).reindex(FL.fid)
        all_ = group['all'].values
        res['causes'][k][name] = dict(all_fills=int(all_.sum()), some_fills=int((group['any'] & ~group['all']).sum()),
                                      all_gross=float(FL.value.abs().values[all_].sum()),
                                      events=int(((EV[k + '_st'] >= 2) if name == 'any_contradiction' else
                                                  (EV[k + '_st'] == (2 if name == 'evidence_cap_status' else 3))).sum()))
assert res['causes']['S1']['any_contradiction']['all_fills'] == 748
assert res['causes']['S1']['evidence_cap_status']['all_fills'] < 748

K = pd.read_parquet(data / 'keep_pull_exec_windows_known_20261118/known_cases.parquet')
U = K.drop_duplicates(['run', 'cid'])
a = U.accepted - U.place
res['activation_lower_endpoint'] = dict(orders=len(U), assumed_min_ms=X.ACT_RANGE[0],
    acknowledgment_upper_below_assumed_min=int((a < X.ACT_RANGE[0]).sum()),
    minimum_upper_ms=float(a.min()))
# Positive and known-bad interval examples: K5 reads COVERED in both, although
# acceptance observed at +1 ms makes activation in [25,100] ms impossible.
synthetic = K.copy()
synthetic['ack'] = synthetic.req + 20.
synthetic['upper'] = np.fmin(synthetic.notice, synthetic.ack)
ref = prior['known']['read']
refs = dict(rest_floor_nonoverlap=39, matched_floor_nonoverlap=0)
reads = {}
for label, delay in [('compatible_upper', 40.), ('lower_endpoint_contradiction', 1.)]:
    synthetic['accepted'] = synthetic.place + delay
    reads[label] = X.known_checks(synthetic, refs, draws=200)[1]['K5 DECLARED RANGES vs MEASURED ORDER PATH']
assert reads == dict(compatible_upper='COVERED', lower_endpoint_contradiction='COVERED')
res['synthetic_K5'] = dict(activation_range=list(X.ACT_RANGE), cancel_ack_delay=20.,
    compatible_acceptance_ack_delay=40., contradictory_acceptance_ack_delay=1., readings=reads)
(out / 'diagnostics.json').write_text(json.dumps(res, indent=2) + '\n')
print(json.dumps(res, indent=2))
