"""Bounded review: pinned synthetic instruments plus saved consumed-day artifacts.
No training or tape replay. Run through PF_SLOT=codex run_light.sh.
"""
import contextlib
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TARGET = '1fd77004be70f436119c9eaf8079a94fe322099f'
root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == TARGET
sys.path.insert(0, str(root / 'predictfun/research'))
import pf_keep_pull_exec_windows as X
import pf_keep_pull_scenarios as S
np, pd, KP = X.np, X.pd, X.KP
day = '20261023'
data = Path('/home/yuqing/ctaNew/data/predict_fun/research')
committed = root / 'predictfun/reviews/keep_pull_scenarios_20261118_day20261023'
ledger = data / ('keep_pull_exec_windows_20261118_' + day)
known = data / 'keep_pull_exec_windows_known_20261118'
scenarios = data / 'keep_pull_scenarios_20261118'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
checked = {}
def check_hash(p, expected):
    actual = sha(p)
    assert actual == expected, f'hash mismatch: {p}'
    checked[str(p)] = actual

R = dict(target=TARGET, asof=datetime.now(timezone.utc).isoformat(),
         consumed_days=[day, '20261108'], untouched_validation_read=False,
         tape_replayed=False, training_run=False)
for nm, base in [('exec_windows.json', ledger), ('known.json', known),
                 ('scenarios_20261023.json', scenarios)]:
    check_hash(base / nm, sha(committed / nm))
    j = load(base / nm)
    module = 'pf_keep_pull_scenarios.py' if nm.startswith('scenarios') else 'pf_keep_pull_exec_windows.py'
    check_hash(root / 'predictfun/research' / module, j['source_sha256'])
    for name, h in j.get('outputs', {}).items():
        check_hash(base / name, h)
    for name, h in j.get('dependencies', {}).items():
        check_hash(root / 'predictfun/research' / name, h)
dr = load(committed / 'scenario_day_receipt.json')
check_hash(scenarios / day / 'receipt.json', sha(committed / 'scenario_day_receipt.json'))
for name, h in dr['outputs'].items():
    check_hash(scenarios / day / name, h)
for name, h in dr['dependencies'].items():
    check_hash(root / 'predictfun/research' / name, h)
kr = load(KP.OUT / day / 'receipt.json')
check_hash(KP.OUT / day / 'receipt.json', dr['keep_pull_receipt'])
for name, h in kr['outputs'].items():
    check_hash(KP.OUT / day / name, h)
for line in (committed / 'sources.sha256').read_text().splitlines():
    h, name = line.split()
    check_hash(root / 'predictfun/research' / name, h)
print('Pinned sources and saved output hashes verified.', flush=True)

with (out / 'selftests.txt').open('w') as log, contextlib.redirect_stdout(log):
    X.selftest()
    S.selftest()
R['synthetic_instruments'] = 'both complete selftests passed, including their known-bad refusals'
print('Both pinned synthetic selftests passed.', flush=True)

# Independently repeat the old three-fill counterexample against the new solver.
F = dict(val=np.array([-10., -10., 10.]), pulled=np.ones(3, bool), T=np.full(3, 5),
         s=np.zeros(3, int), endj=np.full(3, 99), lend=np.full(3, np.inf))
CW = np.zeros((1, KP.NSTEP + 2))
args = (np.array([0, 1, 2, 2]), np.array([0, 1, 0, 1]), 3, np.full(2, 100.), np.full(2, 400.),
        np.zeros(2), np.zeros(3), F, CW, KP.CANCEL)
j = X.joint_bounds(*args)
outer = X.joint_bounds(*args, cap=0)
assert [j.mn.sum(), j.mx.sum()] == [0., 10.] and set(j.kind) == {'joint'}
assert [outer.mn.sum(), outer.mx.sum()] == [-10., 20.] and set(outer.kind) == {'relaxed'}
R['prior_joint_case'] = dict(exact=[0, 10], labelled_outer=[-10, 20])

EV = pd.read_parquet(ledger / 'ledger_events.parquet')
FL = pd.read_parquet(ledger / 'ledger_fills.parquet')
PA = pd.read_parquet(ledger / 'ledger_fill_events.parquet')
OL = pd.read_parquet(ledger / 'ledger_orders.parquet')
J = PA.merge(EV, on=['day', 'group', 'e'], validate='many_to_one').merge(
    FL[['fid', 't0', 't_replay']], on='fid', validate='many_to_one')
saved = load(ledger / 'exec_windows.json')['R']
for k, bad in [('S1', J.S1_st >= 2), ('S2', J.S2_st >= 2),
               ('after_evidence', J.t0 + J.t_replay > J.EV_hi)]:
    count = J.assign(bad=bad).groupby('fid').bad.agg(['sum', 'count']).reindex(FL.fid)
    status = np.where(count['sum'] == 0, 'none', np.where(count['sum'] == count['count'], 'all', 'some'))
    assert np.array_equal(status, FL['contra_' + k])
    for name in X.CST:
        mask = status == name
        assert int(mask.sum()) == saved['V3'][k]['fills'][name]
        np.testing.assert_allclose(FL.value.abs().values[mask].sum(), saved['V3'][k]['gross'][name], atol=1e-8)
assert len(OL) == 30466 and OL.status.value_counts().to_dict() == saved['status']
assert len(FL) == 17670 and len(EV) == 50553
assert all(v['bounds'] == 'exact' and v['fills']['relaxed'] == 0 for v in saved['bounds'].values())
R['ledger'] = dict(counted_orders=len(OL), status=saved['status'], fills=len(FL), events=len(EV),
                   all_candidate_contradictions={k: saved['V3'][k]['fills'] for k in ('S1', 'S2')},
                   saved_bounds_all_labelled_exact=True, contradiction_accounting_reconciled=True)

# Recompute all reported values from the saved per-order/per-fill scenario tables.
# The actual per-tape kernel replay is intentionally not repeated.
with tempfile.TemporaryDirectory(prefix='pf-review-1fd77004-') as tmp:
    temp = Path(tmp)
    (temp / day).symlink_to(scenarios / day, target_is_directory=True)
    with (out / 'scenario_recompute.txt').open('w') as log, contextlib.redirect_stdout(log):
        actual, readings = S.report(out=temp, days=[day])
    expected = load(committed / 'scenarios_20261023.json')
    assert actual == expected['scenarios'] and readings == expected['readings']
    # Positive identity control above; a corrupt saved-table hash must refuse.
    badbase = temp / 'bad'
    (badbase / day).mkdir(parents=True)
    for name in ('orders.parquet', 'fills.parquet'):
        (badbase / day / name).symlink_to(scenarios / day / name)
    badreceipt = copy.deepcopy(dr)
    badreceipt['outputs']['fills.parquet'] = '0' * 64
    (badbase / day / 'receipt.json').write_text(json.dumps(badreceipt))
    try:
        S.report(out=badbase, days=[day])
        raise AssertionError('corrupt scenario table receipt accepted')
    except (SystemExit, ValueError) as e:
        assert 'scenario outputs changed' in str(e)
    assert not (badbase / ('scenarios_' + day + '.json')).exists()
R['scenarios'] = dict(all_15_rows_exactly_reproduced=True, corrupt_hash_refused=True,
    result_asof=expected['asof'], population=dr['stat'], readings=readings,
    metrics={k: {x: v[x] for x in ('E', 'score', 'null', 'KEEP', 'fills', 'eligible')} for k, v in actual.items()})

K, ref = X.known_frame()
expected_known = load(committed / 'known.json')
known_r, known_read = X.known_checks(K, ref)
assert known_r == expected_known['R'] and known_read == expected_known['readings']
finite = np.isfinite(K.Z)
rows = K.loc[finite].copy()
rows['miss_S1'] = X.miss(rows.Z.values - 250., rows.Z.values + 250., rows)
rows['miss_S2'] = X.miss(rows.Z.values - 600., rows.Z.values + 600., rows)
rows['anchor_before_place_ms'] = rows.place - rows.Z
rows['anchor_after_upper_ms'] = rows.Z - rows.upper
rows.loc[rows.miss_S1 | rows.miss_S2, ['run', 'cid', 'miss_S1', 'miss_S2',
    'anchor_before_place_ms', 'anchor_after_upper_ms']].to_csv(out / 'known_misses.csv', index=False)
U = K.drop_duplicates(['run', 'cid'])
uact = (U.accepted - U.place).dropna()
ucancel = (U.ack - U.req).dropna()
R['known'] = dict(asof=expected_known['asof'], legs=len(K), unique_orders=len(U), anchored_legs=int(finite.sum()),
    no_anchor=int((~finite).sum()), read=known_read, K4=known_r['K4'], K5_leg_weighted=known_r['K5'],
    unique_order_ack_bounds=dict(activation_n=len(uact), activation_p95=float(uact.quantile(.95)),
                                cancel_n=len(ucancel), cancel_p95=float(ucancel.quantile(.95))))
# An acknowledgment upper bound does not identify the latent effect time:
# both alternatives satisfy exactly the same request/ack observations.
assert all(0 <= t <= 5033 for t in (20, 2000))
assert (20 <= 100) != (2000 <= 100)
R['ack_bound_counterexample'] = dict(request_ms=0, ack_ms=5033, compatible_effect_ms=[20, 2000],
                                    same_observations_do_not_identify_100ms_coverage=True)
R['verified_hashes'] = checked
(out / 'receipt.json').write_text(json.dumps(R, indent=2) + '\n')
print(json.dumps({k: R[k] for k in ['target', 'asof', 'prior_joint_case', 'ledger', 'known', 'ack_bound_counterexample']}, indent=2))
print('All saved scenario metrics and known-order readings reproduced; ledger contradiction totals reconciled.', flush=True)
