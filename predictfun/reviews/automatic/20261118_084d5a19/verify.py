"""Pinned-source review, synthetic falsifier and saved-table recomputation only.
Run through PF_SLOT=codex run_light.sh. No tape replay or training.
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

TARGET = '084d5a1921460a9a74ba47bef8bb41ff38921676'
root = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == TARGET
sys.path.insert(0, str(root / 'predictfun/research'))
import pf_keep_pull_exec_windows as X
import pf_keep_pull_scenarios as S
import pf_settlement_times as ST
np, pd, KP = X.np, X.pd, X.KP
day = '20261023'
data = Path('/home/yuqing/ctaNew/data/predict_fun/research')
committed = root / 'predictfun/reviews/keep_pull_settlement_day20261023'
ledger = data / ('keep_pull_exec_windows_v4_20261118_' + day)
known = data / 'keep_pull_exec_windows_known_v3_20261118'
scenarios = data / 'keep_pull_scenarios_v3_20261118'
settlement = data / 'settlement_times_20261118' / day
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
for nm, base, saved_name, module in [
    ('exec_windows.json', ledger, 'exec_windows.json', 'pf_keep_pull_exec_windows.py'),
    ('known.json', known, 'known.json', 'pf_keep_pull_exec_windows.py'),
    ('scenarios_20261023.json', scenarios, 'scenarios_20261023.json', 'pf_keep_pull_scenarios.py'),
    ('settlement.json', settlement, 'settlement_20261023.json', 'pf_settlement_times.py'),
]:
    check_hash(base / nm, sha(committed / saved_name))
    j = load(base / nm)
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
for name, h in dr['settlement_inputs'].items():
    check_hash(name, h)
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
    ST.selftest()
R['synthetic_instruments'] = 'three complete pinned selftests passed'
print('Three pinned synthetic selftests passed.', flush=True)

# K8's stated premise allows an event-specific creation lag, even with an
# exactly known, constant clock offset. A feasible common d is not a bound
# on those event-specific lags. Both possible realities below share inputs.
base = 1_800_000_000_000.
K2 = pd.DataFrame(dict(run=['r', 'r'], place=[base, base],
                      upper=base + np.array([40., 900.]),
                      t_set=base + np.array([30., 800.]),
                      M=[base / 1000, base / 1000],
                      notice=[base + 1000, base + 1000]))
off = dict(lo=0., hi=0., exists=True, includes_zero=True)
fit, mlo, mhi, _ = X.settlement_checks(K2, off)
assert fit['consistent'] and fit['d'] == dict(lo=-10., hi=30., exists=True)
assert fit['lag_bound_ms'] == 30.
positive_matches = K2.t_set.values - 20.
variable_matches = base + np.array([20., 100.])
def admissible(matches):
    return bool(np.all((matches >= K2.place.values) & (matches <= K2.upper.values)
                       & (matches <= K2.t_set.values)
                       & (np.floor(matches / 1000.) == K2.M.values)))
assert admissible(positive_matches) and admissible(variable_matches)
assert np.all((positive_matches >= mlo) & (positive_matches <= mhi))
assert not np.all((variable_matches >= mlo) & (variable_matches <= mhi))
assert (K2.t_set.values - variable_matches).tolist() == [10., 700.]
bad = K2.copy()
bad.loc[1, 'place'] = base + 950.
bad.loc[1, 'upper'] = base + 970.
rejected, _, _, _ = X.settlement_checks(bad, off)
assert not rejected['consistent']
R['variable_lag_counterexample'] = dict(
    synthetic=True, common_clock_offset_ms=0, place_ms=[0, 0], upper_ms=[40, 900],
    uuid_times_ms=[30, 800], common_d_fit_ms=[-10, 30], claimed_lag_bound_ms=30,
    positive_constant_lag_matches_ms=(positive_matches - base).tolist(),
    true_variable_lag_matches_ms=(variable_matches - base).tolist(),
    true_variable_lags_ms=[10, 700],
    claimed_match_windows_ms=np.stack([mlo - base, mhi - base], axis=1).tolist(),
    both_worlds_satisfy_observations_and_nonnegative_lag=True,
    variable_lag_world_excluded_by_claimed_bounds=True,
    incompatible_common_offset_control_refused=True)
print('Reproduced K8 variable-lag counterexample; constant-lag and refusal controls passed.', flush=True)

EV = pd.read_parquet(ledger / 'ledger_events.parquet')
FL = pd.read_parquet(ledger / 'ledger_fills.parquet')
PA = pd.read_parquet(ledger / 'ledger_fill_events.parquet')
OL = pd.read_parquet(ledger / 'ledger_orders.parquet')
J = PA.merge(EV, on=['day', 'group', 'e'], validate='many_to_one').merge(
    FL[['fid', 't0', 't_replay']], on='fid', validate='many_to_one')
saved = load(ledger / 'exec_windows.json')['R']
causes = {}
for k in ('S1', 'S2', 'S3', 'ST'):
    st = J[k + '_st']
    count = J.assign(bad=st >= 2).groupby('fid').bad.agg(['sum', 'count']).reindex(FL.fid)
    status = np.where(count['sum'] == 0, 'none', np.where(count['sum'] == count['count'], 'all', 'some'))
    assert np.array_equal(status, FL['contra_' + k])
    for name in X.CST:
        mask = status == name
        assert int(mask.sum()) == saved['V3'][k]['fills'][name]
        np.testing.assert_allclose(FL.value.abs().values[mask].sum(), saved['V3'][k]['gross'][name], atol=1e-8)
    causes[k] = dict(all_contradictory=int((status == 'all').sum()))
    for c, n in [(2, 'evidence_cap'), (3, 'leg_intersection')]:
        causes[k][n] = int(J.assign(bad=st == c).groupby('fid').bad.all().sum())
assert len(OL) == 30466 and OL.status.value_counts().to_dict() == saved['status']
R['ledger'] = dict(counted_orders=len(OL), status=saved['status'], fills=len(FL), events=len(EV),
                   contradiction_causes=causes, contradiction_accounting_reconciled=True,
                   saved_st_lag_ms=saved['st_lag_ms'])

# Recompute saved economic values, not the full queue/tape replay.
with tempfile.TemporaryDirectory(prefix='pf-review-084d5a19-') as tmp:
    temp = Path(tmp)
    (temp / day).symlink_to(scenarios / day, target_is_directory=True)
    with (out / 'scenario_recompute.txt').open('w') as log, contextlib.redirect_stdout(log):
        actual, readings = S.report(out=temp, days=[day])
    expected = load(committed / 'scenarios_20261023.json')
    assert actual == expected['scenarios'] and readings == expected['readings']
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
R['scenarios'] = dict(rows_exactly_reproduced=len(actual), corrupt_hash_refused=True,
    result_asof=expected['asof'], population=dr['stat'], readings=readings,
    metrics={k: {x: v[x] for x in ('E', 'score', 'null', 'KEEP', 'fills', 'eligible')} for k, v in actual.items()})

# Identity-safe saved case tables contain all original K and KE inputs;
# avoid loading the full Nov 8 source tape again.
K = pd.read_parquet(known / 'known_cases.parquet')
KE = pd.read_parquet(known / 'known_case_rows.parquet')
expected_known = load(committed / 'known.json')
known_r, known_read = X.known_checks(K, expected_known['R']['K0_ref'], KE)
known_r = {k: v for k, v in known_r.items() if not k.startswith('_')}
assert known_r == expected_known['R'] and known_read == expected_known['readings']
R['known'] = dict(asof=expected_known['asof'], legs=len(K), unique_orders=known_r['orders'],
    candidate_rows=len(KE), anchored_legs=int(np.isfinite(K.Z).sum()), read=known_read,
    K5=known_r['K5'], K7=known_r['K7'], K8=known_r['K8'])
R['verified_hashes'] = checked
(out / 'receipt.json').write_text(json.dumps(R, indent=2) + '\n')
print(json.dumps({k: R[k] for k in ['target', 'asof', 'variable_lag_counterexample', 'ledger']}, indent=2))
print('All 20 saved scenario rows and known-case readings reproduced.', flush=True)
