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

TARGET = '3bf8f3a9332de737514142bdf63b0b59ab272bb5'
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
committed = root / 'predictfun/reviews/keep_pull_corrected_day20261023'
ledger = data / ('keep_pull_exec_windows_v5_20261118_' + day)
known = data / 'keep_pull_exec_windows_known_v5_20261118'
scenarios = data / 'keep_pull_scenarios_v4_20261118'
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
    check_hash(root / name if name.startswith('predictfun/') else root / 'predictfun/research' / name, h)
print('Pinned sources and saved output hashes verified.', flush=True)

with (out / 'selftests.txt').open('w') as log, contextlib.redirect_stdout(log):
    X.selftest()
    S.selftest()
R['synthetic_instruments'] = 'two complete pinned selftests passed'
print('Both pinned synthetic selftests passed.', flush=True)

# The prior counterexample must remain compatible without a universal delay bound.
base = 1_800_000_000_000.
K2 = pd.DataFrame(dict(run=['r', 'r'], place=[base, base], upper=base + np.array([40., 900.]),
                      t_set=base + np.array([30., 800.]), M=[base / 1000, base / 1000], notice=[base + 1000, base + 1000]))
fit, mlo, mhi, _ = X.settlement_checks(K2, dict(lo=0., hi=0., exists=True))
assert fit['consistent'] and fit['lag_bound_ms'] is None
for actual_match in (np.array([10., 780.]), np.array([20., 100.])):
    assert np.all((mlo <= base + actual_match) & (base + actual_match <= mhi))
bad = K2.copy(); bad.loc[1, 't_set'] = base - 1.
assert not X.settlement_checks(bad, dict(lo=0., hi=0., exists=True))[0]['consistent']
R['prior_counterexample'] = dict(constant_and_variable_lag_worlds_retained=True, no_universal_bound=True, impossible_creation_refused=True)

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

# Independently reconcile the headline's population and trace its lower endpoints to actual leg inputs.
LG = pd.read_parquet(ledger / 'ledger_legs.parquet')
cf = FL.pulled.values & (FL.cls_ST_CP.values == 0)
rem = FL.replay_removed.values.astype(bool)
fallback = LG.assign(fallback=np.isfinite(LG.ST_llo.values) & ~np.isfinite(LG.t_set_ms.values)).groupby(['day', 'group']).fallback.any()
cf_events = PA[PA.fid.isin(FL.fid[cf])].merge(EV, on=['day', 'group', 'e'], validate='many_to_one')
key = pd.MultiIndex.from_frame(cf_events[['day', 'group']])
assert np.isfinite(cf_events.ST_lo.values).all() and fallback.reindex(key).all()
assert not ((LG.ST_code == 0) & np.isfinite(LG.ST_llo)).any()
R['ST_population'] = dict(
    all_pulled_cancel_first_fills=int(cf.sum()), removed_cancel_first_fills=int((cf & rem).sum()),
    kept_cancel_first_fills=int((cf & ~rem).sum()), candidate_links=len(cf_events),
    every_candidate_lower_bound_requires_unmapped_leg_fallback=True,
    removed_fills=int(rem.sum()), removed_gross=float(FL.value.abs().values[rem].sum()),
    cancel_first_removed_gross=float(FL.value.abs().values[cf & rem].sum()),
    cancel_first_share_removed_gross=float(FL.value.abs().values[cf & rem].sum()/FL.value.abs().values[rem].sum()),
    candidate_event_population=len(EV), candidate_events_unknown_lower=int(np.isneginf(EV.ST_lo).sum()),
    candidate_event_legs=len(LG), replay_vs_uuid_population=saved['replay_point_minus_t_set_ms'],
    scenario_event_population=dr['stat']['events'], ledger_asof=load(ledger/'exec_windows.json')['asof'])
# Repeat the declared comparison with explicit assertions, independent of compare.py's printed status.
old = root / 'predictfun/reviews/keep_pull_settlement_day20261023'
oldl = load(old / 'exec_windows.json')['R']
for part in ('V3', 'bounds', 'xtab', 'decisions', 'coverage', 'disagreement'):
    for k in set(oldl.get(part, {})) | set(saved.get(part, {})):
        if not k.startswith('ST'):
            assert oldl[part].get(k) == saved[part].get(k), (part, k)
olds = load(old / 'scenarios_20261023.json')['scenarios']
news = load(committed / 'scenarios_20261023.json')['scenarios']
assert set(olds) - set(news) == {'STlo|A58|L37|N100'} and not (set(news) - set(olds))
assert all(olds[n] == v for n, v in news.items() if not n.startswith('ST'))
R['comparison'] = dict(non_ST_ledger_parts_unchanged=True, non_ST_scenario_rows_unchanged=18, dropped='STlo|A58|L37|N100')

# Recompute saved economic values, not the full queue/tape replay.
with tempfile.TemporaryDirectory(prefix='pf-review-3bf8f3a9-') as tmp:
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
    K5=known_r['K5'], K7=known_r['K7'], K8=known_r['K8'], K6=known_r['K6'], K6_shared_clock=known_r['K6_shared_clock'])
assert known_read['K6 GATE S3'] == 'FAIL' and known_read['K6 GATE S3 (K8 shared clock)'] == 'PASS'
assert known_r['K6']['S3']['false_cancel_first']['some_offset'] == 3
assert known_r['K6_shared_clock']['S3']['false_cancel_first']['some_offset'] == 0
assert known_read['K6 GATE ST'] == 'UNIDENTIFIED-UPPER-BOUND-ONLY'
R['verified_hashes'] = checked
(out / 'receipt.json').write_text(json.dumps(R, indent=2) + '\n')
print(json.dumps({k: R[k] for k in ['target', 'asof', 'prior_counterexample', 'ledger', 'ST_population', 'comparison']}, indent=2))
print('All 19 saved scenario rows and known-case readings reproduced.', flush=True)
