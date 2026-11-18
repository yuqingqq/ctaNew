"""Bounded correction checks and superseding known-order receipt. No tape replay/training.
Usage through PF_SLOT=codex run_light.sh: verify.py synthetic|known|mutants
"""
import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).resolve().parent
SRC = OUT.parents[1] / 'research'
sys.path.insert(0, str(SRC))
import pf_keep_pull_exec_windows as X
import pf_keep_pull_scenarios as S
import pf_keep_pull_exec_windows_mutants as XM
import pf_keep_pull_scenarios_mutants as SM
np, pd = X.np, X.pd
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())


def synthetic():
    X.selftest()
    S.selftest()
    for module in (XM, SM):
        for i in range(len(module.MUT)):
            module.source(i)
    # Independent planted true worlds with variable lag and known admissible clocks.
    rng = np.random.default_rng(20261118)
    for _ in range(200):
        delta = float(rng.uniform(-250, 250))
        match = 1_800_000_000_000. + rng.uniform(0, 10000, 8)
        lag = rng.uniform(0, 2000, 8)
        t = match + delta + lag
        place = match - rng.uniform(1, 500, 8)
        notice = t - delta + rng.uniform(1, 500, 8)
        upper = np.minimum(match + rng.uniform(1, 500, 8), notice)
        K = pd.DataFrame(dict(run='r', place=place, upper=upper, notice=notice, t_set=t, M=np.floor(t / 1000)))
        r, lo, hi, _ = X.settlement_checks(K, dict(lo=delta - 100, hi=delta + 100, exists=True))
        assert r['consistent'] and r['lag_bound_ms'] is None
        assert np.all((lo <= match) & (match <= hi))
        assert np.all((np.array(r['known_leg_lag_ms']['lo']) <= lag) & (lag <= np.array(r['known_leg_lag_ms']['hi'])))
        assert r['clock_offset']['lo'] <= delta <= r['clock_offset']['hi']
    # Exercise the full ledger path with upper-only windows, not only explicit finite synthetic lags.
    O, Fl, L, P = X._synthetic(np.random.default_rng(X.KP.SEED))
    v1, e, score, _ = X._v1_expect(O, Fl, L)
    with tempfile.TemporaryDirectory(dir=str(X.I.K.IO)) as tmp:
        temp = Path(tmp)
        (temp / 'known.json').write_text(json.dumps(dict(R=dict(K8=dict(model=X.SETTLEMENT_MODEL, consistent=True, lag_bound_ms=None)))))
        old = X.OUT_K
        X.OUT_K = temp
        try:
            ts = pd.DataFrame(dict(t_set_ms=np.where(L.tk.values >= 0, L.anc.values - 20., np.nan), set_status=np.where(L.tk.values >= 0, 0, 1)))
            r, read = X.run(out=temp / 'ledger', src=X._Src(L[X.LEG_COLS]), tables=(O, Fl),
                           rows_of=lambda d: L[X.RAW_COLS + ['tk', 'tk_status']].reset_index(drop=True),
                           env_of=lambda d: dict(h=600., q01=-900., q99=1050., inputs={}), v1=v1, days=['20990101'], expect=(e, score), tol=1e-6,
                           sem_of=lambda d, rows: (X.sem_windows(rows, P), {}), set_of_=lambda d: (ts, {}))
            assert r['st_lag_ms'] is None and r['st_lag_role'] == 'UNIDENTIFIED'
            ev = pd.read_parquet(temp / 'ledger' / 'ledger_events.parquet')
            assert (ev.ST_lo_type == 'unknown').any()
        finally:
            X.OUT_K = old
        # A legacy scenario receipt must refuse before loading any table or writing a report.
        (temp / 'bad' / '20261023').mkdir(parents=True)
        (temp / 'bad' / '20261023' / 'receipt.json').write_text(json.dumps(dict(outputs={})))
        try:
            S.report(out=temp / 'bad', days=['20261023'])
            raise AssertionError('legacy scenario receipt accepted')
        except ValueError as err:
            assert 'legacy settlement timing receipt' in str(err)
        assert not (temp / 'bad' / 'scenarios_20261023.json').exists()
    X.C.dump(OUT / 'synthetic.json', dict(asof=X.C.stamp(), model=X.SETTLEMENT_MODEL,
        complete_selftests=2, variable_lag_worlds=200, legs_per_world=8,
        full_upper_only_ledger_rehearsal=True, legacy_scenario_refused=True,
        mutation_targets_valid=dict(exec_windows=len(XM.MUT)-1, scenarios=len(SM.MUT)-1),
        source_sha256={p.name: sha(p) for p in (SRC / 'pf_keep_pull_exec_windows.py', SRC / 'pf_keep_pull_scenarios.py')}))


def known():
    old = Path(X.I.K.IO) / 'keep_pull_exec_windows_known_v3_20261118'
    r0 = load(old / 'known.json')
    for n, h in r0['outputs'].items():
        assert sha(old / n) == h, f'known input changed: {n}'
    K = pd.read_parquet(old / 'known_cases.parquet')
    K = K.drop(columns=[c for c in K if c.startswith(('K4_', 'K8_', 'S1_', 'S2_', 'S3_', 'ST_', 'EV_', 'M50_'))
                       or c in ('n_candidates', 'anchor_origin', 'anchor_own_row')])
    KE = pd.read_parquet(old / 'known_case_rows.parquet')
    R, read = X.known_checks(K, r0['R']['K0_ref'], KE)
    assert R['legs'] == 99 and R['orders'] == 86 and R['K8']['consistent']
    assert R['K8']['lag_bound_ms'] is None and read['K6 GATE ST'] == 'UNIDENTIFIED-UPPER-BOUND-ONLY'
    assert np.array_equal(R['_legs']['K8_match_lo_bot'], K.place.values)
    assert R['K5']['cancel_with_settlement'].get('CONTRADICTED', 0) == 0
    inputs = {str(old / n): sha(old / n) for n in ('known.json', 'known_cases.parquet', 'known_case_rows.parquet')}
    meta = dict(asof_source=r0['asof'], inputs=inputs, env=r0['env'], source_sha256=sha(SRC / 'pf_keep_pull_exec_windows.py'),
                dependencies={p.name: sha(p) for p in (SRC / 'pf_keep_pull_exec_windows.py', SRC / 'pf_settlement_times.py')},
                provenance=X.KP.provenance(), supersedes=dict(path=str(old / 'known.json'), sha256=sha(old / 'known.json')),
                raw_tape_reextracted=False, consumed_days=['20261108'], untouched_validation_read=False)
    # New output directory only; existing versioned artifacts are never overwritten.
    assert not X.OUT_K.exists(), f'corrected known receipt already exists: {X.OUT_K}'
    lines = X.known_write(X.OUT_K, K, KE, R, read, meta)
    for n in ('known.json', 'known.txt', 'known_schema.json'):
        (OUT / n).write_bytes((X.OUT_K / n).read_bytes())
    pd.read_parquet(X.OUT_K / 'known_cases.parquet').to_csv(OUT / 'known_cases.csv', index=False)
    pd.read_parquet(X.OUT_K / 'known_orders.parquet').to_csv(OUT / 'known_orders.csv', index=False)
    assert X.known_lag()[0] is None
    prior = Path(__file__).resolve().parents[3] / 'predictfun/reviews/keep_pull_settlement_day20261023'
    corrections = dict(asof=X.C.stamp(), model=X.SETTLEMENT_MODEL, source_sha256=meta['source_sha256'],
                       known_receipt=dict(path=str(X.OUT_K / 'known.json'), sha256=sha(X.OUT_K / 'known.json')),
                       superseded={str(Path('predictfun/reviews/keep_pull_settlement_day20261023') / n): dict(sha256=sha(prior / n),
                         status='LEGACY_TIMING_PREMISE_WITHDRAWN; saved values remain conditional scenarios; not corrected economic evidence')
                         for n in ('known.json', 'exec_windows.json', 'scenarios_20261023.json', 'scenario_day_receipt.json')},
                       claimed_259ms_bound='WITHDRAWN', default_public_match_lower_bound='UNIDENTIFIED',
                       simulation_timing_validated=False, full_tape_or_training_run=False, untouched_validation_read=False)
    X.C.dump(OUT / 'supersession.json', corrections)
    print('\n'.join(lines))


def mutants():
    # Relevant changed mechanisms only; full historical mutation suites are not rerun.
    selected = [i for i, (name, *_) in enumerate(XM.MUT) if any(q in name for q in ('K8', 'K5S', 'ST windows', 'ST upper-only', 'same-sample'))]
    result = []
    for i in selected:
        name = XM.MUT[i][0]
        r = subprocess.run([sys.executable, str(SRC / 'pf_keep_pull_exec_windows_mutants.py'), str(i)], capture_output=True, text=True)
        passed = r.returncode not in (0, XM.STALE)
        result.append(dict(index=i, name=name, rejected=passed, returncode=r.returncode, last_output=(r.stderr or r.stdout)[-1200:]))
        print(f'{i}: {name}: rc={r.returncode}, rejected={passed}', flush=True)
    X.C.dump(OUT / 'mutants.json', dict(asof=X.C.stamp(), cases=result))
    assert all(r['rejected'] for r in result), result


def identity():
    """Validate the newly used creation-before-receipt constraint at the actual identity-linked notice."""
    old = Path(X.I.K.IO) / 'keep_pull_exec_windows_known_v3_20261118'
    receipt = load(old / 'known.json')
    fp = X.FC / 'fill_events.parquet'
    assert sha(fp) == receipt['inputs'][str(fp)]
    K = pd.read_parquet(old / 'known_cases.parquet')
    F = pd.read_parquet(fp)
    # Match actual order/trade IDs and exact receipt time, never proximity or vocabulary.
    observed = K[['order_hash', 'trade_id', 'notice']].merge(
        F, left_on=['order_hash', 'trade_id', 'notice'], right_on=['order_hash', 'trade_id', 'at'], how='left', indicator=True)
    assert (observed['_merge'] == 'both').all()
    assert not K[['order_hash', 'trade_id']].duplicated().any()
    X.C.dump(OUT / 'notice_identity.json', dict(asof=X.C.stamp(), known_legs=len(K), matched_rows=len(observed),
        input_sha256={str(fp): sha(fp), str(old / 'known_cases.parquet'): sha(old / 'known_cases.parquet')},
        all_first_notice_times_present_on_exact_order_and_trade_identity=True,
        source_asof=receipt['asof'], consumed_day='20261108'))
    print(f'All {len(K)} known legs have their exact first-notice timestamp on an event carrying the same order/trade identity.')


if __name__ == '__main__':
    {'synthetic': synthetic, 'known': known, 'mutants': mutants, 'identity': identity}[sys.argv[1]]()
