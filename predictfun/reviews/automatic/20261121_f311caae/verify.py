"""Bounded read-only review of f311caae: saved artifacts and synthetic controls.

Run via PF_SLOT=codex run_light.sh; no fit, real branch replay or new tape lookup.
Temporary corruptions affect private copies only. Pinned checkout imports only.
"""
from pathlib import Path
import collections, gzip, hashlib, io, json, subprocess, sys, tempfile

ROOT = Path('/home/yuqing/ctaNew')
PIN = Path('/tmp/pf-review-f311caae')
RES = PIN / 'predictfun/research'
RC = PIN / 'predictfun/reviews/movement_audit_20261121'
OUT = ROOT / 'predictfun/reviews/automatic/20261121_f311caae'
TARGET = 'f311caaef6ea50b8a7df7ebd6c74a3cf22f39dc1'
sys.path.insert(0, str(RES))
import numpy as np
import pandas as pd
import pf_movement_audit as M
import pf_movement_audit_selftest as ST
sys.path.insert(0, str(RC))
import movement_checks as X
sys.path.insert(0, str(RES)); X.RES = RES; X.LOGS = RC / 'logs'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git', '-C', str(PIN), *a])
def rd(p): return pd.read_csv(p, float_precision='round_trip', dtype=X.STR)
def avg(x, w): return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))
def close(a, b): assert np.isclose(a, b, rtol=1e-12, atol=1e-9), (a, b)

def metadata():
    assert git('rev-parse', 'HEAD').decode().strip() == TARGET
    checked = []
    for line in (RC / 'outputs/SHA256SUMS').read_text().splitlines():
        h, n = line.split(); p = RC / 'outputs' / n
        assert sha(p) == h, n
        b = gzip.decompress(p.read_bytes()) if n.endswith('.csv.gz') else p.read_bytes()
        dest = M.OUT / (n[:-3] if n.endswith('.csv.gz') else n)
        assert b == dest.read_bytes(), n
        checked.append(n)
    for n in ('pf_movement_audit.py', 'pf_movement_audit_selftest.py', 'pf_movement_audit_mutants.py'):
        assert git('show', '03243e5b:predictfun/research/' + n) == (RES / n).read_bytes()
    for n in ('movement_checks.py', 'declaration.md', 'launch_movement.sh'):
        assert git('show', '03243e5b:predictfun/reviews/movement_audit_20261121/' + n) == (RC / n).read_bytes()
    D = X.load(X.REAL); prior = 0
    for collection, receipt in (('pa', 'pi0_admission_20261121'), ('pf', 'entry_retention_20261121')):
        for k, (b, rec) in D[collection].items():
            assert hashlib.sha256(b).hexdigest() == rec['sha256']
            assert rec == read(PIN / 'predictfun/reviews' / receipt / 'outputs/parts' / f'part_{k:03d}.json')
            prior += 1
    for k, rec in D['mids'].items():
        assert rec['part'] == k and rec['sources'] == M.sources()
        assert rec['admission_record_sha256'] == sha(M.ADM / 'parts' / f'part_{k:03d}.json')
        assert rec['admission_manifest_sha256'] == sha(M.ADM / 'manifest.json')
        rows = X.gz_rows(D['pa'][k][0]); required = {X.kg(r) for r in rows if r['status'] == 'ok' and r['held']['fills']}
        assert set(rec['mids']) == required
        for r in rows:
            if X.kg(r) not in required: continue
            mr = rec['mids'][X.kg(r)]; assert mr['status'] == 'ok'
            assert set(mr['mids']) == {str(i) for i in range(len(r['held']['fills']))}
            assert rec['day'] == str(r['day'])
    assert M.frozen() and M.inputs_frozen()
    assert all(Path(z.__file__).is_relative_to(RES) for n, z in sys.modules.items() if n.startswith('pf_') and getattr(z, '__file__', None))
    su = X.suites(X.LOGS / 'mutants_mov'); assert su['ok']
    return dict(target=TARGET, frozen_files_unchanged=6, receipt_artifacts_equal=len(checked), prior_partition_records_bound=prior,
                mids_record_bindings=len(D['mids']), mutants=su)

def data():
    ck, det = X.run_checks(X.REAL); assert all(ck.values()), (ck, det)
    D = X.load(X.REAL); ar, _ = M.adm_rows(); fr = M.forced_rows()
    O1, F1 = M.tables_of(ar, 'admitted', M.adm_mref(D['mk']))
    O2, F2 = M.tables_of(fr, 'forced', M.forced_mids)
    O = pd.concat([O1, O2], ignore_index=True); F = pd.concat([F1, F2], ignore_index=True)
    G = M.PA.gen_table(ar); s1, S1 = M.sources_table(ar, 'admitted'); s2, S2 = M.sources_table(fr, 'forced')
    funcs = dict(opportunities=O, fills=F, moments=M.moments(O), strata=M.strata(O), prepost=M.prepost(F),
                 concentration=M.concentration(G, O), contributions=M.contributions(G, O), sources=s1+s2,
                 transitions=pd.concat([S1, S2], ignore_index=True))
    preds = [M.predicates(O, funcs['strata'], funcs['concentration'], n) for n in M.STAGES]
    assert preds == D['audit']['predicates']
    funcs['evidence'] = [e for p in preds for e in M.evidence_rows(p)]
    for n, tab in funcs.items():
        assert pd.DataFrame(tab).to_csv(index=False).encode() == (M.OUT / f'{n}.csv').read_bytes(), n
    E = D['O'][D['O'].acc_status == 'ok']; diagnostics = []
    # Direct arithmetic independent of the aggregation helpers; every cohort/stage/view.
    for _, r in D['MO'].iterrows():
        g = E[(E.cohort == r.cohort) & (E.stage == r.stage)]
        pool = avg(g.z, g.weight)
        if r.view in ('day', 'day_side'): g = g[g.day == r.day]
        if r.view in ('side', 'day_side'): g = g[g.side == r.side]
        if r.view in ('overlap', 'overlap_admission'): g = g[g.day.isin(X.OVL)]
        if r.view in ('admission', 'overlap_admission'): g = g[g.admission == r.admission]
        w = g.weight; f = g[g.z == 1]
        Mv = g.q * 0 + g.M
        close(avg(g.V-g.value, w), r.bias_V)
        close(avg((g.V-g.value)**2, w), r.mse_V)
        close(avg(g.value**2, w), r.mse_zero)
        close(avg((g.edge*g.Eq-g.value)**2, w), r.mse_mech)
        close(avg((g.P-g.z)**2, w), r.brier)
        close(avg((pool-g.z)**2, w), r.brier_pooled)
        close(avg((g.Eqm-Mv)**2, w), r.mse_m)
        if len(f):
            close(avg(f.Eqm/f.P, f.weight), r.mean_mhat_fill)
            close(avg(f.M, f.weight), r.mean_M_fill)
            close(avg(f.Eqm/f.P-f.M, f.weight), r.c_m)
        assert np.allclose(g.V-g.value, g.edge*(g.Eq-g.q)+(g.Eqm-g.M), rtol=0, atol=1e-6)
        assert np.allclose(g.Eqm-g.M, (g.P-g.z)*(g.Eqm/g.P)+g.z*(g.Eqm/g.P-g.M), rtol=0, atol=1e-6)
        assert np.allclose(g.Eq-g.q, (g.P-g.z)*(g.Eq/g.P)+g.z*(g.Eq/g.P-g.q), rtol=0, atol=1e-6)
    for _, r in D['PP'].iterrows():
        g = D['F'][(D['F'].cohort == r.cohort) & (D['F'].stage == r.stage)]
        if r['view'] == 'day_side': g = g[(g.day == str(r.cell)) & (g.side == r.side)]
        if r['view'] == 'delay_class': g = g[g.delay_class == r.cell]
        if r['view'] == 'overlap': g = g[g.day.isin(X.OVL)]
        g = g[g.ref_status == 'ok']; denom = float((g.weight*g.q).sum())
        for col in ('total', 'pre', 'post'):
            close(float((g.weight*g[col]).sum())/denom, r[col+'_per_share'])
        if r.stage == 'entry' and r['view'] in ('all', 'overlap'):
            diagnostics.append(dict(cohort=r.cohort, view=r['view'], fills=len(g), unweighted_shares=float(g.q.sum()),
                                    weighted_shares=denom, weighted_total_c=float((g.weight*g.total).sum()),
                                    total_c_per_share=r.total_per_share, pre_c_per_share=r.pre_per_share, post_c_per_share=r.post_per_share))
    pop = E.groupby(['cohort', 'stage', 'day', 'side']).agg(n=('z','size'), fills=('z','sum'), w=('weight','sum')).reset_index()
    pop.to_csv(OUT / 'population.csv', index=False)
    entries = E[E.stage == 'entry']
    return dict(target=TARGET, checks=ck, details=det, aggregate_csvs_byte_reproduced=list(funcs),
                statuses=D['audit']['statuses'], per_share=diagnostics,
                moment_views_independently_checked=len(D['MO']), prepost_views_independently_checked=len(D['PP']),
                forced_entry_state=dict(n=len(entries[entries.cohort == 'forced']), t=sorted(entries[entries.cohort == 'forced'].t_dec.unique().tolist()),
                                        ordinal=sorted(entries[entries.cohort == 'forced'].ordinal.unique().tolist())),
                predicates=preds)

def controls():
    ST.run(M); X.selftest()
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        cases, ok = X.falsify_on(X.REAL, td, True)
    assert ok, cases
    return dict(target=TARGET, selftests_passed=True, cases=cases, falsifier_ok=ok)

def instrument():
    # Test the coverage boundary without modifying any original artifact.
    with tempfile.TemporaryDirectory(dir=OUT) as td:
        Q = X.copy_tree(X.REAL, Path(td)/'copy')
        clean, _ = X.run_checks(Q); assert all(clean.values())
        for name, col in (('moments','mean_mhat_fill'), ('prepost','post_per_share'), ('strata','c_m')):
            p = Path(Q['out']) / (name+'.csv'); df = rd(p)
            i = df[(df.cohort == 'admitted') & (df.stage == 'entry') & df[col].notna()].index[0]
            df.loc[i,col] = float(df.loc[i,col]) + 100.; df.to_csv(p, index=False)
        altered, _ = X.run_checks(Q)
        # This is a demonstrated checker gap, not corruption of the reviewed outputs.
        assert all(altered.values()), altered
        # Receipt SHA binding and the direct aggregate reproduction reject these edits.
        hashes_refuse = all((Path(Q['out'])/(n+'.csv')).read_bytes() != (M.OUT/(n+'.csv')).read_bytes() for n in ('moments','prepost','strata'))
        assert hashes_refuse
    return dict(target=TARGET, clean=clean, altered=altered, deliberate_aggregate_changes=3, independent_hash_comparison_refuses=True,
                interpretation='A-I do not independently validate all reported aggregate values; current values reproduced in data.json')

if __name__ == '__main__':
    mode = sys.argv[1]; result = globals()[mode]()
    (OUT / (mode+'.json')).write_text(json.dumps(result, indent=2, sort_keys=True, default=str)+'\n')
    print(json.dumps(dict(mode=mode, passed=True)))
