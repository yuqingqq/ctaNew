"""Bounded, synthetic-only review of c23e6f29; does not change development code.

Run through PF_SLOT=codex run_light.sh, passing an isolated pinned checkout.
Controls: a coherent singleton-event sweep, all-contradictory linked candidates,
and a corrupt cross-tab that the independent conservation check must reject.
"""
import contextlib
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TARGET = 'c23e6f297c9b822cb3af8247da0b5e1ea99148f5'
root = Path(sys.argv[1]).resolve()
dest = Path(__file__).resolve().parent
assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == TARGET
sys.path.insert(0, str(root / 'predictfun/research'))
import pf_keep_pull_exec_windows as X

np, pd, KP, A, V1 = X.np, X.pd, X.KP, X.A, X.V1
receipt = dict(target=TARGET, asof=datetime.now(timezone.utc).isoformat(), synthetic_only=True,
               empirical_data_read=False, source_sha256=hashlib.sha256(Path(X.__file__).read_bytes()).hexdigest())
with (dest / 'selftest.txt').open('w') as stream, contextlib.redirect_stdout(stream):
    X.selftest()
receipt['pinned_selftest'] = 'PASS'

# Two unique fills (one at A, one at B), and an ambiguous fill from A or B.
# Every fill shares the same cancellation threshold; value sums are in synthetic cents.
F = dict(val=np.array([-10., -10., 10.]), pulled=np.ones(3, bool), T=np.full(3, 5),
         s=np.zeros(3, int), endj=np.full(3, 99), lend=np.full(3, np.inf))
CW = np.zeros((1, KP.NSTEP + 2))
u = {k: v[:2] for k, v in F.items()}
mn_u, mx_u = X.sweep(np.arange(2), np.full(2, 100.), np.full(2, 400.), np.zeros(2), u, CW, KP.CANCEL)
m = {k: np.repeat(v[2:], 2) for k, v in F.items()}
mn_m, mx_m = X.sweep(np.arange(2), np.full(2, 100.), np.full(2, 400.), np.zeros(2), m, CW, KP.CANCEL)
relaxed = [float(mn_u.sum() + mn_m.min()), float(mx_u.sum() + mx_m.max())]
# Enumerate every state: only the 287 ms threshold changes a contribution here.
joint = [float(X.contrib(np.array([a, b, (a, b)[event]]), F, CW).sum())
         for a in (200., 300.) for b in (200., 300.) for event in (0, 1)]
assert relaxed == [-10., 20.] and [min(joint), max(joint)] == [0., 10.]
control = {k: v[[0, 2]] for k, v in F.items()}
cmn, cmx = X.sweep(np.zeros(2, int), np.array([100.]), np.array([400.]), np.zeros(2), control, CW, KP.CANCEL)
assert [cmn[0], cmx[0]] == [0., 0.]
receipt['multi_event_exactness'] = dict(fills=3, events=2, candidate_assignments=2,
    shared_event_control=[float(cmn[0]), float(cmx[0])], current_relaxation=relaxed,
    exhaustive_joint_bounds=[min(joint), max(joint)], meaning='outer bounds, not exact joint extrema')

# A complete synthetic run with valid input identity and deliberately contradictory
# scenario windows. Both unique and multi-event fills must remain in the accounting.
O, Fl, L = X._synthetic(np.random.default_rng(KP.SEED))
L['how'] = 1
L['t_ms'] = L.anc
L['t_e2'] = np.nan
L['exec_s'] = np.floor(L.anc / 1000.) - 10.
day = '20990101'
rows = L[X.RAW_COLS + ['tk', 'tk_status']].reset_index(drop=True)
E = O[O.status == 'eligible'].reset_index(drop=True)
cell = E.groupby(['day', 'm', 'side', 'phase'], sort=True).ngroup().values
T = KP.score_steps(E[[f'f{j:02d}' for j in range(KP.NSTEP + 1)]].values, E.side.values, E.p.values)[KP.DECLARED]
endj = A.endj_of(E.t_full.values, E.g_struct.values)
act, cw = V1.null_weights(T, cell, endj)
ri, t, val, src = A.fills_of(E, Fl, X.SHIFT)
pf = A.per_fill(T, cell, endj, (ri, t, val, src))
fd = pd.DataFrame(dict(fid=np.arange(len(ri)), m=E.m.values[ri], side=E.side.values[ri],
                       a=t + X.SHIFT + E.t.values[ri], p=E.p.values[ri]))
hull = V1.link(fd, L[X.V1_COLS], 600., -900., 1050.).reindex(np.arange(len(ri)))
r = KP.landing(np.clip(T[ri], 1, KP.NSTEP))
f1 = dict(val=val, pulled=act[ri], r=r, s=cell[ri], endj=endj[ri],
          lend=np.where(np.isnan(E.land_struct.values[ri]), np.inf, E.land_struct.values[ri]))
v1 = {}
for k, (a, b) in (('S1', ('lo1', 'hi1')), ('S2', ('lo2', 'hi2'))):
    lo, hi = hull[a].values - E.t.values[ri], hull[b].values - E.t.values[ri]
    mn, mx = V1.bounds(lo, hi, f1, cw)
    cls = np.where(act[ri], V1.order_of(lo, hi, r), -1)
    v1[k] = dict(E_min=float(mn.sum()), E_max=float(mx.sum()),
                 order={name: int(((cls == j) & act[ri]).sum()) for j, name in enumerate(('AFTER', 'BEFORE', 'AMBIGUOUS'))})
with tempfile.TemporaryDirectory(prefix='pf-review-c23e6f29-') as tmp:
    out = Path(tmp) / 'run'
    with (dest / 'contradictions.txt').open('w') as stream, contextlib.redirect_stdout(stream):
        R, readings = X.run(out=out, src=X._Src(L[X.LEG_COLS]), tables=(O, Fl), rows_of=lambda _: rows,
                            env_of=lambda _: dict(h=600., q01=-900., q99=1050., inputs={}), v1=v1, days=[day],
                            expect=(float(pf['score'].sum() - pf['null'].sum()), float(pf['score'].sum())), tol=1e-6)
    ev = pd.read_parquet(out / 'ledger_events.parquet')
    fills = pd.read_parquet(out / 'ledger_fills.parquet')
    pairs = pd.read_parquet(out / 'ledger_fill_events.parquet')
    assert readings['V0 CONTROLS'] == 'YES'
    for k in ('S1', 'S2'):
        assert (ev[f'{k}_st'] == 2).all()
        assert (fills.loc[fills.pulled, f'cls_{k}_CP'] == 5).all()
    linked = pairs.merge(ev[['e', 'EV_hi']], on='e', validate='many_to_one').merge(
        fills[['fid', 't0', 't_replay']], on='fid', validate='many_to_one')
    assert (linked.t0 + linked.t_replay > linked.EV_hi).all()
    n, multi = len(fills), int((fills.link == 'multi_event').sum())
    assert multi > 0
    assert R['V3']['S1']['fills_contradictory'] == n - multi
    assert R['V3']['replay_after_evidence']['fills'] == n - multi
    good = X.xtab(np.array([0]), np.array([True]), np.array([-10.]), np.array([True]))
    assert X.conserved(good)
    good['REMOVED']['CANCEL_FIRST']['fills'] = 0
    assert not X.conserved(good), 'Independent checker failed to reject a lost fill'
    receipt['contradiction_accounting'] = dict(counted_orders=len(O), eligible_orders=len(E), fills=n, events=len(ev),
        unique_fills=n - multi, multi_event_fills=multi, all_candidate_windows_contradictory=True,
        reported_contradictory=R['V3']['S1']['fills_contradictory'], expected_all_candidate_contradictory=n,
        reported_replay_after_evidence=R['V3']['replay_after_evidence']['fills'], expected_replay_after_evidence=n,
        omitted_gross=float(fills.loc[fills.link == 'multi_event', 'value'].abs().sum()),
        V0=json.loads((out / 'exec_windows.json').read_text())['V0'], corrupt_cross_tab_refused=True)

(dest / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
