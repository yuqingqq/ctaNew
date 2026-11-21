"""ENTRY-TARGET-AUDIT-v1: THE ACCOUNTING OF THE EXISTING POST / WAIT LABELS (P-2026-005; 2026-11-21, session aea5bfb9). Review d6d5d55e's one
next priority: decompose the existing labels and fill traces before choosing a different entry target or decision unit. No fit, no new label,
no tape access, no policy replay. Declared and frozen before any output.
POPULATION: every POST row of R2-F's accepted Oct 24-25 dataset (2c62fc9b...; in-sample for the frozen models) and of R3's accepted Oct 26-27
 dataset (902526da...; adaptive development), streamed part by part with the frozen keys and weights, every status kept (a non-ok row is
 counted, never dropped). Picks reported separately (mandatory / extra / only) and as the declared weighted total. R4E's 760 reached SMOOTH
 entry pairs are a separate world / side-balanced diagnostic, never mixed with the weighted sample.
ACCOUNTING of an ok POST pair (account(); fills (t, gid, side, q, price, it); g = the fork tick k x 50 ms; pre-fork = t <= g, identical in
 both arms and cancelled; a fill's value at the common END mark = sign(side) x q x (mark - price), sign +1 bid / -1 ask):
 - CLASS, exhaustive and exclusive, in this order:
   no_post_fork_fills (neither arm fills after g); economic_identical (the arms' post-fork (side, q, price) multisets are equal: gid and
   time discarded for this named diagnostic only); same_cash_inventory (otherwise, equal post-fork cash and inventory); offset_by_mark
   (otherwise, |Y| <= 1e-6 c: different cash / inventory offset by the common mark); meaningful (|Y| > 1e-6 c). The first four must carry
   |Y| <= 1e-6 c. The timed comparison (time, side, q, price) is reported beside it. Exact zeros and residues (0 < |Y| <= 1e-6) are flagged.
 - DECOMPOSITION: the forced generation = the POST arm's decision at t = g on the focal side, state absent, action POST (exactly one, with a
   gid; else forced_generation_not_unique). V_initial = its fills valued at the mark. D_following = every other post-fork POST-arm fill value
   minus every post-fork WAIT-arm fill value, split into D_focal (the focal side) and D_opposite. Gids are arm-local and never joined across
   arms. Y = V_initial + D_following must hold to the frozen ledger tolerance (1e-6 c), else identity_failure. V_initial is not a deployable
   entry advantage (no replacement or opportunity cost).
 - EVIDENCE: the initial generation's filled quantity (no_fill / partial / full of its size); the WAIT arm's next actual focal POST after g
   (delay ms, or none). No refusal, queue or removal cause is asserted: the stored traces do not distinguish them.
PREDICTIONS (cross-tabs only; no threshold, fit or search): Oct 26-27 rows join R4E's saved predictions (SMOOTH, RICH, their actions and
 support) by the exact key (day, m, t_ms, side, pick); Oct 24-25 rows get the frozen models' in-sample predictions from their stored 107
 inputs (R3's RICH POST heads, R4E's SMOOTH heads, R3's support). A missing join is a status, never a drop.
DECLARED READINGS (summaries(); per set train / eval, day, side and pick group, weighted by the stored w): counts, weight totals, class shares,
 positive / negative / meaningful counts, weighted means and absolute totals of Y, V_initial, D_focal, D_opposite, D_following; the top-5
 |Y| share; the initial fill and WAIT-next-POST distributions; strata (source, horizon, generation, other side); the actions' cross-tab.
DECLARED PREDICATES (per set, over its ok rows, weighted; the thresholds fixed here):
 - rare_execution: the meaningful share < 0.10 and the initial generation fills nothing in >= 0.50 of rows;
 - cancellation: sum w |V_initial| > 0 and sum w |Y| <= 0.5 x sum w |V_initial| (the continuation removes at least half of the initial
   value's absolute mass);
 - diagnosis: both / rare_execution / cancellation / not_identified.
Usage (run_light): pf_entry_audit.py selftest | run"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402  frozen (R3 freeze 2): readers, models
import pf_adv2_entry as N                                # noqa: E402  frozen (R4E): SMOOTH heads
import pf_adv2_eval as AE                                # noqa: E402  frozen: the evaluation plan / directory
import pf_adv2_entry_eval as NE                          # noqa: E402  frozen: the exact CSV reader, the key
import pf_adv2_entry_replay as NR                        # noqa: E402  frozen: R4E's replay parts

C, pd, T, LS, L = A.C, A.pd, A.T, A.LS, A.L
VERSION = 'entry_target_audit_v1'
SETS = dict(train=dict(dir=A.TRAIN['dir'], sha=A.TRAIN['dataset_sha256'], days=A.TRAIN['days']),
            eval=dict(dir=AE.EVAL_DIR, sha=NE.R3_DATASET, days=('20261026', '20261027')))
R4E_PRED = Path('predictfun/reviews/r4e_entry_smooth_20261121/eval/predictions.csv.gz')
OUT = A.DATA / 'entry_target_audit_v1_20261121'
CLASSES = ('no_post_fork_fills', 'economic_identical', 'same_cash_inventory', 'offset_by_mark', 'meaningful')
MEANINGFUL, TOL, TOLP = 1e-6, LS.TOL_V, LS.TOL_P
RARE_MEANINGFUL, RARE_NOFILL, CANCEL = .10, .50, .5
GEN_I = T.FEATS.index('ctx_generation_ordinal')
FILES = dict(rows='rows.csv.gz', reached_rows='reached_rows.csv', groups='groups.csv', strata='strata.csv', crosstab='crosstab.csv', reached_summary='reached_summary.csv',
             statuses='statuses.csv', reached_statuses='reached_statuses.csv')          # distinct names (freeze amendment: the reached rows were overwritten)


def sg(s):
    return 1. if int(float(s)) == 0 else -1.


def fv(f, mark):
    """A fill's value at the END mark: sign x q x (mark - price)."""
    return sg(f[2]) * float(f[3]) * (mark - float(f[4]))


def account(res, side):
    """One ok POST pair's result -> the accounting record (header)."""
    g, mark, Y = float(res['k']) * L.STEP, float(res['mark']), float(res['Y']); a, b = res['arms']['POST'], res['arms']['WAIT']
    fa = [f for f in a['fill_list'] if float(f[0]) > g]; fb = [f for f in b['fill_list'] if float(f[0]) > g]
    pre_equal = [f for f in a['fill_list'] if float(f[0]) <= g] == [f for f in b['fill_list'] if float(f[0]) <= g]
    forced = [x for x in a['decision_list'] if float(x[0]) == g and int(x[1]) == side and x[2] == 'absent' and x[3] == 'POST' and x[4] is not None]
    if len(forced) != 1:
        return dict(status='forced_generation_not_unique', Y=Y)
    gid0 = int(forced[0][4])
    econ = lambda fl: sorted((int(float(f[2])), round(float(f[3]), 9), round(float(f[4]), 9)) for f in fl)
    timed = lambda fl: sorted((float(f[0]), int(float(f[2])), float(f[3]), float(f[4])) for f in fl)
    cash = lambda fl: -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in fl)
    pos = lambda fl: sum(sg(f[2]) * float(f[3]) for f in fl)
    dcash, dpos = cash(fa) - cash(fb), pos(fa) - pos(fb)
    if not fa and not fb:
        cls = 'no_post_fork_fills'
    elif econ(fa) == econ(fb):
        cls = 'economic_identical'
    elif abs(dcash) <= TOL and abs(dpos) <= TOLP:
        cls = 'same_cash_inventory'
    elif abs(Y) <= MEANINGFUL:
        cls = 'offset_by_mark'
    else:
        cls = 'meaningful'
    v0 = sum(fv(f, mark) for f in fa if int(float(f[1])) == gid0)
    d_focal = sum(fv(f, mark) for f in fa if int(float(f[1])) != gid0 and int(float(f[2])) == side) - sum(fv(f, mark) for f in fb if int(float(f[2])) == side)
    d_opp = sum(fv(f, mark) for f in fa if int(float(f[2])) != side) - sum(fv(f, mark) for f in fb if int(float(f[2])) != side)
    gap = Y - (v0 + d_focal + d_opp); consistent = cls == 'meaningful' or abs(Y) <= MEANINGFUL
    q0 = sum(float(f[3]) for f in fa if int(float(f[1])) == gid0); size0 = float(a['sizes'][gid0]) if 0 <= gid0 < len(a['sizes']) else float('nan')
    init_fill = 'no_fill' if q0 <= TOLP else ('full' if q0 >= size0 - TOLP else 'partial')
    wp = [float(x[0]) for x in b['decision_list'] if float(x[0]) > g and int(x[1]) == side and x[3] == 'POST']
    ok = bool(pre_equal and consistent and abs(gap) <= TOL and np.isfinite(size0))
    return dict(status='ok' if ok else 'identity_failure', cls=cls, Y=Y, exact_zero=bool(Y == 0.), residue=bool(Y != 0. and abs(Y) <= MEANINGFUL), v_initial=v0,
                d_focal=d_focal, d_opposite=d_opp, d_following=d_focal + d_opp, identity_gap=gap, prefork_equal=bool(pre_equal), class_consistent=bool(consistent),
                init_fill=init_fill, init_filled=q0, init_size=size0, wait_next_post_ms=(min(wp) - g) if wp else None, timed_identical=bool(timed(fa) == timed(fb)),
                fills_post=len(fa), fills_wait=len(fb), gid0=gid0)


def base(r, kind):
    rk = r.get('row_key') or {}
    return dict(set=kind, day=str(r['day']), m=int(r['m']), t_ms=int(r['t_ms']), side=int(r['side']), pick=r.get('pick'), source=r.get('source'), status=r['status'],
                w=r.get('w'), decision_ms=rk.get('decision_ms'))


def cats_of(rows):
    """The horizon bin and the six support categories of the accounted rows (their stored fork inputs, price, generation)."""
    x = pd.DataFrame([dict(zip(T.FEATS, r['_inputs'])) for r in rows]); x['price'] = [r['_price'] for r in rows]; x['generation'] = x['ctx_generation_ordinal'].astype(float)
    c = T.categories(x); hb = T.hbin(A.END - np.array([float(r['decision_ms']) for r in rows], float))
    for i, r in enumerate(rows):
        r['hbin'] = str(hb[i]); r.update({k: str(c[k][i]) for k in A.R2.DIMS})


def collect(kind):
    """Every POST row of one set (the dataset and part identities checked by pf_adv2.dataset_rows) -> accounted rows."""
    s = SETS[kind]; out = []
    for r in A.dataset_rows(s['dir'], s['sha']):
        if r.get('head') != 'post':
            continue
        C.require(str(r['day']) in s['days'], f"a row of {r['day']} outside the set's days")
        b = base(r, kind)
        if r['status'] == 'ok':
            acc = account(r['result'], int(r['side'])); b.update(acc); b['status'] = acc['status']
            b['_inputs'] = [float(v) for v in r['fork']['inputs']]; b['_price'] = float(r['fork']['price'])
        out.append(b)
    oks = [r for r in out if '_inputs' in r]
    if oks:
        cats_of(oks)
    return out


def predictions(rows, kind, smooth, rich, csup):
    """The frozen models' predictions / actions per row: Oct 26-27 joined from R4E's saved file by the exact key; Oct 24-25 in-sample."""
    if kind == 'eval':
        P = NE.read_r3(N.REPO / R4E_PRED); P['day'] = P.day.astype(str); idx = {tuple(k): i for i, k in enumerate(P[NE.KEY].itertuples(index=False, name=None))}
        for r in rows:
            i = idx.get((r['day'], r['m'], r['t_ms'], r['side'], r['pick']))
            if i is None:
                if '_inputs' in r:
                    r['pred_status'] = 'prediction_missing'
                continue
            r.update(pred_SMOOTH=float(P.pred_SMOOTH.iat[i]), pred_RICH=float(P.pred_RICH.iat[i]), act_SMOOTH=P.act_SMOOTH.iat[i], act_RICH=P.act_RICH.iat[i], support=P.support.iat[i],
                     pred_status='joined')
    else:
        for r in rows:
            if '_inputs' not in r:
                continue
            x = np.asarray([r['_inputs']], np.float32); s = r['side']
            row = dict(zip(T.FEATS, r['_inputs'])); row.update(price=r['_price'], generation=r['_inputs'][GEN_I], decision_ms=float(r['decision_ms']))
            sup = csup.check(s, 'post', row) or 'supported'; ps, pr = float(N.predict(smooth[(s, 'post')], x)[0]), float(A.predict(rich['RICH'][(s, 'post')], 'RICH', x)[0])
            act = lambda p: ('POST' if p > 0 else 'WAIT') if sup == 'supported' and np.isfinite(p) else 'fallback'
            r.update(pred_SMOOTH=ps, pred_RICH=pr, act_SMOOTH=act(ps), act_RICH=act(pr), support=sup, pred_status='in_sample')


def reached_rows():
    """R4E's reached SMOOTH entry pairs (every replay part's bytes against its record) -> accounted rows (a balanced diagnostic, unweighted)."""
    out = []
    for p in sorted((NR.OUT / 'parts').glob('part_*[0-9].json')):
        rec = json.loads(p.read_text()); b = (NR.OUT / 'parts' / f'{p.stem}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'], f'{p.stem}: bytes differ from its record')
        for r in LS.gz_rows(b):
            for x in (r.get('reached') or []) if r['status'] == 'ok' else []:
                d = dict(set='reached', day=str(r['day']), m=int(r['m']), t_ms=int(r['t_ms']), side=int(x['side']), status=x['status'])
                if x['status'] == 'ok':
                    acc = account(x['result'], int(x['side'])); d.update(acc); d['status'] = acc['status']
                    d.update(pred_SMOOTH=(x.get('pred') or {}).get('SMOOTH'), pred_RICH=(x.get('pred') or {}).get('RICH'))
                out.append(d)
    return out


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float(np.sum(w * x) / np.sum(w)) if len(x) and np.sum(w) > 0 else None


def group_summary(g, weighted=True):
    """One group of accounted rows -> the declared readings (header)."""
    if not len(g):
        return dict(rows=0)
    w = g.w.astype(float).values if weighted else np.ones(len(g)); y = g.Y.astype(float).values; W = float(w.sum())
    out = dict(rows=int(len(g)), originals=int(len(g[['day', 'm', 't_ms', 'side']].drop_duplicates())), worlds=int(len(g[['day', 'm', 't_ms']].drop_duplicates())), w_sum=W,
               positive=int((y > MEANINGFUL).sum()), negative=int((y < -MEANINGFUL).sum()), meaningful=int((np.abs(y) > MEANINGFUL).sum()),
               exact_zero=int((y == 0.).sum()), residue=int(((y != 0.) & (np.abs(y) <= MEANINGFUL)).sum()))
    for c in CLASSES:
        m = (g.cls == c).values; out[f'share_{c}'] = float(w[m].sum() / W); out[f'n_{c}'] = int(m.sum())
    for k in ('Y', 'v_initial', 'd_focal', 'd_opposite', 'd_following'):
        v = g[k].astype(float).values; out[f'wmean_{k}'] = float(np.sum(w * v) / W); out[f'abs_{k}'] = float(np.sum(w * np.abs(v)))
    a = np.sort(np.abs(y))[::-1]; out['top5_abs_share'] = float(a[:5].sum() / a.sum()) if a.sum() > 0 else None
    for f in ('no_fill', 'partial', 'full'):
        out[f'init_{f}'] = float(w[(g.init_fill == f).values].sum() / W)
    wn = g.wait_next_post_ms
    out['wait_no_later_post'] = float(w[wn.isna().values].sum() / W)
    for lo, hi in ((0, 100), (100, 500), (500, 1000), (1000, 1e9)):
        out[f'wait_next_{lo}_{int(hi) if hi < 1e9 else "inf"}'] = float(w[((wn > lo) & (wn <= hi)).fillna(False).values].sum() / W)
    out['opposite_sign_share'] = float(w[((np.sign(g.v_initial) * np.sign(g.d_following)) < 0).values].sum() / W)
    return out


def predicates(s):
    """The declared predicates of one set's summary (header)."""
    rare = bool(s['rows'] and s['share_meaningful'] < RARE_MEANINGFUL and s['init_no_fill'] >= RARE_NOFILL)
    cancel = bool(s['rows'] and s['abs_v_initial'] > 0 and s['abs_Y'] <= CANCEL * s['abs_v_initial'])
    return dict(rare_execution=rare, cancellation=cancel, diagnosis='both' if rare and cancel else 'rare_execution' if rare else 'cancellation' if cancel else 'not_identified')


def summaries(D, R):
    """The declared tables: per set / day / side / pick group (weighted), the sets' totals and predicates, strata, action cross-tabs, and the
    reached diagnostic (unweighted)."""
    ok = D[D.status == 'ok']; rows, strata, cross, preds = [], [], [], {}
    for kind, g0 in ok.groupby('set', sort=True):
        preds[kind] = dict(predicates(group_summary(g0)), **{k: group_summary(g0)[k] for k in ('share_meaningful', 'init_no_fill', 'abs_v_initial', 'abs_Y')})
        rows.append(dict(set=kind, day='all', side='all', pick='all', **group_summary(g0)))
        for (day, s), g1 in g0.groupby(['day', 'side'], sort=True):
            rows.append(dict(set=kind, day=day, side=int(s), pick='all', **group_summary(g1)))
            for pk, g2 in g1.groupby('pick', sort=True):
                rows.append(dict(set=kind, day=day, side=int(s), pick=pk, **group_summary(g2)))
        for dim in ('source', 'hbin', 'c_gen', 'c_other'):
            for val, g1 in g0.groupby(dim, sort=True):
                strata.append(dict(set=kind, dim=dim, value=str(val), **{k: v for k, v in group_summary(g1).items() if not k.startswith('wait_next')}))
        for cand in ('SMOOTH', 'RICH'):
            for (day, s, act), g1 in g0.groupby(['day', 'side', f'act_{cand}'], sort=True):
                gs = group_summary(g1); cross.append(dict(set=kind, candidate=cand, day=day, side=int(s), action=act, rows=gs['rows'], w_sum=gs['w_sum'],
                                                          wmean_Y=gs['wmean_Y'], wmean_v_initial=gs['wmean_v_initial'], wmean_d_following=gs['wmean_d_following'],
                                                          share_meaningful=gs['share_meaningful'], init_no_fill=gs['init_no_fill']))
    rk = R[R.status == 'ok']; reached = [dict(day=day, **group_summary(g, weighted=False)) for day, g in rk.groupby('day', sort=True)]
    st = D.groupby(['set', 'day', 'status']).size().rename('n').reset_index().to_dict('records')
    return dict(groups=rows, strata=strata, crosstab=cross, reached=reached, statuses=st, predicates=preds,
                reached_statuses=R.groupby(['day', 'status']).size().rename('n').reset_index().to_dict('records'))


def write_outputs(out, D, R, S, js):
    """Every output under its own declared name (FILES, checked distinct), audit.json last."""
    out = Path(out); C.require(len(set(FILES.values())) == len(FILES), 'two outputs share a file name')
    D.to_csv(out / FILES['rows'], index=False); R.to_csv(out / FILES['reached_rows'], index=False)
    for k in ('groups', 'strata', 'crosstab', 'statuses', 'reached_statuses'):
        pd.DataFrame(S[k]).to_csv(out / FILES[k], index=False)
    pd.DataFrame(S['reached']).to_csv(out / FILES['reached_summary'], index=False)
    (out / 'audit.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')


def cmd_run(out=OUT):
    """The audit -> out/rows.csv.gz, reached.csv, the summary CSVs and audit.json (header). Never overwrites an existing audit."""
    out = Path(out); C.require(not (out / 'audit.json').exists(), 'the audit exists'); out.mkdir(parents=True, exist_ok=True); t0 = time.time()
    fz, smooth, rich, csup = N.frozen()
    rows = []
    for kind in ('train', 'eval'):
        rk = collect(kind); predictions(rk, kind, smooth, rich, csup); rows += rk
    D = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]); R = pd.DataFrame(reached_rows())
    acc = {k: json.loads((Path(s['dir']) / 'accepted.json').read_text()) for k, s in SETS.items()}
    exp = {k: sum(v for kk, v in a['statuses'].items() if kk.split()[1] == 'post') for k, a in acc.items()}
    C.require(all(int((D.set == k).sum()) == exp[k] for k in SETS), f'not every POST row once: {D.set.value_counts().to_dict()} against {exp}')
    S = summaries(D, R)
    js = dict(version=VERSION, datasets={k: s['sha'] for k, s in SETS.items()}, expected_post_rows=exp, rows=int(len(D)), reached_rows=int(len(R)), predicates=S['predicates'],
              thresholds=dict(rare_meaningful=RARE_MEANINGFUL, rare_nofill=RARE_NOFILL, cancel=CANCEL), r4e_predictions_sha256=C.sha(N.REPO / R4E_PRED),
              fit_sha256=C.sha(N.OUT / 'fit.json'), source_sha256=C.sha(Path(__file__)), seconds=round(time.time() - t0, 1), asof=C.stamp())
    write_outputs(out, D, R, S, js)
    print(json.dumps(dict(rows=js['rows'], reached=js['reached_rows'], predicates=S['predicates'], statuses=S['statuses']), indent=1, default=str))


def selftest():
    import pf_entry_audit_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        cmd_run()
    else:
        sys.exit(__doc__)
