"""ADV2-RICH v1: THE OFFLINE PREDICTION DIAGNOSTIC ON CONSUMED OCT 26-27 AND THE LOCAL ACTION-INFORMATION CONTROL (plan v2 R3; P-2026-005;
2026-11-21, session aea5bfb9). Review 315979d2's next priority, items 6-10. Declared and frozen (freeze 2) before any fit and before any
Oct 26-27 label is read. It refuses to read a label unless the fit manifest is committed (HEAD holds the receipt copy of fit.json, equal
byte for byte to the one the models are verified against): the model hashes precede every evaluation outcome.

POPULATION: every row of the accepted evaluation sample of the frozen plan (pf_adv2_sample, plan 8619eec5...; accepted.json must name that
 plan; its dataset hash is recorded). ok rows are evaluated; every other status is counted with its reason (never a zero label, never
 dropped). All three candidates predict the identical rows from the stored 107 inputs at the fork (before the forced action).
SUPPORT AND SELECTED ACTION: the candidate support (pf_adv2.Support, frozen with the fit) on the stored inputs, price, generation and decision
 ms; the controller's action: post head POST iff supported and prediction > 0, else WAIT; keep head PULL iff supported and prediction < 0,
 else KEEP. An unsupported row takes the fallback (WAIT / KEEP); fallbacks are reported separately and are outside the four screen groups.
 d = 1 for POST / KEEP (both heads), 0 for WAIT / PULL.
WEIGHTS: the stored sampling weights w = 1 / (pi N q) (the inverse-probability estimator of the census population; each original / head
 totals 1 / pi_h); weighted means are sum(w x) / sum(w). Two consumed days: no interval anywhere.
READINGS by day, side, head and group (POST / WAIT / KEEP / PULL / fallback, and 'all'):
 - rows, distinct originals (day, market, t0, side) and worlds; exact nonzero labels and their originals; meaningful nonzero (|Y| > 1e-6 c,
   review 315979d2 item 2) originals; support statuses, sources, remaining-horizon bins (pf_r2_teacher.hbin of END - decision);
 - COVERAGE (review 315979d2 item 1): per day / side / head and group (all, and RICH's selected action / fallback), for the source, the
   horizon bin and each of the six support dimensions (pf_r2_teacher.categories: remaining size, queue ahead, generation, inventory, the
   other side's state, price): rows, distinct originals, exact- and meaningful-nonzero originals, weight; the same for the Oct 24-25 training
   rows (group all). Counts only: coverage is not calibration;
 - raw and weighted mean prediction, label and bias;
 - fixed-bin calibration of the prediction (plan v2 edges; the exact zero prediction its own bin);
 - weighted MSE / MAE of each candidate against the zero baseline and the Oct 24-25 weighted training mean of the side / head (fit.json);
   the Spearman correlation of prediction and label (secondary; ties averaged);
 - the local decision value sum(w d Y) / sum(w), the oracle sum(w max(Y, 0)) / sum(w), the regret (oracle - value), the always-1 / always-0
   values; the missed positive value sum(w (1 - d) max(Y, 0)) and the admitted negative value sum(w d min(Y, 0)), overall and for
   |Y| > 30 c; tail event counts and sums. Local pi0-conditional diagnostics, never summed into policy profit.
SCREENS (plan v2 section 9; per day / side / head and group POST, WAIT, KEEP, PULL): support = at least 200 distinct originals and 20
 distinct originals with a meaningful nonzero label (the exact-nonzero count reported beside it), else INSUFFICIENT_SUPPORT; an empty group
 or a zero mean label is NOT_EVALUABLE; sign = weighted mean label > 0 for POST / KEEP and < 0 for WAIT / PULL; amount =
 |mean prediction - mean label| <= 0.05 c + 0.20 |mean label|; PASS needs both, else FAIL. No tolerance fitting, no interval.
LOCAL ACTION-INFORMATION CONTROL (item 10): anchors = the mandatory original-time POST pick (when it exists and is ok) and the one sampled
 KEEP pick per original / side / head; eligible = anchors the candidate decides by its model (supported; fallbacks counted apart). Strata =
 (day, side, head, source, instrument, minute of t0). Each of 500 draws (seed 20261121; strata in sorted key order) places the candidate's
 selected-action count of each stratum on uniformly drawn anchors of that stratum; a draw's value = sum over chosen anchors of w Y / the fixed
 sum of w over all eligible anchors. Lift = the candidate's realized value - the mean draw value; the full draw distribution and the share of
 draws at or above the realized value are reported, per day / head and per day (both heads). Singleton and no-variation strata are counted:
 they add the same amount to every draw. This is additive local selection, not a randomized-policy return.
Usage (run_light): pf_adv2_eval.py selftest | eval
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402
import pf_adv2_sample as AS                              # noqa: E402

C, pd, T, LS = A.C, A.pd, A.T, A.LS
VERSION = 'adv2_eval_v1'
EVAL_PLAN_SHA = '8619eec52bf75b0c1164d5af26d5a6d86bc9fda8d39a6cab2bfbf6b94660e100'
EVAL_DIR = A.OUT / 'eval_labels_v1'
RECEIPT = Path('predictfun/reviews/r3_adv2_20261121')
REPO = HERE.parent.parent
EDGES = (-10., -5., -2., -1., -.5, 0., .5, 1., 2., 5., 10.)
MEANINGFUL, TAIL, ABS_TOL, REL_TOL, MIN_ORIG, MIN_NZ = 1e-6, 30., .05, .20, 200, 20
DRAWS, SEED = 500, 20261121
GROUPS = dict(post=('POST', 'WAIT'), keep=('KEEP', 'PULL'))
COV_DIMS = ('source', 'hbin') + tuple(A.R2.DIMS)


def committed(rel, local):
    """HEAD holds the file `rel` (repo-relative) and it equals `local` byte for byte (the model hashes precede the evaluation)."""
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(local).read_bytes()


def cbin(p):
    """The fixed calibration bin of a prediction (plan v2 edges; the exact zero its own bin)."""
    if p == 0.:
        return '0 (exact)'
    if not np.isfinite(p):
        return 'nonfinite'
    e = (-np.inf,) + EDGES + (np.inf,); i = int(np.searchsorted(EDGES, p, side='left'))
    lo, hi = e[i], e[i + 1]
    return f'({lo:g}, {hi:g}]' if not (lo == 0. or hi == 0.) else (f'({lo:g}, 0)' if hi == 0. else f'(0, {hi:g}]')


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float(np.sum(w * x) / np.sum(w)) if len(x) and np.sum(w) > 0 else None


def spearman(a, b):
    if len(a) < 3 or np.all(np.asarray(a) == np.asarray(a)[0]) or np.all(np.asarray(b) == np.asarray(b)[0]):
        return None
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def with_cats(df, e):
    """The six support categories (pf_r2_teacher.categories) of rows from their stored inputs, price and generation, as columns."""
    x = pd.DataFrame(e['X'], columns=T.FEATS); x['price'] = df.price.values; x['generation'] = df.generation.values
    for k, v in T.categories(x).items():
        df[k] = v
    return df


def train_frame(E):
    """The training rows (keys, label, weight, horizon bin, categories; no prediction) for their coverage."""
    out = []
    for (s, h), e in E.items():
        if len(e['y']):
            df = pd.DataFrame(e['meta']); df['Y'] = e['y']; df['w'] = e['w']; df['hbin'] = T.hbin(A.END - df.decision_ms.astype(float).values)
            out.append(with_cats(df, e))
    return pd.concat(out, ignore_index=True)


def coverage(D, cand=None):
    """Item 1's coverage (header): per day / side / head, group (all; with cand, its selected actions and fallback) and dimension value ->
    rows, distinct originals, exact / meaningful nonzero originals, weight."""
    out = []; o = ['day', 'm', 't_ms', 'side']
    for (day, s, h), g0 in D.groupby(['day', 'side', 'head'], sort=True):
        parts = [('all', g0)] + ([(a, g0[g0[f'act_{cand}'] == a]) for a in GROUPS[h] + ('fallback',)] if cand else [])
        for gname, g in parts:
            for dim in COV_DIMS:
                for val, gv in g.groupby(dim, sort=True):
                    y = gv.Y.values
                    out.append(dict(day=day, side=int(s), head=h, group=gname, dim=dim, value=str(val), rows=int(len(gv)), originals=int(len(gv[o].drop_duplicates())),
                                    nonzero_originals=int(len(gv[y != 0.][o].drop_duplicates())), meaningful_originals=int(len(gv[np.abs(y) > MEANINGFUL][o].drop_duplicates())),
                                    w_sum=float(gv.w.sum())))
    return out


def frame(E, heads, csup, fz):
    """One table of every ok evaluation row with the three candidates' predictions, support status, selected action and d."""
    out = []
    for (s, h), e in E.items():
        if not len(e['y']):
            continue
        df = pd.DataFrame(e['meta']); df['Y'] = e['y']; df['w'] = e['w']
        df['hbin'] = T.hbin(A.END - df.decision_ms.astype(float).values); df = with_cats(df, e)
        sup = []
        for i, m in enumerate(e['meta']):
            row = {c: float(e['X'][i, j]) for j, c in enumerate(T.FEATS)}; row.update(price=m['price'], generation=m['generation'], decision_ms=m['decision_ms'])
            sup.append(csup.check(s, h, row) or 'supported')
        df['support'] = sup
        for cand in A.CANDS:
            p = A.predict(heads[cand][(s, h)], cand, e['X']); df[f'pred_{cand}'] = p
            ok = (df.support == 'supported').values & np.isfinite(p)
            if h == 'post':
                act = np.where(ok & (p > 0), 'POST', 'WAIT')
            else:
                act = np.where(ok & (p < 0), 'PULL', 'KEEP')
            df[f'act_{cand}'] = np.where(ok, act, 'fallback'); df[f'd_{cand}'] = np.where(ok, (act == ('POST' if h == 'post' else 'KEEP')).astype(float), 0. if h == 'post' else 1.)
        df['train_mean'] = fz['models'][f'RICH {s} {h}']['train_weighted_mean_y']
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def group_stats(g, cand):
    """The readings of one group of rows for one candidate."""
    y, w, p, d = g.Y.values, g.w.values, g[f'pred_{cand}'].values, g[f'd_{cand}'].values; n = len(g)
    if not n:
        return dict(rows=0)
    orig = g[['day', 'm', 't_ms', 'side']].drop_duplicates(); nz = g[y != 0.]; mz = g[np.abs(y) > MEANINGFUL]; tm = g.train_mean.values
    big = np.abs(y) > TAIL
    return dict(rows=int(n), originals=int(len(orig)), worlds=int(len(g[['day', 'm', 't_ms']].drop_duplicates())), nonzero=int(len(nz)),
                nonzero_originals=int(len(nz[['day', 'm', 't_ms', 'side']].drop_duplicates())), meaningful_originals=int(len(mz[['day', 'm', 't_ms', 'side']].drop_duplicates())),
                mean_pred=float(p.mean()), mean_y=float(y.mean()), wmean_pred=wmean(p, w), wmean_y=wmean(y, w), wbias=wmean(p - y, w),
                wmse=wmean((p - y) ** 2, w), wmae=wmean(np.abs(p - y), w), wmse_zero=wmean(y ** 2, w), wmae_zero=wmean(np.abs(y), w),
                wmse_train_mean=wmean((tm - y) ** 2, w), wmae_train_mean=wmean(np.abs(tm - y), w), spearman=spearman(p, y),
                value=wmean(d * y, w), oracle=wmean(np.maximum(y, 0.), w), regret=wmean(np.maximum(y, 0.) - d * y, w), always1=wmean(y, w), always0=0.,
                missed_positive=float(np.sum(w * (1 - d) * np.maximum(y, 0.))), admitted_negative=float(np.sum(w * d * np.minimum(y, 0.))),
                missed_positive_tail=float(np.sum(w * (1 - d) * np.maximum(y, 0.) * big)), admitted_negative_tail=float(np.sum(w * d * np.minimum(y, 0.) * big)),
                tail_events=int(big.sum()), tail_sum=float(np.sum(w * y * big)), w_sum=float(w.sum()))


def screen(st, group):
    """Plan v2's prediction screen of one supported-model group -> (verdict, reasons)."""
    if not st.get('rows'):
        return 'NOT_EVALUABLE', 'empty group'
    if st['originals'] < MIN_ORIG or st['meaningful_originals'] < MIN_NZ:
        return 'INSUFFICIENT_SUPPORT', f"{st['originals']} originals, {st['meaningful_originals']} meaningful nonzero originals"
    my, mp = st['wmean_y'], st['wmean_pred']
    if my == 0.:
        return 'NOT_EVALUABLE', 'zero mean label'
    sign_ok = my > 0 if group in ('POST', 'KEEP') else my < 0
    amount_ok = abs(mp - my) <= ABS_TOL + REL_TOL * abs(my)
    return ('PASS' if sign_ok and amount_ok else 'FAIL'), f'sign {"ok" if sign_ok else "wrong"}, amount {"ok" if amount_ok else "outside"}'


def readings(D):
    """Groups x candidates -> (rows of the readings table, rows of the screens table)."""
    rows, scr = [], []
    for (day, s, h), g0 in D.groupby(['day', 'side', 'head'], sort=True):
        for cand in A.CANDS:
            parts = [('all', g0)] + [(a, g0[g0[f'act_{cand}'] == a]) for a in GROUPS[h] + ('fallback',)]
            for gname, g in parts:
                st = group_stats(g, cand); rows.append(dict(day=day, side=int(s), head=h, candidate=cand, group=gname, **st))
                if gname in GROUPS[h]:
                    v, why = screen(st, gname); ex = st.get('nonzero_originals', 0)
                    scr.append(dict(day=day, side=int(s), head=h, candidate=cand, group=gname, verdict=v, why=why, originals=st.get('originals', 0),
                                    meaningful_originals=st.get('meaningful_originals', 0), exact_nonzero_originals=ex,
                                    exact_count_support=bool(st.get('originals', 0) >= MIN_ORIG and ex >= MIN_NZ), wmean_pred=st.get('wmean_pred'), wmean_y=st.get('wmean_y')))
    return rows, scr


def calibration(D):
    out = []
    for (day, s, h), g0 in D.groupby(['day', 'side', 'head'], sort=True):
        for cand in A.CANDS:
            g = g0[g0.support == 'supported']; b = g[f'pred_{cand}'].map(cbin)
            for lab, gb in g.groupby(b, sort=False):
                out.append(dict(day=day, side=int(s), head=h, candidate=cand, bin=lab, rows=len(gb), wmean_pred=wmean(gb[f'pred_{cand}'], gb.w), wmean_y=wmean(gb.Y, gb.w),
                                nonzero=int((gb.Y != 0).sum())))
    return out


def anchors(D, cand):
    """The eligible anchors of one candidate (header) -> frame with the stratum key."""
    a = D[((D['head'] == 'post') & (D.pick == 'mandatory')) | ((D['head'] == 'keep') & (D.pick == 'keep'))]
    a = a[a[f'act_{cand}'] != 'fallback'].copy(); a['minute'] = (a.t_ms // 60000).astype(int)
    a['stratum'] = [A.ck([d, int(s), h, src, int(m), int(mi)]) for d, s, h, src, m, mi in zip(a.day, a.side, a['head'], a.source, a.m, a.minute)]
    return a


def control(D, cand, scope_cols, draws=DRAWS, seed=SEED):
    """The local action-information control of one candidate, per scope (e.g. day / head) -> rows with the realized value, the draw
    distribution and the lift. One generator per scope, strata in sorted key order (deterministic)."""
    a = anchors(D, cand); out = []
    for key, g in a.groupby(scope_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        W = float(g.w.sum()); real = float(np.sum(g.w * g[f'd_{cand}'] * g.Y)) / W; rng = np.random.default_rng(seed); vals = np.zeros(draws)
        strata = sorted(g.groupby('stratum', sort=True), key=lambda kv: kv[0]); single = novar = 0
        for _, sg in strata:
            n_, c_ = len(sg), int(sg[f'd_{cand}'].sum()); wy = (sg.w * sg.Y).values
            if n_ == 1:
                single += 1
            elif c_ in (0, n_):
                novar += 1
            for j in range(draws):
                vals[j] += wy[rng.choice(n_, c_, replace=False)].sum() if 0 < c_ < n_ else (wy.sum() if c_ == n_ else 0.)
        vals /= W
        out.append(dict(zip(scope_cols, key), candidate=cand, anchors=int(len(g)), strata=len(strata), singleton_strata=single, no_variation_strata=novar,
                        realized=real, draw_mean=float(vals.mean()), lift=real - float(vals.mean()), draw_p05=float(np.quantile(vals, .05)),
                        draw_p50=float(np.quantile(vals, .5)), draw_p95=float(np.quantile(vals, .95)), draw_min=float(vals.min()), draw_max=float(vals.max()),
                        share_draws_ge_realized=float(np.mean(vals >= real - 1e-12)), w_sum=W))
    return out


def evaluate(D, st, fz):
    rows, scr = readings(D); cal = calibration(D)
    ctl = []
    for cand in A.CANDS:
        ctl += control(D, cand, ['day', 'head']) + control(D, cand, ['day'])
    sup = D.groupby(['day', 'side', 'head', 'support']).size().rename('rows').reset_index().to_dict('records')
    src = D.groupby(['day', 'side', 'head', 'source']).size().rename('rows').reset_index().to_dict('records')
    hb = D.groupby(['day', 'side', 'head', 'hbin']).size().rename('rows').reset_index().to_dict('records')
    return dict(readings=rows, screens=scr, calibration=cal, control=ctl, support=sup, sources=src, horizon=hb, coverage=coverage(D, 'RICH'), statuses=st)


def cmd_eval(out=A.OUT, rec=RECEIPT):
    """The evaluation (header) -> out/eval/*.csv + eval.json. Refuses unless the fit manifest is committed in HEAD."""
    out = Path(out); dst = out / 'eval'; C.require(not (dst / 'eval.json').exists(), 'the evaluation exists')
    C.require(committed(str(rec / 'fit' / 'fit.json'), out / 'fit.json'), 'the fit manifest is not committed in HEAD: no label is read before the model hashes')
    fz, heads, csup = A.frozen_fit(out)
    acc = json.loads((EVAL_DIR / 'accepted.json').read_text()); C.require(acc['plan_sha256'] == EVAL_PLAN_SHA and acc['accepted'] is True, 'not the frozen evaluation plan\'s accepted sample')
    E, st = A.extract(A.dataset_rows(EVAL_DIR, acc['dataset_sha256']), AS.EVAL_DAYS)
    ex = json.loads((out / 'train' / 'extract.json').read_text()); Et = A.load_train(out)
    C.require(A.manifest_of(Et, ex['statuses'])['heads'] == ex['heads'], 'the training files differ from their extraction manifest')
    D = frame(E, heads, csup, fz); R = evaluate(D, st, fz); R['coverage_train'] = coverage(train_frame(Et)); dst.mkdir(parents=True, exist_ok=True)
    for k in ('readings', 'screens', 'calibration', 'control', 'support', 'sources', 'horizon', 'coverage', 'coverage_train'):
        pd.DataFrame(R[k]).to_csv(dst / f'{k}.csv', index=False)
    keep = ['day', 'm', 't_ms', 'side', 'head', 'pick', 'source', 'decision_ms', 'hbin', 'support', 'w', 'Y'] + [f'{p}_{c}' for c in A.CANDS for p in ('pred', 'act', 'd')]
    D[keep].to_csv(dst / 'predictions.csv.gz', index=False)
    man = dict(version=VERSION, eval_plan_sha256=EVAL_PLAN_SHA, eval_dataset_sha256=acc['dataset_sha256'], fit_sha256=C.sha(out / 'fit.json'), rows=int(len(D)),
               statuses=st, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (dst / 'eval.json').write_text(json.dumps(dict(man, screens=R['screens'], control=R['control']), indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(rows=len(D), screens={f"{r['day']} {r['side']} {r['head']} {r['candidate']} {r['group']}": r['verdict'] for r in R['screens'] if r['candidate'] == 'RICH'}), indent=1))


def selftest():
    import pf_adv2_eval_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['eval']:
        cmd_eval()
    else:
        sys.exit(__doc__)
