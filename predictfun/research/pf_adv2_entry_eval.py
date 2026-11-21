"""ADV2-ENTRY-SMOOTH-v1: THE OFFLINE ENTRY DIAGNOSTIC ON THE EXISTING CONSUMED OCT 26-27 POST LABELS (P-2026-005; 2026-11-21, session aea5bfb9).
Review 371aab56's next priority, item 4 (and checklist item 5). Declared and frozen before the fit; it refuses to read a label unless HEAD holds
the receipt copy of this round's fit.json byte for byte. These dates informed this design: all results are adaptive consumed-day development.
POPULATION: exactly R3's accepted Oct 26-27 evaluation dataset (pf_adv2_sample plan 8619eec5..., dataset R3_DATASET), its POST rows (both
 picks), statuses counted. Labels, weights and keys are reused unchanged: before any reading, the rows' keys, Y and w and R3's RICH POST
 predictions recomputed here must equal R3's saved predictions.csv.gz exactly (an altered label, weight, row set or model refuses the run).
CANDIDATES on identical rows: SMOOTH (this round's two heads) and RICH (R3's POST heads: ENTRY_ONLY's entry model, the comparator). Support:
 R3's POST support (unchanged); unsupported rows WAIT (fallback, outside the screen groups). d = 1 for POST, 0 for WAIT; an exact tie WAITs.
READINGS per day / side, groups all / POST / WAIT / fallback (pf_adv2_eval.group_stats): rows, distinct originals, exact and meaningful (|Y| >
 1e-6 c) nonzero originals; raw / weighted mean prediction, label, bias; weighted MSE / MAE against zero and the Oct 24-25 weighted training
 mean; local value sum(w d Y) / sum(w), oracle, regret, missed positive and admitted negative value overall and for |Y| > 30 c; tail events.
 Calibration: fixed bins (pf_adv2_eval.cbin) with each bin's rows, weight total, weighted mean prediction and label (pooling by weights is then
 exact; review 371aab56 item 5). Strata (counts, no sparse-cell skill claim): generation 1 / 2 / >= 3, source, remaining-horizon bin and the
 other side's state, per candidate the posts, weighted mean prediction, local value and weighted mean label. Coverage: pf_adv2_eval.coverage.
SCREENS (plan v2, unchanged): per day / side and group POST / WAIT, 200 distinct originals and 20 meaningful nonzero originals, then sign and
 amount (pf_adv2_eval.screen). CONTROL: pf_adv2_eval.control with the candidate's own action counts, 500 draws, seed 20261121, scopes
 (day, side) and (day).
PREDICATES (declared here, reported separately per day / side for each candidate; lower MSE alone is not success):
 - admissions: the number of supported POST decisions; zero admissions = no demonstrated entry skill, never a successful selector;
 - post_value_positive: the selected POST group's weighted mean label > 0 (it has positive local value);
 - screen_pass: the POST group's screen verdict is PASS;
 - above_draw_mean / above_draw_p95: the realized control value exceeds the matched draws' mean / 95th percentile (scope day, side).
 No favorable side or threshold is selected afterwards; two consumed days carry no significance claim.
Usage (run_light): pf_adv2_entry_eval.py selftest | eval"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402  frozen (freeze 2)
import pf_adv2_entry as N                                # noqa: E402
import pf_adv2_eval as AE                                # noqa: E402  frozen (freeze 2): group_stats, screen, control, cbin, coverage
import pf_adv2_sample as AS                              # noqa: E402  frozen (freeze 1)

C, pd, T = A.C, A.pd, A.T
VERSION = 'adv2_entry_smooth_eval_v1'
R3_DATASET = '902526da20ecea3c224dab403135f701ac47386a72b0779723d36077b2553572'
R3_PRED = Path('predictfun/reviews/r3_adv2_20261121/eval/predictions.csv.gz')
RECEIPT = Path('predictfun/reviews/r4e_entry_smooth_20261121')
CANDS = (N.NAME, 'RICH')
SCOPES = (('day', 'side'), ('day',))
STRATA = ('c_gen', 'source', 'hbin', 'c_other')
KEY = ['day', 'm', 't_ms', 'side', 'pick']


def frame(E, smooth, rich, csup, fz):
    """Every ok POST row with both candidates' predictions, support, selected action and d (identical rows)."""
    out = []
    for (s, h) in N.HEADS:
        e = E[(s, h)]
        if not len(e['y']):
            continue
        df = pd.DataFrame(e['meta']); df['Y'] = e['y']; df['w'] = e['w']; df['hbin'] = T.hbin(A.END - df.decision_ms.astype(float).values); df = AE.with_cats(df, e)
        sup = []
        for i, m in enumerate(e['meta']):
            row = {c: float(e['X'][i, j]) for j, c in enumerate(T.FEATS)}; row.update(price=m['price'], generation=m['generation'], decision_ms=m['decision_ms'])
            sup.append(csup.check(s, h, row) or 'supported')
        df['support'] = sup
        for cand in CANDS:
            p = N.predict(smooth[(s, h)], e['X']) if cand == N.NAME else A.predict(rich['RICH'][(s, h)], 'RICH', e['X']); df[f'pred_{cand}'] = p
            ok = (df.support == 'supported').values & np.isfinite(p); act = np.where(ok & (p > 0), 'POST', 'WAIT')
            df[f'act_{cand}'] = np.where(ok, act, 'fallback'); df[f'd_{cand}'] = np.where(ok, (act == 'POST').astype(float), 0.)
        df['train_mean'] = fz['models'][f'{N.NAME} {s} {h}']['train_weighted_mean_y']
        out.append(df)
    D = pd.concat(out, ignore_index=True); D['day'] = D.day.astype(str)
    return D


def read_r3(p):
    """A saved predictions file, every float parsed exactly (pandas' default C parser is not round-trip exact: ~1 ulp on ~28 % of R3's weights;
    freeze amendment 2026-11-21)."""
    return pd.read_csv(p, float_precision='round_trip')


def r3_identity(D, r3):
    """The existing labels reused unchanged: the same POST row keys, Y, w and RICH predictions as R3's saved predictions -> dict of booleans."""
    r = r3[r3['head'] == 'post'].copy(); r['day'] = r.day.astype(str); a = D.set_index(KEY).sort_index(); b = r.set_index(KEY).sort_index()
    same_keys = bool(len(a) == len(b) and a.index.equals(b.index) and a.index.is_unique)
    eq = lambda c1, c2: bool(same_keys and np.array_equal(a[c1].values.astype(float), b[c2].values.astype(float)))
    return dict(rows=int(len(a)), r3_rows=int(len(b)), same_keys=same_keys, y_equal=eq('Y', 'Y'), w_equal=eq('w', 'w'), rich_pred_equal=eq('pred_RICH', 'pred_RICH'))


def readings(D):
    rows, scr = [], []
    for (day, s), g0 in D.groupby(['day', 'side'], sort=True):
        for cand in CANDS:
            for gname, g in [('all', g0)] + [(a, g0[g0[f'act_{cand}'] == a]) for a in ('POST', 'WAIT', 'fallback')]:
                st = AE.group_stats(g, cand); rows.append(dict(day=day, side=int(s), head='post', candidate=cand, group=gname, **st))
                if gname in ('POST', 'WAIT'):
                    v, why = AE.screen(st, gname); ex = st.get('nonzero_originals', 0)
                    scr.append(dict(day=day, side=int(s), head='post', candidate=cand, group=gname, verdict=v, why=why, originals=st.get('originals', 0),
                                    meaningful_originals=st.get('meaningful_originals', 0), exact_nonzero_originals=ex,
                                    exact_count_support=bool(st.get('originals', 0) >= AE.MIN_ORIG and ex >= AE.MIN_NZ), wmean_pred=st.get('wmean_pred'), wmean_y=st.get('wmean_y')))
    return rows, scr


def calibration(D):
    """Fixed bins per day / side / candidate with each bin's weight total (review 371aab56 item 5)."""
    out = []
    for (day, s), g0 in D.groupby(['day', 'side'], sort=True):
        g = g0[g0.support == 'supported']
        for cand in CANDS:
            b = g[f'pred_{cand}'].map(AE.cbin)
            for lab, gb in g.groupby(b, sort=False):
                out.append(dict(day=day, side=int(s), candidate=cand, bin=lab, rows=int(len(gb)), w_sum=float(gb.w.sum()), wmean_pred=AE.wmean(gb[f'pred_{cand}'], gb.w),
                                wmean_y=AE.wmean(gb.Y, gb.w), nonzero=int((gb.Y != 0).sum())))
    return out


def strata(D):
    """Per day / side, dimension and value: counts and each candidate's posts, weighted mean prediction, local value and weighted mean label."""
    out = []; o = ['day', 'm', 't_ms', 'side']
    for (day, s), g0 in D.groupby(['day', 'side'], sort=True):
        for dim in STRATA:
            for val, g in g0.groupby(dim, sort=True):
                y, w = g.Y.values, g.w.values
                r = dict(day=day, side=int(s), dim=dim, value=str(val), rows=int(len(g)), originals=int(len(g[o].drop_duplicates())),
                         meaningful_originals=int(len(g[np.abs(y) > AE.MEANINGFUL][o].drop_duplicates())), w_sum=float(w.sum()), wmean_y=AE.wmean(y, w))
                for cand in CANDS:
                    d = g[f'd_{cand}'].values
                    r.update({f'posts_{cand}': int(d.sum()), f'wmean_pred_{cand}': AE.wmean(g[f'pred_{cand}'], w), f'value_{cand}': AE.wmean(d * y, w)})
                out.append(r)
    return out


def predicates(rd, sc, ct):
    """The declared per day / side predicates of each candidate (header)."""
    rd, sc, ct = pd.DataFrame(rd), pd.DataFrame(sc), pd.DataFrame(ct); out = []
    for (day, s), x in rd[rd.group == 'all'].groupby(['day', 'side'], sort=True):
        for cand in CANDS:
            pg = rd[(rd.day == day) & (rd.side == s) & (rd.candidate == cand) & (rd.group == 'POST')].iloc[0]
            n = int(pg.rows) if pd.notna(pg.rows) else 0
            v = sc[(sc.day == day) & (sc.side == s) & (sc.candidate == cand) & (sc.group == 'POST')].verdict.iloc[0]
            c = ct[(ct.day == day) & (ct.side == s) & (ct.candidate == cand)]
            real, dm, p95 = (float(c.realized.iloc[0]), float(c.draw_mean.iloc[0]), float(c.draw_p95.iloc[0])) if len(c) else (np.nan, np.nan, np.nan)
            out.append(dict(day=day, side=int(s), candidate=cand, admissions=n, status='no_admissions' if n == 0 else 'admitted',
                            post_value_positive=bool(n > 0 and float(pg.wmean_y) > 0), screen_pass=bool(v == 'PASS'), screen=v,
                            above_draw_mean=bool(n > 0 and real > dm), above_draw_p95=bool(n > 0 and real > p95), realized=real, draw_mean=dm, draw_p95=p95))
    return out


def evaluate(D):
    rows, scr = readings(D); ctl = []
    for cand in CANDS:
        for sc_ in SCOPES:
            ctl += AE.control(D, cand, list(sc_))
    return dict(readings=rows, screens=scr, calibration=calibration(D), control=ctl, strata=strata(D), coverage=AE.coverage(D, N.NAME),
                predicates=predicates(rows, scr, [c for c in ctl if 'side' in c]))


def cmd_eval(out=N.OUT, rec=RECEIPT):
    """The evaluation (header) -> out/eval/*.csv + eval.json. Refuses unless the fit manifest is committed and the reused rows are R3's."""
    out = Path(out); dst = out / 'eval'; C.require(not (dst / 'eval.json').exists(), 'the evaluation exists')
    C.require(N.committed_equal(rec / 'fit' / 'fit.json', out / 'fit.json'), 'the fit manifest is not committed in HEAD: no label is read before the model hashes')
    fz, smooth, rich, csup = N.frozen(out)
    acc = json.loads((AE.EVAL_DIR / 'accepted.json').read_text())
    C.require(acc['accepted'] is True and acc['plan_sha256'] == AE.EVAL_PLAN_SHA and acc['dataset_sha256'] == R3_DATASET, "not R3's accepted evaluation dataset")
    E, st = A.extract(A.dataset_rows(AE.EVAL_DIR, R3_DATASET), AS.EVAL_DAYS); E = {h: E[h] for h in N.HEADS}
    D = frame(E, smooth, rich, csup, fz); idn = r3_identity(D, read_r3(N.REPO / R3_PRED))
    C.require(all(v for k, v in idn.items() if k not in ('rows', 'r3_rows')), f"the reused rows / labels / weights / RICH predictions differ from R3's: {idn}")
    R = evaluate(D); dst.mkdir(parents=True, exist_ok=True)
    for k in ('readings', 'screens', 'calibration', 'control', 'strata', 'coverage', 'predicates'):
        pd.DataFrame(R[k]).to_csv(dst / f'{k}.csv', index=False)
    keep = KEY + ['head', 'source', 'decision_ms', 'hbin', 'support', 'w', 'Y'] + [f'{p}_{c}' for c in CANDS for p in ('pred', 'act', 'd')]
    D[keep].to_csv(dst / 'predictions.csv.gz', index=False)
    man = dict(version=VERSION, r3_dataset_sha256=R3_DATASET, eval_plan_sha256=AE.EVAL_PLAN_SHA, fit_sha256=C.sha(out / 'fit.json'), identity=idn, statuses=st,
               source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (dst / 'eval.json').write_text(json.dumps(dict(man, predicates=R['predicates']), indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(identity=idn, predicates=R['predicates']), indent=1, default=str))


def selftest():
    import pf_adv2_entry_eval_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['eval']:
        cmd_eval()
    else:
        sys.exit(__doc__)
