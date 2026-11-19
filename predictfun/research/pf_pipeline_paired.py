"""THE PAIRED ACTION-VALUE CANDIDATE: POST / WAIT AND KEEP / PULL ADVANTAGES UNDER A FROZEN CONTINUATION pi0 (2026-11-19, session aea5bfb9).
Plan 0c3d2523 (predictfun/handoffs/20261119_entry_retention_priority.md) steps 2-3; USER: "Build the new model and get the numbers" (step 1, the
no-fit baseline, skipped by the USER). Declared before any label or prediction of this module is read. Consumed days only; nothing is chosen on
results (no threshold, horizon, leaf, round or feature sweep).

pi0, THE FROZEN REFERENCE CONTINUATION (the plan's initial reference, the corrected E1 rule): entry only at the original opportunities (one 30-sh JOIN
 at the best per window-side every 5 s) iff the coherent J POST forecast's mapped value is above 0; while live, pull at the first 50 ms step (from
 +100 ms) whose coherent J RETAIN forecast is below 0 (a step without a coherent forecast keeps); no re-entry before the common END (the next
 opportunity of the window-side is the END); no inventory cap (one order per episode). States: absent -> post-pending (until +58 ms) -> live ->
 cancel-pending (until the landing, +37 ms) -> terminal (cancelled, filled, structurally exited or END). Its J model (pf_pipeline_bench.fit 'J',
 pinned parameters) is fitted on Oct 23 ONLY (the plan's warm-up), frozen, and predicts Oct 24-27 point in time.
EPISODE: rooted at an original opportunity; the common absolute END = its start + 5 s for every branch (the export's labels); value = the local mark
 at END of the fills (cents, M50 clock); fills before an action's earliest effect are common to both branches and cancel.
PAIRED TARGETS on the neutral decision population (every exported POST row with a path and every eligible RETAIN row):
 Delta_keep(j) = R(KEEP through the next tick, then pi0) - R(PULL now)
               = the value of the fills in [landing(j), landing(T*)), T* = pi0's first pull step >= j + 1 (none: up to END)
               = RV(j) - RV(T*), RV(k) = the exported removable value at step k (the fills at or after landing(k)).
   KEEP preserves the order (identity, age, remaining, queue), so pi0's later decisions on the KEEP branch are its forecasts on this order's own
   states, which are the neutral path's (the same order with the same fills until it is cancelled); PULL stays exposed until its landing.
 Delta_post(t) = R(POST now, then pi0) - R(WAIT through the next tick, then pi0) = the order's value under pi0's cancellation - 0.
   Under pi0 nothing can be posted after a WAIT before the common END (its entry schedule is the 5 s opportunity grid), so the WAIT branch carries
   no exposure: this is pi0's definition, not a replacement of a continuation (a pi0 with grid-time re-entry needs new-generation simulation,
   which is not built). A post-only refusal is worth 0.
 Diagnostics: the same differences of the quantity (q) and fill-weighted move (qm) labels (columns pair_v / pair_q / pair_qm; the neutral labels beside them).
CANDIDATES (declared; multiplicity 3 input variants x 4 side / action heads = 12, plus the controls' 16 + 16 J heads):
 A   (primary) direct paired-advantage regression heads, separate bid / ask x POST / KEEP, on the rich causal inputs (pf_pipeline_export.INPUTS,
     91), pinned LightGBM parameters (pf_stop_ev_model.PARAMS, 200 rounds, objective regression): a flagged head / target change; zero action
     boundaries: POST iff the predicted Delta_post > 0, PULL iff the predicted Delta_keep < 0; a tie (or a nonfinite prediction, counted)
     preserves absence / the order
 AF  the same heads on fair_edge and t_to_end only (diagnostic control)       AM  the same heads on the mechanical inputs (diagnostic control)
 Controls: pi0 (the Oct 23 J, E1 rule); NEU = the old neutral-target J fitted on Oct 24-25, the candidate's history (E1 rule).
CHRONOLOGY: pi0 fitted on Oct 23; labels on Oct 24-25 (pi0 never saw them); A / AF / AM and NEU fitted on Oct 24-25 (RETAIN rows thinned to every
 5th step as the benchmark, POST rows all); evaluated on Oct 26-27, whose labels use the same frozen pi0.
G3 DIAGNOSTICS (Oct 26-27; per head, side and day; on the neutral rows and on the rows the candidate's own rule reaches - before its pull):
 the plan's fixed bins [-inf, -10, -5, -2, -1, -0.5, 0, 0.5, 1, 2, 5, 10, inf] c (right-closed; exact 0 apart): n, sum predicted, sum realized,
 bias, MAE; Spearman rank (diagnostic).
 PREDICTION GATE (A, per day and side): the selected POST (pred > 0) realize a positive mean Delta_post; the selected KEEP (pred > 0) a positive
 mean Delta_keep; the selected PULL (pred < 0) a negative mean Delta_keep; and in each of these groups predicted and realized totals have the
 same sign with a ratio in [0.8, 1.25]. An empty group or a zero realized total is NOT-EVALUABLE, never PASS. -> PASS / FAIL / NOT-EVALUABLE.
CONTROLS (any failure -> no output):
 K1 chronology: each model's training days precede every day it predicts; pi0 only Oct 23, the candidates and NEU only Oct 24-25
 K2 no label, identity or endpoint field among any input set
 K3 every export part matches its receipt (checked before anything is written)
 K4 every label reconciles with the keep / pull run's own fills (M50 clock): Delta_keep on every labelled RETAIN row and Delta_post on every
    labelled POST row recomputed from the fills, the landings and pi0's pull steps (and the exported removable values from the same fills)
 K5 every eligible RETAIN row and every POST row with a path is labelled; the other rows are counted by status
 K6 every saved model reloads and reproduces its predictions on sampled rows
OUTPUTS (OUT): labels_<day>.parquet (Oct 24-27: ids, statuses, pi0's forecasts and step T*, the paired and neutral labels), pred_<day>.parquet
 (Oct 26-27: the labels with every policy's forecasts - pi0 / NEU coherent values, A / AF / AM advantages), models/, paired.json / paired.txt.
LIMITS: consumed days (development evidence, not validation; G = 2); single-order episodes (no joint bid / ask, shared inventory or re-post; a
 grid-time re-entry pi0 is not built); labels at the baseline timing B only; queue rule M (C5); replay fills not validated on live orders (C10).
SELFTEST (a synthetic world whose export rows, removable values and fills come from the same planted fills; no data): the labels recomputed by a
 per-order loop from the fills and pi0's saved forecasts; the plan's counterexample and its converse (a neutral hold that loses while KEEP-one-
 tick-then-PULL gains, and the reverse) on the label function; the planted advantage recovered by A and not by AF; refusals by their own control
 with no output: the neutral continuation as the RETAIN and as the POST label, T* from the current step, pi0 fitted with an evaluation day and
 the candidates trained with one, a label among the inputs, a changed export part, a saved model replaced, a row left unlabelled (a crash in place
 of a refusal is a failure); the gate groups, reached rows and fixed bins on planted values. Mutations: pf_pipeline_paired_mutants.py.
Usage (run_light, PF_SLOT=claude): pf_pipeline_paired.py selftest | run
"""
import contextlib
import gc
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_flow_price_common as FC                         # noqa: E402
import pf_keep_pull as KP                                 # noqa: E402
import pf_pipeline_bench as BX                            # noqa: E402  rows, the J fit / prediction and the pinned parameters (imported)
import pf_pipeline_export as EX                           # noqa: E402

C, pd, I = KP.C, KP.pd, KP.I
OUT = Path(I.K.IO) / 'pipeline_paired_20261119'
EXPORT = Path(I.K.IO) / 'pipeline_export_v2_20261119'
KPOUT = KP.OUT
MARK = 'PIPELINE PAIRED'
WARM, LABEL, EVAL = ('20261023',), ('20261024', '20261025'), ('20261026', '20261027')
ADV = {'A': list(EX.INPUTS), 'AF': ['fair_edge', 't_to_end'], 'AM': list(FC.MECHANICAL)}
EDGES = (-10., -5., -2., -1., -.5, 0., .5, 1., 2., 5., 10.)
LAB = ('v', 'q', 'qm')                                    # the paired label's value, quantity and fill-weighted move parts
NEUTRAL_LABEL = NEUTRAL_POST = TSTAR_SAME = False        # planted-error switches (selftest only): the RETAIN / POST label of the neutral hold;
SWAP_SAVED = DROP_ROW = TRAIN_EVAL = False                # T* from the current step; a saved model replaced after fitting; a labelled row dropped;
                                                          # the candidates trained with an evaluation day


def tstar(df):
    """Per RETAIN row (one side and day, any order of rows): T* = pi0's first pull step >= j + 1 within the order (NaN: none) and, per order, its
    first pull step (NaN: none). A pull needs a coherent pi0 forecast below 0."""
    d = df[['oid', 'j']].copy(); d['s'] = np.where(df.pi0_ok.values.astype(bool) & (df.pi0_V.values < 0), df.j.values.astype(float), np.inf)
    d = d.sort_values(['oid', 'j'], kind='stable'); inc = d.iloc[::-1].groupby('oid', sort=False).s.cummin().iloc[::-1]
    nxt = inc if TSTAR_SAME else inc.groupby(d.oid.values).shift(-1)
    t = pd.Series(np.where(np.isfinite(nxt.values), nxt.values, np.nan), index=d.index).reindex(df.index)
    first = d.groupby('oid').s.min(); return t.values, first.where(np.isfinite(first), np.nan)


def label_retain(df):
    """Delta_keep (value, q, qm) per RETAIN row = RV(j) - RV(T*), from the exported removable labels and pi0's T*."""
    ts, first = tstar(df); key = pd.Series(np.arange(len(df)), index=pd.MultiIndex.from_arrays([df.oid.values.astype(int), df.j.values.astype(int)]))
    at = key.reindex(pd.MultiIndex.from_arrays([df.oid.values.astype(int), np.nan_to_num(ts, nan=-1).astype(int)])).values; has = np.isfinite(ts)
    C.require(not (has & np.isnan(at)).any(), 'K4: a pi0 pull step without its row')
    out = {}
    for x, col in zip(LAB, ('value_m50', 'q_m50', 'qm_m50')):
        rv = df[col].values.astype(float); later = np.where(has, rv[np.nan_to_num(at, nan=0).astype(int)], 0.)
        out[f'pair_{x}'] = rv if NEUTRAL_LABEL else rv - later
    return out, ts, first


def label_post(dp, first, r):
    """Delta_post (value, q, qm) per POST row = the neutral POST label minus the removable label at pi0's first pull (0 when refused); r: the
    side-day's RETAIN rows with their removable labels."""
    f = first.reindex(dp.oid.values.astype(int)).values; has = np.isfinite(f)
    m = pd.DataFrame(dict(oid=dp.oid.values.astype(int), j=np.nan_to_num(f, nan=-1).astype(int))).merge(
        r[['oid', 'j', 'value_m50', 'q_m50', 'qm_m50']].assign(oid=lambda z: z.oid.astype(int), j=lambda z: z.j.astype(int)), on=['oid', 'j'], how='left')
    C.require(len(m) == len(dp) and not (has & m.value_m50.isna().values).any(), "K4: a POST row's pi0 pull step without its RETAIN row")
    out = {}
    for x, col in zip(LAB, ('value_m50', 'q_m50', 'qm_m50')):
        later = np.where(has, m[col].values, 0.)
        out[f'pair_{x}'] = np.where(dp.status.values == 'refused_post_only', 0., dp[col].values.astype(float) - (0. if NEUTRAL_POST else later))
    return out, f


def kp_fills(kp, day):
    """The keep / pull run's own orders (side, price, mark) and M50 fills of one day, sorted by order and time."""
    o = pd.read_parquet(kp / day / 'orders.parquet', columns=['oid', 'side', 'p', 'mark']).set_index('oid')
    f = pd.read_parquet(kp / day / 'fills.parquet', columns=['oid', 'clock', 't', 'q']); f = f[f.clock == 50.].sort_values(['oid', 't'], kind='stable')
    sg = np.where(o.side.reindex(f.oid.values).values == 0, 1., -1.); f = f.assign(v=f.q.values * sg * (o.mark.reindex(f.oid.values).values - o.p.reindex(f.oid.values).values))
    return o, f


def groups(oid, j=None):
    """Index groups of the same order (sorted by order, then step)."""
    oid = np.asarray(oid).astype(int); o = np.lexsort((np.asarray(j), oid)) if j is not None else np.argsort(oid, kind='stable')
    return [g for g in np.split(o, np.flatnonzero(np.diff(oid[o])) + 1) if len(g)]


def window_sums(f, oid, lo, hi):
    """Per query: the value of the order's fills with lo <= t < hi (hi NaN: no upper bound), by a per-order search (K4's second path)."""
    out = np.zeros(len(oid)); oid = np.asarray(oid).astype(int); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    g = {k: (v.t.values.astype(float), np.r_[0., np.cumsum(v.v.values)]) for k, v in f.groupby('oid', sort=False)}
    for ix in groups(oid):
        t, cs = g.get(int(oid[ix[0]]), (np.zeros(0), np.zeros(1)))
        ia = np.searchsorted(t, lo[ix], 'left'); ib = np.where(np.isfinite(hi[ix]), np.searchsorted(t, np.nan_to_num(hi[ix], nan=0.), 'left'), len(t))
        out[ix] = cs[np.maximum(ib, ia)] - cs[ia]
    return out


def tstar_search(oid, j, pull):
    """K4's second path for pi0's steps: per row the next pull step > j of its order, and per order the first pull step (NaN: none)."""
    oid, j, pull = np.asarray(oid).astype(int), np.asarray(j).astype(int), np.asarray(pull, bool); nxt = np.full(len(oid), np.nan); first = {}
    for ix in groups(oid, j):
        steps = j[ix][pull[ix]]; k = np.searchsorted(steps, j[ix], 'right')
        nxt[ix] = np.where(k < len(steps), steps[np.minimum(k, max(len(steps) - 1, 0))] if len(steps) else np.nan, np.nan); first[int(oid[ix[0]])] = float(steps[0]) if len(steps) else np.nan
    return nxt, first


def fit_adv(df, target, feats, rounds=None):
    return BX.fit_one(df[feats].to_numpy(np.float32), df[target].to_numpy(float), 'regression', feats, rounds)


def spearman(a, b):
    a, b = pd.Series(np.asarray(a, float)), pd.Series(np.asarray(b, float)); return float(a.rank().corr(b.rank())) if len(a) > 2 else float('nan')


def bins_of(v):
    v = np.asarray(v, float); return np.where(v == 0., -1, np.digitize(v, EDGES, right=True))


def bin_label(i):
    e = (-np.inf,) + EDGES + (np.inf,)
    return 'exactly 0' if i < 0 else f'({e[i]:g}, {e[i + 1]:g}]'


def ratio_ok(p, r):
    return bool(r != 0 and np.sign(p) == np.sign(r) and .8 <= p / r <= 1.25)


def gate_group(pred, real, want_pos):
    """One group of the prediction gate: realized mean of the wanted sign and predicted / realized totals of the same sign within [0.8, 1.25]."""
    if not len(real) or float(np.sum(real)) == 0.:
        return 'NOT-EVALUABLE'
    m = float(np.mean(real)); ok = (m > 0 if want_pos else m < 0) and ratio_ok(float(np.sum(pred)), float(np.sum(real)))
    return 'PASS' if ok else 'FAIL'


def reached(oid, j, pred):
    """The rows the candidate's own rule reaches: up to and including its first pull (pred < 0), per order."""
    d = pd.DataFrame(dict(oid=oid, j=j, s=np.where(np.asarray(pred, float) < 0, j, np.inf))); first = d.groupby('oid').s.transform('min').values
    return d.j.values <= first


def diag(Pq, Rq):
    """G3 diagnostics on one evaluation day and side: per head, the fixed bins (neutral and reached rows), ranks and the prediction gate."""
    out = {}
    for h in ADV:
        pp, pr = Pq[f'{h}_V'].values, Pq.pair_v.values; kr, kv = Rq[f'{h}_V'].values, Rq.pair_v.values; rch = reached(Rq.oid.values, Rq.j.values, kr)
        x = dict(post_rank=spearman(pp, pr), keep_rank=spearman(kr, kv), keep_rank_reached=spearman(kr[rch], kv[rch]), reached_rows=int(rch.sum()), rows=int(len(kr)))
        g = {'POST selected (pred > 0)': gate_group(pp[pp > 0], pr[pp > 0], True), 'KEEP selected (pred > 0)': gate_group(kr[rch & (kr > 0)], kv[rch & (kr > 0)], True),
             'PULL selected (pred < 0)': gate_group(kr[rch & (kr < 0)], kv[rch & (kr < 0)], False)}
        x['gate_groups'] = g; x['gate'] = 'PASS' if all(v == 'PASS' for v in g.values()) else ('NOT-EVALUABLE' if any(v == 'NOT-EVALUABLE' for v in g.values()) and
                                                                                                 not any(v == 'FAIL' for v in g.values()) else 'FAIL')
        x['groups'] = {k: dict(n=int(m.sum()), predicted=float(p_[m].sum()), realized=float(r_[m].sum())) for k, (p_, r_, m) in
                       (('POST selected', (pp, pr, pp > 0)), ('KEEP selected', (kr, kv, rch & (kr > 0))), ('PULL selected', (kr, kv, rch & (kr < 0))))}
        bins = []
        for kind, p_, r_, msk in (('post', pp, pr, np.ones(len(pp), bool)), ('keep neutral', kr, kv, np.ones(len(kr), bool)), ('keep reached', kr, kv, rch)):
            bi = bins_of(p_)
            for b in sorted(set(bi[msk])):
                mb = msk & (bi == b)
                bins.append(dict(kind=kind, bin=bin_label(b), n=int(mb.sum()), predicted=float(p_[mb].sum()), realized=float(r_[mb].sum()),
                                 bias=float((p_[mb] - r_[mb]).mean()), mae=float(np.abs(p_[mb] - r_[mb]).mean())))
        x['bins'] = bins; out[h] = x
    return out


def run(exp=None, kp=None, out=None, rounds=None):
    """The whole pipeline; a refusal at any point leaves no output (the partial folder is removed)."""
    part = Path(out or OUT); part = part.parent / (part.name + '.partial')
    try:
        return _run(exp, kp, out, rounds)
    except BaseException:
        if part.exists():
            shutil.rmtree(part)
        raise


def _run(exp=None, kp=None, out=None, rounds=None):  # noqa: C901
    """pi0 on Oct 23 -> labels on Oct 24-27 -> A / AF / AM and NEU on Oct 24-25 -> predictions and diagnostics on Oct 26-27."""
    start = time.time(); exp = Path(exp or EXPORT); kp = Path(kp or KPOUT); out = Path(out or OUT); C.require(not (out / 'paired.json').exists(), 'output exists')
    bad = [f'K2 {k}: label, identity or endpoint field {x} among the inputs' for k, fs in list(ADV.items()) + [('J', BX.SETS['J'])] for x in fs
           if x.startswith(BX.LABELS) or x.startswith('pair_') or x in BX.FORBIDDEN]
    bad += [f'K1: the warm-up day {w} is not before {d}' for w in WARM for d in LABEL + EVAL if not w < d]
    bad += [f'K1: the label day {w} is not before {d}' for w in LABEL for d in EVAL if not w < d]
    if bad:
        sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad))
    BX.verify_exports(exp, WARM + LABEL + EVAL); part = out.parent / (out.name + '.partial')
    if part.exists():
        shutil.rmtree(part)
    (part / 'models').mkdir(parents=True); files = {}; D = {}; st = {d: {} for d in LABEL + EVAL}; fills = {}
    keepc = ['day', 'oid', 'kind', 'j', 'side', 'status', 'value_m50', 'q_m50', 'qm_m50', 'pi0_V', 'pi0_ok', 'tstar'] + [f'pair_{x}' for x in LAB]
    def compact(df):                                                                        # memory: small types; values stay float64 (K4's tolerance)
        x = df[keepc].copy(); x['oid'] = x.oid.astype(np.int32); x['j'] = x.j.astype(np.int16); x['side'] = x.side.astype(np.int8); x['pi0_V'] = x.pi0_V.astype(np.float32)
        for c_ in ('day', 'kind', 'status'):
            x[c_] = x[c_].astype(str).astype('category')
        return x
    def refuse_now():
        if bad:
            sys.exit(f'{MARK} FAILED -> no output: ' + ' | '.join(bad[:12]))
    def check(d, s, r, p):                                                                  # K4 / K5 on every labelled row of one day and side
        if d not in fills:
            fills[d] = kp_fills(kp, d)[1]
        f = fills[d]; ex = pd.concat([pd.read_parquet(x, columns=['oid', 'kind', 'j', 'status', 'side']) for x in sorted((exp / d).glob('rows_*.parquet'))]); ex = ex[ex.side == s]
        st[d][s] = (ex.kind.astype(str) + ':' + ex.status.astype(str)).value_counts().to_dict()
        ok_ = ((ex.kind == 'retain') & (ex.status == 'eligible')) | ((ex.kind == 'post') & ex.status.isin(('eligible', 'refused_post_only')))
        want = set(zip(ex.oid[ok_].astype(int), ex.kind[ok_].astype(str), ex.j[ok_].astype(int)))
        got = set(zip(r.oid.astype(int), ['retain'] * len(r), r.j.astype(int))) | set(zip(p.oid.astype(int), ['post'] * len(p), p.j.astype(int)))
        if got != want or len(got) != len(r) + len(p):
            bad.append(f'K5 {d} side {s}: {len(want - got):,} eligible rows unlabelled, {len(got - want):,} labelled rows the export does not have')
        nx, fst = tstar_search(r.oid.values, r.j.values, r.pi0_ok.values.astype(bool) & (r.pi0_V.values < 0))     # pi0's steps by a second path
        lo = KP.landing(r.j.values.astype(int)); hi = np.where(np.isfinite(nx), KP.landing(np.nan_to_num(nx, nan=KP.J0).astype(int)), np.nan)
        w = window_sums(f, r.oid.values, lo, hi); off_r = int((~np.isclose(w, r.pair_v.values, rtol=0, atol=1e-6)).sum())
        fp_ = np.array([fst.get(int(o), np.nan) for o in p.oid.values]); hp = np.where(np.isfinite(fp_), KP.landing(np.nan_to_num(fp_, nan=KP.J0).astype(int)), np.nan)
        wp = np.where(p.status.values == 'refused_post_only', 0., window_sums(f, p.oid.values, np.zeros(len(p)), hp)); off_p = int((~np.isclose(wp, p.pair_v.values, rtol=0, atol=1e-6)).sum())
        off_t = int((~((r.tstar.values == nx) | (np.isnan(r.tstar.values) & np.isnan(nx)))).sum())
        if off_r or off_p or off_t:
            bad.append(f'K4 {d} side {s}: {off_r:,} RETAIN and {off_p:,} POST labels differ from the keep / pull run\'s fills ({off_t:,} pi0 steps differ from the second path)')
    def save(name, b):
        f = part / 'models' / f'{name}.txt'; b.save_model(str(f)); files[f.name] = f
    for s in (0, 1):                                                                        # one side at a time (memory: RSS self-guard)
        pi0 = {k: BX.fit(BX.rows_of(exp, WARM, k, True, side=s), 'J', rounds) for k in ('retain', 'post')}
        for k, m in pi0.items():
            for h, b in m.items():
                save(f'pi0_{k}_{s}_{h}', b)
        trn = {'retain': [], 'post': []}
        for d in LABEL + EVAL:
            r = BX.rows_of(exp, (d,), 'retain', False, side=s).reset_index(drop=True); pr = BX.predict(pi0['retain'], r, 'J')
            r['pi0_V'], r['pi0_ok'] = pr['V'], pr['ok']; lr, ts, first = label_retain(r); r = r.assign(**lr, tstar=ts)
            p = BX.rows_of(exp, (d,), 'post', False, side=s).reset_index(drop=True); pp = BX.predict(pi0['post'], p, 'J')
            p['pi0_V'], p['pi0_ok'] = pp['V'], pp['ok']; lp, f = label_post(p, first, r); p = p.assign(**lp, tstar=f)
            if DROP_ROW and d in LABEL:
                r = r.iloc[1:].reset_index(drop=True)
            check(d, s, r, p); refuse_now(); rc_, pc_ = compact(r), compact(p)
            pd.concat([rc_, pc_], ignore_index=True).to_parquet(part / f'labels_{d}_s{s}.parquet', index=False)
            if d in LABEL or (TRAIN_EVAL and d == EVAL[0]):
                trn['retain'].append(r[(r.j - KP.J0) % BX.THIN == 0].copy()); trn['post'].append(p)
            if d in EVAL:
                D[(d, s)] = (rc_, pc_)
            del r, p, pr, pp; gc.collect(); C.guard('pipeline paired labels')
        tr = {k: pd.concat(v, ignore_index=True) for k, v in trn.items()}; del trn
        for k in ('retain', 'post'):
            C.require(set(tr[k].day.astype(str)) <= set(LABEL), f'K1: {k} training rows outside {LABEL}')
        heads = {(h, k): fit_adv(tr[k], 'pair_v', fs, rounds) for h, fs in ADV.items() for k in ('retain', 'post')}
        neu = {k: BX.fit(tr[k], 'J', rounds) for k in ('retain', 'post')}
        for (h, k), b in heads.items():
            save(f'{h}_{"keep" if k == "retain" else "post"}_{s}', b)
        if SWAP_SAVED:                                                                      # planted: the POST head saved as the KEEP head
            heads[('A', 'post')].save_model(str(part / 'models' / f'A_keep_{s}.txt'))
        for k, m in neu.items():
            for h, b in m.items():
                save(f'NEU_{k}_{s}_{h}', b)
        del tr; gc.collect()
        for d in EVAL:
            qs = []
            for k in ('retain', 'post'):
                e = BX.rows_of(exp, (d,), k, False, side=s).reset_index(drop=True); lab = D[(d, s)][0 if k == 'retain' else 1]
                C.require(len(e) == len(lab) and np.array_equal(e.oid.values, lab.oid.values) and np.array_equal(e.j.values, lab.j.values), f'K5: {d} {k} rows changed between reads')
                q = lab.copy(); nn = BX.predict(neu[k], e, 'J'); q['NEU_V'], q['NEU_ok'] = nn['V'], nn['ok']
                for h, fs in ADV.items():
                    q[f'{h}_V'] = heads[(h, k)].predict(e[fs].to_numpy(np.float32), num_threads=1)
                ix = np.linspace(0, len(e) - 1, min(256, len(e)), dtype=int)                   # K6 saved models reload and reproduce
                import lightgbm as lgb
                xs = {h: e[fs].iloc[ix].to_numpy(np.float32) for h, fs in ADV.items()}
                chk = [(f'{h}_{"keep" if k == "retain" else "post"}_{s}', heads[(h, k)], xs[h]) for h in ADV]
                chk += [(f'{nm}_{k}_{s}_{hh}', mm[k][hh], xs['A']) for nm, mm in (('pi0', pi0), ('NEU', neu)) for hh in mm[k]]
                for nm, live, xx in chk:
                    b = lgb.Booster(model_file=str(part / 'models' / f'{nm}.txt'))
                    if not np.allclose(b.predict(xx, num_threads=1), live.predict(xx, num_threads=1), rtol=0, atol=1e-10):
                        bad.append(f'K6 {nm}: the saved model does not reproduce its predictions')
                qs.append(q); del e; gc.collect()
            refuse_now(); pd.concat(qs, ignore_index=True).to_parquet(part / f'pred_{d}_s{s}.parquet', index=False); del qs, D[(d, s)]; gc.collect()
        del heads, neu, pi0; gc.collect(); C.guard('pipeline paired fit'); print(json.dumps(dict(stage='paired side', side=s, seconds=round(time.time() - start), rss=round(I.K.rss_mib()))), flush=True)
    G, lsum = {}, {}
    for d in LABEL + EVAL:                                                                  # one file per day; the per-side parts removed
        lab = pd.concat([pd.read_parquet(part / f'labels_{d}_s{s_}.parquet') for s_ in (0, 1)], ignore_index=True); lab.to_parquet(part / f'labels_{d}.parquet', index=False)
        k_ = lambda k: lab[lab.kind.astype(str) == k]
        lsum[d] = {k: dict(rows=int(len(k_(k))), mean_adv=float(k_(k).pair_v.mean()), sum_adv=float(k_(k).pair_v.sum()), mean_neutral=float(k_(k).value_m50.mean()),
                           pi0_pulls=int(np.isfinite(k_('post').tstar.values.astype(float)).sum()) if k == 'post' else None) for k in ('post', 'retain')}
        for s_ in (0, 1):
            (part / f'labels_{d}_s{s_}.parquet').unlink()
        del lab; gc.collect()
    for d in EVAL:
        Q = pd.concat([pd.read_parquet(part / f'pred_{d}_s{s_}.parquet') for s_ in (0, 1)], ignore_index=True); Q.to_parquet(part / f'pred_{d}.parquet', index=False); G[d] = {}
        for s_ in (0, 1):
            (part / f'pred_{d}_s{s_}.parquet').unlink(); G[d][s_] = diag(Q[(Q.kind.astype(str) == 'post') & (Q.side == s_)], Q[(Q.kind.astype(str) == 'retain') & (Q.side == s_)])
        del Q; gc.collect()
    gate = {d: {s: G[d][s]['A']['gate'] for s in (0, 1)} for d in EVAL}
    R = {'PREDICTION GATE (A, per day and side)': gate, 'PREDICTION GATE (A, all)': 'PASS' if all(v == 'PASS' for x in gate.values() for v in x.values()) else
         ('FAIL' if any(v == 'FAIL' for x in gate.values() for v in x.values()) else 'NOT-EVALUABLE'), 'CONTROLS': 'YES'}
    Lt = [f'THE PAIRED ACTION-VALUE CANDIDATE (pf_pipeline_paired.py; pi0 fitted on {", ".join(WARM)}, labels {", ".join(LABEL + EVAL)}, candidates on {", ".join(LABEL)}, '
          f'evaluated {", ".join(EVAL)}; c; local marks at END, not P&L; CONTROLS: YES (K1-K6))']
    for d in lsum:
        Lt.append(f"  labels {d}: " + '; '.join(f"{k} {v['rows']:,} rows, mean paired {v['mean_adv']:+.3f} c (neutral {v['mean_neutral']:+.3f}), sum {v['sum_adv']:+,.0f}"
                                                  + (f", pi0 pulls {v['pi0_pulls']:,} orders" if v['pi0_pulls'] is not None else '') for k, v in lsum[d].items()))
    for d in EVAL:
        for s in (0, 1):
            for h, x in G[d][s].items():
                gg = x['groups']
                Lt.append(f"  {d} side {s} {h:<2}: POST rank {x['post_rank']:+.3f}, KEEP rank {x['keep_rank']:+.3f} (reached {x['keep_rank_reached']:+.3f}, {x['reached_rows']:,} of {x['rows']:,} rows); "
                          f"gate {x['gate']} {x['gate_groups']}; " + '; '.join(f"{k} n {v['n']:,} predicted {v['predicted']:+,.1f} realized {v['realized']:+,.1f}" for k, v in gg.items()))
    Lt += ['', 'READINGS (declared before this module\'s outputs were read)'] + [f'  {k}: {v}' for k, v in R.items()]
    Lt += ['', 'LIMITS: consumed days (G = 2); single-order episodes; no grid-time re-entry in pi0 (a WAIT carries no exposure before END); labels at B; queue rule M; no live validation.']
    (part / 'paired.txt').write_text('\n'.join(Lt) + '\n')
    C.dump(part / 'paired.json', dict(asof=C.stamp(), readings=R, diagnostics=G, labels=lsum, statuses={d: {str(k): v for k, v in x.items()} for d, x in st.items()}, warm=list(WARM), label_days=list(LABEL), eval=list(EVAL),
                                      sets={k: v for k, v in ADV.items()}, multiplicity=dict(candidate_heads=len(ADV) * 4, control_heads=len(files) - len(ADV) * 4),
                                      params=BX.PARAMS, rounds=rounds or BX.ROUNDS, thin=BX.THIN, exports={d: C.sha(exp / d / 'receipt.json') for d in WARM + LABEL + EVAL},
                                      keep_pull={d: C.sha(kp / d / 'receipt.json') for d in LABEL + EVAL if (kp / d / 'receipt.json').exists()},
                                      models={k: C.sha(v) for k, v in files.items()},
                                      outputs={x.name: C.sha(x) for x in sorted(part.glob('*.parquet'))} | {'paired.txt': C.sha(part / 'paired.txt')},
                                      source_sha256=C.sha(Path(__file__).resolve()), provenance=KP.provenance(), seconds=time.time() - start))
    part.rename(out); print('\n'.join(Lt), flush=True)
    return R, G, lsum


SIG_U, SIG_T = 'bn_age', 'sigma'                          # synthetic: the per-share value's sign and the fill timing ride on two market inputs


def syn_world(exp, kp, n=640, seed=3):
    """A synthetic world for every declared day: per order a per-share value u (-3 c if SIG_U > 0, else +2 c: the END mark), two 5-sh fills early
    (SIG_T > 0: 60-400 ms) or late (1-4.5 s), a structural end; export rows (POST + one RETAIN row per live step from +100 ms; labels from those
    fills: the removable value at landing(j)) and the keep / pull run's orders and M50 fills from the same fills. Every 25th opportunity is a
    post-only refusal (no fills, no RETAIN row)."""
    rng = np.random.default_rng(seed); names = list(EX.INPUTS); iu, it = names.index(SIG_U), names.index(SIG_T)
    for d in WARM + LABEL + EVAL:
        X = rng.normal(0., 1., (n, len(names))).astype(np.float32); o = np.arange(n); side = o % 2; sg = np.where(side == 0, 1., -1.); p = np.where(side == 0, 50., 51.)
        ref = o % 25 == 0; u = np.where(X[:, iu] > 0, -3., 2.); early = X[:, it] > 0; endj = rng.integers(30, 62, n); mark = p + sg * u
        ft = np.sort(np.where(early[:, None], rng.uniform(60., 400., (n, 2)), rng.uniform(1000., 4500., (n, 2))), axis=1)
        ok = (ft < KP.STEP * endj[:, None]) & ~ref[:, None]                                   # fills while live; none for a refusal
        def lab(mask):                                                                      # labels of the fills selected by mask (n_rows x 2)
            q = (mask * 5.).sum(1); v = q * u_r; return dict(q_m50=q, value_m50=v, qm_m50=v - .5 * q, severe_m50=np.maximum(-v - KP.SEVERE, 0.), nofill_m50=(q == 0).astype(float), live_m50=True)
        cols = {k: X[:, i_] for i_, k in enumerate(names)}; cols.update(pf_mid=p + .5 * sg, fp_remaining=np.full(n, 30.))
        u_r = u; post = pd.DataFrame(dict(cols, day=d, oid=o, m=o // 2, side=side, t=5000. * o, p=p, phase=1, mark=mark, kind='post', j=0, decision_ms=5000. * o,
                                          effective_ms=5000. * o + KP.PLACE, endpoint_ms=5000. * o + KP.H, status=np.where(ref, 'refused_post_only', 'eligible'), **lab(ok)))
        nr = np.where(ref, 0, endj - KP.J0); oi = np.repeat(o, nr); jj = np.concatenate([np.arange(KP.J0, e) for e, r_ in zip(endj, ref) if not r_]).astype(int)
        u_r = u[oi]; rc = {k: v[oi] for k, v in cols.items()}; rc['t_to_end'] = (KP.H - KP.STEP * jj).astype(np.float32)
        ret = pd.DataFrame(dict(rc, day=d, oid=oi, m=oi // 2, side=side[oi], t=5000. * oi, p=p[oi], phase=1, mark=mark[oi], kind='retain', j=jj, decision_ms=5000. * oi + KP.STEP * jj,
                                effective_ms=5000. * oi + KP.landing(jj), endpoint_ms=5000. * oi + KP.H, status='eligible', **lab(ok[oi] & (ft[oi] >= KP.landing(jj)[:, None]))))
        (exp / d).mkdir(parents=True); pd.concat([post, ret], ignore_index=True).to_parquet(exp / d / 'rows_000.parquet', index=False)
        C.dump(exp / d / 'receipt.json', dict(outputs={'rows_000.parquet': C.sha(exp / d / 'rows_000.parquet')}))
        (kp / d).mkdir(parents=True); pd.DataFrame(dict(oid=o, side=side, p=p, mark=mark, status=np.where(ref, 'refused_post_only', 'eligible'))).to_parquet(kp / d / 'orders.parquet', index=False)
        fo, fk = np.nonzero(ok); pd.DataFrame(dict(oid=fo, clock=50., t=ft[fo, fk], q=5., src=0)).to_parquet(kp / d / 'fills.parquet', index=False)


def selftest():  # noqa: C901
    """Synthetic only (see the module's SELFTEST)."""
    global WARM, NEUTRAL_LABEL, NEUTRAL_POST, TSTAR_SAME, SWAP_SAVED, DROP_ROW, TRAIN_EVAL
    bad = []; C.require(SIG_U not in ADV['AM'] + ADV['AF'] and SIG_T not in ADV['AM'] + ADV['AF'], 'the synthetic signals must be outside AF / AM')
    t3 = pd.DataFrame(dict(oid=[1, 1, 1, 2, 2, 2], j=[2, 3, 4, 2, 3, 4], value_m50=[-15., -20., -20., 15., 20., 20.], q_m50=[3., 4., 4., 3., 4., 4.], qm_m50=[0.] * 6,
                           pi0_V=[1., -1., 1., 1., -1., 1.], pi0_ok=[True] * 6))                  # the plan's counterexample and its converse
    lr, ts, first = label_retain(t3)
    if not (np.allclose(lr['pair_v'], [5., -20., -20., -5., 20., 20.]) and list(np.nan_to_num(ts, nan=-1)) == [3., -1., -1., 3., -1., -1.] and list(first.values) == [3., 3.]):
        bad.append(f"the label function on the counterexample: {lr['pair_v']} / T* {ts} (want +5 where the neutral hold loses -15, and -5 where it gains +15)")
    gg = (gate_group(np.r_[1., 1.], np.r_[.9, 1.1], True), gate_group(np.r_[1., 1.], np.r_[-1., -1.], True), gate_group(np.r_[3., 3.], np.r_[1., 1.], True),
          gate_group(np.zeros(0), np.zeros(0), True), gate_group(np.r_[1.], np.r_[0.], True), gate_group(np.r_[-1., -1.], np.r_[-1., -1.1], False),
          gate_group(np.r_[-1., -1.], np.r_[1., 1.], False), gate_group(np.r_[-1., -1.], np.r_[-1., -1.], True))   # the prediction gate on planted groups
    if gg != ('PASS', 'FAIL', 'FAIL', 'NOT-EVALUABLE', 'NOT-EVALUABLE', 'PASS', 'FAIL', 'FAIL'):               # (the last: calibrated, but of the wrong sign)
        bad.append(f'the prediction-gate groups on planted values: {gg}')
    rch = reached(np.r_[1, 1, 1, 1, 2, 2], np.r_[2, 3, 4, 5, 2, 3], np.r_[1., -1., 1., -1., 1., 1.])
    if rch.tolist() != [True, True, False, False, True, True] or bins_of(np.r_[0., -10., -9.9, .5, .51, 11.]).tolist() != [-1, 0, 1, 6, 7, 11]:
        bad.append(f'the reached rows {rch.tolist()} or the fixed bins {bins_of(np.r_[0., -10., -9.9, .5, .51, 11.]).tolist()}')
    with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
        tmp = Path(tmp); exp, kp = tmp / 'exp', tmp / 'kp'; syn_world(exp, kp)
        def go(out_):
            with contextlib.redirect_stdout(io.StringIO()):
                return run(exp=exp, kp=kp, out=out_, rounds=30)
        R = None
        try:
            R, G, lsum = go(tmp / 'o')
        except (SystemExit, ValueError) as e:
            bad.append(f'the synthetic run refused: {str(e)[:400]}')
        if R is not None:
            off = 0                                                                         # every label by a per-order loop from the fills and pi0's saved forecasts
            for d in LABEL + EVAL:
                lab = pd.read_parquet(tmp / 'o' / f'labels_{d}.parquet'); fl = pd.read_parquet(kp / d / 'fills.parquet'); od = pd.read_parquet(kp / d / 'orders.parquet').set_index('oid')
                for o, g in lab.groupby('oid'):
                    f = fl[(fl.oid == o) & (fl.clock == 50.)]; u = (1. if od.side.loc[o] == 0 else -1.) * (od.mark.loc[o] - od.p.loc[o])
                    rr = g[g.kind == 'retain'].sort_values('j'); pulls = [int(j) for j, v, k in zip(rr.j, rr.pi0_V, rr.pi0_ok) if k and v < 0]
                    for r in rr.itertuples():
                        nx = next((k for k in pulls if k > r.j), None); hi = KP.landing(nx) if nx is not None else np.inf
                        off += abs(float(((f.t >= KP.landing(r.j)) & (f.t < hi)).sum() * 5. * u) - r.pair_v) > 1e-9
                    pr = g[g.kind == 'post'].iloc[0]; hi = KP.landing(pulls[0]) if pulls else np.inf
                    off += abs((0. if pr.status == 'refused_post_only' else float((f.t < hi).sum() * 5. * u)) - pr.pair_v) > 1e-9
            if off:
                bad.append(f'{off} labels differ from a per-order loop over the fills')
            Q = pd.concat([pd.read_parquet(tmp / 'o' / f'pred_{d}.parquet') for d in EVAL]); qr = Q[Q.kind == 'retain']; qp = Q[Q.kind == 'post']
            if not (R['CONTROLS'] == 'YES' and spearman(qr.A_V, qr.pair_v) > spearman(qr.AF_V, qr.pair_v) + .05 and spearman(qp.A_V, qp.pair_v) > .2
                    and {'pi0_V', 'NEU_V', 'A_V', 'AF_V', 'AM_V', 'pair_v', 'value_m50'} <= set(Q.columns) and (qp.status == 'refused_post_only').any()
                    and (qp[qp.status == 'refused_post_only'].pair_v == 0).all()):
                bad.append(f'synthetic recovery: keep rank A {spearman(qr.A_V, qr.pair_v):+.3f} / AF {spearman(qr.AF_V, qr.pair_v):+.3f}, post rank A {spearman(qp.A_V, qp.pair_v):+.3f}')
        def refused(name, msg, change, undo):
            o_ = tmp / ('z' + str(abs(hash(name)) % 100000)); change()
            try:
                go(o_); bad.append(f'the run accepted {name}')
            except (SystemExit, ValueError) as e:
                if msg not in str(e):
                    bad.append(f'{name} refused by another control: {str(e)[:300]}')
            except Exception as e:                                                          # a crash is not a refusal by a control
                bad.append(f'{name} crashed instead of being refused by a control: {type(e).__name__}: {str(e)[:200]}')
            finally:
                undo()
            if o_.exists() or (o_.parent / (o_.name + '.partial')).exists():
                bad.append(f'a refused run left output ({name})')
        sw = lambda k, v: (lambda: globals().update({k: v}))
        refused('the neutral hold as the RETAIN label', 'K4', sw('NEUTRAL_LABEL', True), sw('NEUTRAL_LABEL', False))
        refused('the neutral hold as the POST label', 'K4', sw('NEUTRAL_POST', True), sw('NEUTRAL_POST', False))
        refused('T* from the current step', 'K4', sw('TSTAR_SAME', True), sw('TSTAR_SAME', False))
        refused('a saved model replaced', 'K6', sw('SWAP_SAVED', True), sw('SWAP_SAVED', False))
        refused('a labelled row dropped', 'K5', sw('DROP_ROW', True), sw('DROP_ROW', False))
        keep_w, keep_am = WARM, list(ADV['AM'])
        refused('pi0 fitted with an evaluation day', 'K1', sw('WARM', WARM + (EVAL[0],)), sw('WARM', keep_w))
        refused('the candidates trained with an evaluation day', 'K1', sw('TRAIN_EVAL', True), sw('TRAIN_EVAL', False))
        refused('a label among the inputs', 'K2', lambda: ADV['AM'].append('pair_v'), lambda: ADV.update(AM=keep_am))
        f0 = exp / LABEL[0] / 'rows_000.parquet'; raw = f0.read_bytes()
        def corrupt():
            g = pd.read_parquet(f0); g.loc[g.index[0], 'value_m50'] += 1.; g.to_parquet(f0, index=False)
        refused('an export part changed after its receipt', 'export part', corrupt, lambda: f0.write_bytes(raw))
    if bad:
        sys.exit(f'{MARK} SELFTEST FAILED: ' + ' | '.join(bad))
    print('pipeline paired selftests passed (synthetic only): the label function on the plan\'s counterexample and its converse; the prediction-gate groups, '
          'the reached rows and the fixed bins on planted values; every label recomputed by a '
          'per-order loop from the fills and pi0\'s saved forecasts; the planted advantage recovered by A (KEEP rank above AF\'s, POST rank positive); refused posts '
          'labelled 0; refusals by their own control with no output: the neutral hold as the RETAIN and as the POST label and T* from the current step (K4), a saved model replaced (K6), '
          'a labelled row dropped (K5), pi0 fitted with an evaluation day and the candidates trained with one (K1), a label among the inputs (K2), an export part '
          'changed after its receipt; a crash in place of a refusal is a failure', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['run']:
        run()
    else:
        sys.exit('usage: pf_pipeline_paired.py selftest | run')
