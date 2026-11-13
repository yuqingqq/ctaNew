"""PULL-DEPTH (2026-11-13, session ce68ebe0). USER: "I mean if we can pull the quotes only at risk, for example if current price is 0.5, and we
fired some danger signal, it will probably go to like 0.48, we can pull 0.5, 0.49 and 0.48, keep our best bid at 0.47, so we still keep our
position, then the other side needs to repost. Can we check 2 sides as well, because different side have different issues, in the above case,
we need to repost our best ask again, everyone else should also repost, in this case, I think latency is more important? But for bid, we can
still keep our bid at 0.47 there to be in the front, can we also check different side behavior in these cases for top makers, how do they
handle". After QUEUE-PLACE (881e8b27): orders already at a price when it becomes the best earn +0.20 c/sh at 57% STOP vs ~0 for later joins,
and the guard pull (every rung, on v <= -0.10) is what takes us out of the price when the new best forms.

9 consumed replay days (10-23..27, 10-31, 11-01..03), first 60 s; the guard ladder (LDG: 30 sh, rungs d = 0..5 ticks behind the known best,
every rung pulled on the guard and re-placed once clear for 250 ms) as in DEEP-PULL / TIMING-SPLIT.

PART A, how far the danger move goes. Rows: every 50 ms grid step with the side's guard on (v <= -0.10) and a valid known book. Target mv =
 ticks the side's best goes against us (bids: down; asks: up) beyond the known best, over the frames stamped after the known frame up to
 t + 1 s (invalid frames skipped; mv250 the same to t + 250 ms, printed). Features at the step: S4's 37 inputs (step-5 set + guard value,
 own / opposite 250 ms depth changes, activity, Binance imbalance change), the side, FRONT-PULL's per-day P (fitted without the day), the
 known frame's own sizes at levels 1-5, the tick gaps of levels 2-5 to the best, the spread, cumulative size to levels 1-3, steps since the
 guard came on. Model: LightGBM binary for mv >= j, j = 1..5, LEAVE-ONE-DAY-OUT (fit on the other 8 days, training rows subsampled to 300,000
 with a fixed seed; parameters pinned below), predictions made cumulative (P_j = min(P_j, P_j-1)).
PART B, pull only the prices at risk (the USER's rule): while the guard is on, a resting rung d ticks behind the known best is pulled if the
 move is likely to reach its price: P(mv >= d) >= tau (d = 0, the best, always; d >= 5 uses P_5, which overstates the risk). Kept rungs stay
 in their queue; entries stay blocked while the guard is on (as LDG). Arms (each rung its own lane, S4's rest lane with idle-only triggers
 and a decision callback):
   LDG    every rung pulled on the guard (reference)
   PD10 / PD25   pull to the predicted depth, tau = 0.10 / 0.25
   PDF2 / PDF3   fixed depth: pull d < 2 / d < 3 (a 1- / 2-tick move assumed; PDF2 is close to KEEP-PLACE's KPN), keep the rest
 Fill clocks M 0 / M 50. Values as TIMING-SPLIT: c/sh at 1 s / 5 s from the trade's anchor; $/day w/o young (order >= 100 ms) by trade
 timing (C12: WS = the taker order has a WS-timed leg; E2 = estimated); per at-best fill whether our order was at the price before it
 became the best (PRE, QUEUE-PLACE's run on the simulator's clock).
PART C, the two sides and the top makers (real legs; groups by daily maker volume, B12). Event = guard onset on side s (first step on).
   adverse side s: the first level that becomes s's best by FALLBACK (the level in front went away) in (t0 - 100 ms, t0 + 1 s]
   other side 1-s: the first level that becomes 1-s's best by IMPROVEMENT (a better price appeared) in the same interval: the repost race
   calm: every other fallback / improve birth of the first minute with no onset on either side in [birth - 1 s, birth + 100 ms)
 Per level (while it stays the best, at most 2 s): the shown size at the birth frame and +100 / 250 / 500 / 1000 ms (how fast the queue
 forms); real WS-timed fills: group, time since the birth, c/sh at 1 s, STOP / THROUGH; the FIRST fill (the front): group, and PROOF = its
 order (certain hash, ORDERS' map) had a fill at that price before the onset (calm: > 250 ms before the birth): it rested there and stayed
 through the move (a lower bound). Ours at those levels (M 50): LDG / PD10 / PD25 fills (all rungs) on the adverse side; on the other side
 rung 0 BASE (= LDG rung 0) and FAST (BEST-GAP's 5 ms order path + fastest feed): our order's go-live after the birth, c/sh, STOP.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule >= 7 of 9; a day with < 30 fills in a compared cell does not count):
 R490 PREDICTABLE (j = 1, 2, 3, each): out-of-day AUC of P_j >= 0.70 pooled and >= 0.65 on >= 7 days -> PREDICTABLE; else WEAK.
 R491 PULL-TO-DEPTH (PD10, PD25, each, against LDG): $/day w/o young at 1 s on WS-timed trades, higher at M 0 AND M 50 on >= 7 days ->
      HELPS; lower at both on >= 7 -> HURTS; else NOT-SHOWN. All-trade and E2 $/day, M 50 at 5 s and PDF2 / PDF3 printed.
 R492 FRONT (descriptive): per arm, at-best fills from PRE orders (share, c/sh, STOP), the share of deep fills at high P (DEPTH-TIMING's
      tercile line), pulls.
 R493 ADVERSE-FRONT: the top 5's share of FIRST fills at adverse-side levels against their share of all real at-best WS fill shares: ratio
      >= 1.2 pooled and higher on >= 7 days -> TOP-FRONT-AFTER-MOVE; <= 1 / 1.2 and lower on >= 7 -> TOP-NOT-FRONT; else PROPORTIONAL.
      PROOF shares (top 5 / field) and calm fallback births printed.
 R494 OTHER-SIDE-EARLY: real WS fills at other-side levels, c/sh at 1 s of fills < 100 ms after the birth minus fills >= 500 ms after: > 0
      on >= 7 days -> EARLY-PAYS; < 0 on >= 7 -> EARLY-DOES-NOT-PAY; else NOT-SHOWN. Queue growth (median size at +250 ms / at the birth)
      and calm improve births printed.
 R495 OUR-SPEED-THERE: ours at other-side levels, FAST minus BASE c/sh at 1 s (WS, aged) > 0 on >= 7 days -> SPEED-PAYS-HERE; < 0 on >= 7
      -> SPEED-HURTS-HERE; else NOT-SHOWN; calm improve births printed.
What the outcomes mean, fixed in advance: R490 WEAK = the depth of the move cannot be told at the signal well enough to keep deeper quotes;
R491 HELPS = keeping quotes beyond the predicted move beats pulling every layer in this replay (a candidate, not a result: consumed days);
R493 TOP-FRONT-AFTER-MOVE = the top makers are disproportionately first in the queue of the new best after a danger move (consistent with
keeping deeper orders through it); R494 / R495 = whether the repost race on the other side rewards being first, and whether our faster path
captures it.
Controls (any failure -> no output): (i) PD with tau = 0 (the callback engaged, every rung pulled) equals LDG fill for fill and order for
order on every window-side, rung and clock (off-parity through the override); PD25 differs from LDG on some window-sides (the override can
fire); (ii) LDG's fills equal TIMING-SPLIT's LDG rows (count and value per day and clock, checked in the report); (iii) FRONT-PULL's P
aligned with its fit rows; BEST-GAP's clock constants restored after FAST; (iv) synthetic cases (the move on both sides and an invalid known
frame; the pull decision for both rules; onsets) with three planted errors (move sign flipped, decision inverted, every guard step an onset)
that must each fail a case; (v) the fit sees no row of its held-out day (asserted); memory self-guard.
Limits fixed in advance: replay rule M (B1a); consumed days, exploratory, five candidate arms; the guard and our P are OUR signals, the top
makers' own signals are unknown; their orders are seen only through fills (PROOF is a lower bound); the rungs are independent orders; a
level's first fill is the front only up to the leg order inside one taker order (not used).
Usage (run_light, PF_SLOT=claude): pf_pull_depth.py selftest | rows <day> | fit | eval <day> | report; driver run_pull_depth.sh
"""
import bisect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_qr_queue_race as QR                             # noqa: E402  (book runs)
import pf_front_pull as FP                                # noqa: E402  (per-day P models)
import pf_best_gap as BG                                  # noqa: E402  (FAST order path)
import pf_queue_place as QP                               # noqa: E402  (run_at)
import pf_timing_split as TS                              # noqa: E402  (WS / E2 flag)

DAYS9 = K.DAYS9
OUTD = f'{K.IO}/pull_depth_20261113'
ORD = f'{K.IO}/orders_20261113'
TSD = f'{K.IO}/timing_split_20261113'
JS = (1, 2, 3, 4, 5); H_MS = 1000.0; H_SHORT = 250.0
TAUS = {'PD10': 0.10, 'PD25': 0.25}; FIXK = {'PDF2': 2, 'PDF3': 3}
ARMS = ('LDG', 'PD10', 'PD25', 'PDF2', 'PDF3')
CFGS = [('M', 0.0), ('M', 50.0)]
SIZE = 30.0; RUNGS = range(6)
TERC = (0.09040728596733005, 0.4236075378593026)          # DEPTH-TIMING's terciles of P over real deep fills
WIN_PRE = 100.0; WIN_POST = 1000.0; LEVEL_MS = 2000.0; PROOF_MS = 250.0
GROW = (0.0, 100.0, 250.0, 500.0, 1000.0)
GAPB = (('< 100 ms', -1e18, 100.0), ('100-500 ms', 100.0, 500.0), ('>= 500 ms', 500.0, 1e18))
SUB_N = 300000; N_ROUNDS = 200
LGB_PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, seed=1, num_threads=1, verbose=-1)
MIN_FILLS = 30
INF = float('inf'); NAN = float('nan')
FEAT = [f'x{i}' for i in range(37)] + ['side', 'P', 's1', 's2', 's3', 's4', 's5', 'g2', 'g3', 'g4', 'g5', 'spread', 'c1', 'c2', 'c3', 'gage']
MUT_MOVE_SIGN = MUT_DEC_INV = MUT_ONSET_ANY = 0           # planted errors (self-test only)
MEM_CAP_MIB = 2048.0


# ---------------------------------------------------------------- pieces
def move_ticks(b0, valid, tfl, side, jk, t, h=H_MS):
    """Ticks the side's best goes against us beyond the known frame jk's best, over frames stamped after it up to t + h."""
    if jk < 0 or not valid[jk]:
        return NAN
    j1 = bisect.bisect_right(tfl, t + h)
    seg = b0[jk + 1:j1][valid[jk + 1:j1]]
    if not len(seg):
        return 0.0
    m = (b0[jk] - seg.min()) if side == 0 else (seg.max() - b0[jk])
    if MUT_MOVE_SIGN:
        m = -m
    return float(max(0, m))


def onsets(vg):
    return [k for k in range(len(vg)) if vg[k] and (MUT_ONSET_ANY or k == 0 or not vg[k - 1])]


def pd_dec(vg, best, side, Pm, tau=None, kfix=None):
    """While the guard is on: pull a resting rung d ticks behind the known best if P(mv >= d) >= tau (or d < kfix); d <= 0 always."""
    def dec(k, t, cur):
        if not vg[k]:
            return False
        b = best[k]
        if b < 0:
            return True
        d = (b - cur['pt']) if side == 0 else (cur['pt'] - b)
        if d <= 0:
            return True
        if kfix is not None:
            r = d < kfix
        else:
            p = Pm[k, min(d, len(JS)) - 1]
            r = (not (p == p)) or p >= tau
        return (not r) if MUT_DEC_INV else r
    return dec


def av_P(w, mdl):
    nb = len(K.G32); n = w['n']
    Xw = np.column_stack([w['X'], np.full(n, float(w['side']))]).astype(np.float32)
    P = mdl.predict(Xw) if n else np.zeros(0); vk = w['X'][:, nb] if n else np.zeros(0)
    return np.where(~np.isfinite(vk) | (vk == -1.0) | (np.asarray(w['best']) < 0) | (w['jk'] < 0), np.nan, P)


def feats(w, k, P, gage):
    jk = int(w['jk'][k]); P5, S5, opp = w['P5'], w['S5'], w['opp']; b = int(P5[jk, 0])
    sz = [float(S5[jk, i]) if P5[jk, i] >= 0 else 0.0 for i in range(5)]
    gaps = [float(abs(int(P5[jk, i]) - b)) if P5[jk, i] >= 0 else -1.0 for i in range(1, 5)]
    spr = float(abs(int(opp[jk]) - b)) if opp[jk] >= 0 else -1.0
    cum = np.cumsum(sz)[:3].tolist()
    return list(np.asarray(w['X'][k], float)) + [float(w['side']), float(P[k])] + sz + gaps + [spr] + cum + [float(gage)]


def selftest():
    bad = []
    tfl = [0.0, 50.0, 100.0, 150.0, 200.0]
    b0 = np.array([50, 50, 49, 48, 49]); valid = np.array([True] * 5)
    if move_ticks(b0, valid, tfl, 0, 0, 10.0) != 2.0 or move_ticks(b0, valid, tfl, 0, 0, 10.0, h=60.0) != 0.0:
        bad.append('bid move')
    a0 = np.array([50, 51, 52, 52, 50])
    if move_ticks(a0, valid, tfl, 1, 0, 10.0) != 2.0:
        bad.append('ask move')
    if not np.isnan(move_ticks(b0, np.array([False] + [True] * 4), tfl, 0, 0, 10.0)):
        bad.append('invalid known frame must be nan')
    Pm = np.array([[0.9, 0.3, 0.05, 0.01, 0.0]]); best = [50]
    c = lambda pt: dict(pt=pt)
    d25 = pd_dec([True], best, 0, Pm, tau=0.25)
    if [d25(0, 0.0, c(p)) for p in (50, 49, 48, 47)] != [True, True, True, False]:
        bad.append(f'tau rule {[d25(0, 0.0, c(p)) for p in (50, 49, 48, 47)]}')
    da = pd_dec([True], [50], 1, Pm, tau=0.25)
    if [da(0, 0.0, c(p)) for p in (50, 51, 52, 53)] != [True, True, True, False]:
        bad.append('ask tau rule')
    f3 = pd_dec([True], best, 0, Pm, kfix=3)
    if [f3(0, 0.0, c(p)) for p in (50, 48, 47)] != [True, True, False] or pd_dec([False], best, 0, Pm, tau=0.25)(0, 0.0, c(50)):
        bad.append('fixed rule / guard off')
    if pd_dec([True], best, 0, np.array([[np.nan] * 5]), tau=0.25)(0, 0.0, c(47)) is not True:
        bad.append('missing prediction must pull')
    if onsets([False, True, True, False, True]) != [1, 4]:
        bad.append('onsets')
    return bad


def mutation_check():
    out = {}
    for nm in ('MUT_MOVE_SIGN', 'MUT_DEC_INV', 'MUT_ONSET_ANY'):
        globals()[nm] = 1
        try:
            out[nm] = bool(selftest())
        finally:
            globals()[nm] = 0
    return out


def all_selftests():
    b = selftest(); m = mutation_check()
    if b or not all(m.values()):
        sys.exit(f'PULL-DEPTH SELFTEST FAILED -> no output: {b} planted errors caught {m}')
    return m


# ---------------------------------------------------------------- part A: rows and the leave-one-day-out fit
def rows_day(day):
    import lightgbm as lgb
    all_selftests()
    T0 = time.time(); D = K.load_day(day); mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    out = []; st = dict(window_sides=0, guard_steps=0, invalid_book=0)
    for w in K.window_sides(D):
        st['window_sides'] += 1
        n = w['n']
        if not n:
            continue
        P = av_P(w, mdl); vg = np.asarray(w['vkl'], float) <= K.GUARD_PULL
        b0, valid, rs, kind = QR.runs(w['tf'], w['P5'], w['opp'], w['side']); tfl = w['tf'].tolist(); gage = 0
        for k in range(n):
            if not vg[k]:
                gage = 0; continue
            gage += 1; st['guard_steps'] += 1
            jk = int(w['jk'][k])
            if jk < 0 or not valid[jk] or w['best'][k] < 0:
                st['invalid_book'] += 1; continue
            t = float(w['tg'][k])
            out.append([int(day), int(w['m']), int(w['side']), k, t, int(gage == 1)] + feats(w, k, P, gage) +
                       [move_ticks(b0, valid, tfl, w['side'], jk, t), move_ticks(b0, valid, tfl, w['side'], jk, t, H_SHORT)])
    os.makedirs(OUTD, exist_ok=True)
    R = pd.DataFrame(out, columns=['day', 'm', 'side', 'k', 't', 'onset'] + FEAT + ['mv1000', 'mv250'])
    R.to_parquet(f'{OUTD}/rows_{day}.parquet', index=False)
    st['rows'] = len(R); json.dump(st, open(f'{OUTD}/rows_{day}.json', 'w'))
    print(f'{day}: {st} in {time.time() - T0:.0f} s', flush=True)


def auc(y, p):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def fit():
    import lightgbm as lgb
    all_selftests()
    R = {d: pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=['day', 'm', 'side', 'k', 'mv1000'] + FEAT).astype({c: 'float32' for c in FEAT}) for d in DAYS9}
    rng = np.random.default_rng(7); A = {}; base = {}
    for d in DAYS9:
        tr = pd.concat([R[x] for x in DAYS9 if x != d], ignore_index=True)
        assert not (tr.day == int(d)).any(), 'held-out day in the training rows'
        tr = tr[np.isfinite(tr.mv1000.values)]
        if len(tr) > SUB_N:
            tr = tr.iloc[np.sort(rng.choice(len(tr), SUB_N, replace=False))]
        te = R[d]; Xtr = tr[FEAT].values.astype(np.float32); Xte = te[FEAT].values.astype(np.float32); pred = {}
        prev = np.ones(len(te))
        for j in JS:
            b = lgb.train(LGB_PARAMS, lgb.Dataset(Xtr, (tr.mv1000.values >= j).astype(int)), N_ROUNDS)
            p = np.minimum(b.predict(Xte), prev); prev = p; pred[f'P{j}'] = p
            ok = np.isfinite(te.mv1000.values)
            A[f'{d}|{j}'] = auc(te.mv1000.values[ok] >= j, p[ok]); base[f'{d}|{j}'] = float((te.mv1000.values[ok] >= j).mean())
            b.save_model(f'{OUTD}/mv_{d}_j{j}.txt')
        pd.DataFrame(dict(m=te.m.values, side=te.side.values, k=te.k.values, **pred)).to_parquet(f'{OUTD}/pred_{d}.parquet', index=False)
        print(f'fit {d}: train {len(tr)} rows, AUC ' + ' '.join(f'j{j} {A[f"{d}|{j}"]:.3f}' for j in JS), flush=True)
    json.dump(dict(auc=A, base_rate=base, params=LGB_PARAMS, rounds=N_ROUNDS, sub_n=SUB_N), open(f'{OUTD}/fit.json', 'w'), indent=1)


# ---------------------------------------------------------------- parts B and C: one day
def level_stats(w, x, p, tb, te_run, t_ref, ctx, ev_id, kind_ev, day, ours):
    """Queue growth, real fills and our fills at level p of side x born at tb (run ends at te_run)."""
    tfl = ctx['tfl'][x]; P5, S5 = w['P5'], w['S5']; tend = min(te_run, tb + LEVEL_MS)
    grow = []
    for g in GROW:
        j = bisect.bisect_right(tfl, tb + g) - 1
        grow.append(float((S5[j] * (P5[j] == p)).sum()) if (j >= 0 and tfl[j] < te_run) else NAN)
    ev = (day, ev_id, kind_ev, x, int(p), float(tb), float(te_run - tb)) + tuple(grow)
    fr = []
    tf, mff = w['tf_mff']; sg = w['sg']
    mid1 = lambda t: (lambda j: mff[j] if j >= 0 else NAN)(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1)
    anc = w['anc']; ptl = w['ptl']; lrow = ctx['lrow'][x]; src = ctx['src'][x]
    i0 = int(np.searchsorted(anc, tb, side='right')); i1 = int(np.searchsorted(anc, tend, side='right'))
    first = True
    for i in range(i0, i1):
        if ptl[i] != p or not src[i] or not (w['ws'] <= anc[i] < w['ws'] + 61000.0):
            continue
        r_ = lrow[i]; t = float(anc[i]); g_ = ctx['grp'](r_); oh = ctx['ohash'].get(r_)
        proof = bool(oh is not None and ctx['ohmin'].get(oh, INF) < t_ref)
        fr.append((day, ev_id, kind_ev, x, g_, -1, float(w['shl'][i]), sg * (mid1(t) - p), t - tb, ctx['ttype'](r_, x, p), first, proof, True))
        first = False
    for arm, rung, F, tt in ours:
        for f in F:
            i, pt, qq, thr, age = f[0], int(f[1]), f[2], f[3], f[4]
            t = float(anc[i])
            if pt != p or not (tb < t <= tend) or not src[i]:
                continue
            ty = 'THROUGH' if thr else ctx['ttype'](lrow[i], x, p)
            fr.append((day, ev_id, kind_ev, x, arm, rung, float(qq), sg * (mid1(t) - p), float(tt[i] - age - tb), ty, False, False, age >= 100.0))
    return ev, fr


def eval_day(day):
    import lightgbm as lgb
    import pf_gap_top_makers as GP
    import pf_depth_timing as DT
    mut = all_selftests()
    T0 = time.time(); D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    LT = D['LT']; ix = LT.index.values
    grp = np.where(tkf[ix] >= 0, tkf[ix], -1 - ix)
    WS = pd.Series(TS.ws_groups(LT.how.values, grp), index=LT.index)
    GRP = pd.Series(grp, index=LT.index)
    TPG = pd.Series(LT.ptick.values).groupby(grp).agg(list).to_dict()
    SIG = pd.Series(full.signer.values[ix].astype('int64'), index=LT.index)
    MP = pd.read_parquet(f'{ORD}/map_{day}.parquet').set_index('row')
    hk = MP.hash_ok.reindex(LT.index).eq(True).values
    OHS = pd.Series(MP.ohash.reindex(LT.index).values, index=LT.index)
    ohmin = pd.Series(LT.anc.values[hk]).groupby(OHS.values[hk]).min().to_dict()
    ohash = {int(r_): o_ for r_, o_, h_ in zip(LT.index.values, OHS.values, hk) if h_}
    mdl = lgb.Booster(model_file=f'{FP.OUTF}/models/av_{day}.txt')
    z = np.load(f'{FP.MOVE_DIR}/day_{day}.npz'); zm = z['B|m'].astype(np.int64); zt = z['B|t']; zs = z['B|side'].astype(int)
    pz = np.load(f'{FP.OUTF}/p_rows_{day}.npy')
    ROWS = {(int(m_), int(s_)): (zt[ix_], pz[ix_]) for (m_, s_), ix_ in pd.DataFrame(dict(m=zm, s=zs)).groupby(['m', 's']).indices.items()}
    PR = pd.read_parquet(f'{OUTD}/pred_{day}.parquet'); PRI = PR.groupby(['m', 'side']).indices
    st.update(window_sides=0, ctrl_differ=0, pd25_differs=0, align_rows=0, align_bad=0, clock_restored=1, fills=0, real_atbest_q={}, events={})
    fills = []; evs = []; frs = []; pend = {}

    def ttype(r_, x, p):
        return DT.trade_type(x, p, TPG[GRP.at[r_]]) if GRP.at[r_] in TPG else 'n/a'
    gfun = lambda r_: (lambda g_: 'top 5' if g_ == 'top 5' else 'field')(G.get(int(SIG.at[r_]), 'rest'))
    for w in K.window_sides(D):
        st['window_sides'] += 1; m, side, sg, n = w['m'], w['side'], w['sg'], w['n']
        tf, mff = w['tf_mff']; tfl = tf.tolist(); P5, opp = w['P5'], w['opp']
        P = av_P(w, mdl) if n else np.zeros(0)
        if (m, side) in ROWS:
            tz_, pz_ = ROWS[(m, side)]
            ii = np.searchsorted(w['tg'], tz_); ok = (ii < n) & (w['tg'][np.clip(ii, 0, max(n - 1, 0))] == tz_)
            st['align_rows'] += len(tz_); st['align_bad'] += int((~ok).sum()) + int((~(np.abs(P[ii[ok]] - pz_[ok]) <= 1e-9)).sum())
        Pm = np.full((n, len(JS)), np.nan)
        if (m, side) in PRI:
            x_ = PR.iloc[PRI[(m, side)]]; Pm[x_.k.values] = x_[[f'P{j}' for j in JS]].values
        vg = (np.asarray(w['vkl'], float) <= K.GUARD_PULL).tolist()
        b0, valid, rs, kind = QR.runs(tf, P5, opp, side)
        if (m, side) in D['LI']:
            l = D['LT'].iloc[D['LI'][(m, side)]]
            o_ = np.lexsort(((-l.ptick.values if side == 0 else l.ptick.values), l.anc.values)); lrow = l.index.values[o_]
        else:
            lrow = np.zeros(0, 'int64')
        src = WS.reindex(lrow).values.astype(bool) if len(lrow) else np.zeros(0, bool)
        terc = lambda t: (lambda k: 'n/a' if k < 0 or not (P[k] == P[k]) else ('low' if P[k] <= TERC[0] else ('mid' if P[k] <= TERC[1] else 'high')))(
            int(np.searchsorted(w['tg'], t - 24.0, side='right')) - 1)
        for i, r_ in enumerate(lrow):                    # real at-best WS shares by group (R493's denominator)
            t = float(w['anc'][i])
            if src[i] and w['ws'] <= t < w['ws'] + 60000.0 and GP.depth_of(tfl, P5, opp, side, t, int(w['ptl'][i])) == 0:
                g_ = gfun(r_); st['real_atbest_q'][g_] = st['real_atbest_q'].get(g_, 0.0) + float(w['shl'][i])
        ours = []
        for q, shf in CFGS:
            tt = w['anc'] - shf; cfg = f'{q}{int(shf)}'
            for d in RUNGS:
                res = {'LDG': K.sim_arm(w, q, shf, 'rest', aux=vg, offset=d, size=SIZE)}
                Fc, Oc, _ = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=pd_dec(vg, w['best'], side, Pm, tau=0.0), idle_only=1, offset=d, size=SIZE)
                st['ctrl_differ'] += int(Fc != res['LDG'][0] or Oc != res['LDG'][1])
                for arm in ARMS[1:]:
                    dec = pd_dec(vg, w['best'], side, Pm, tau=TAUS.get(arm), kfix=FIXK.get(arm))
                    res[arm] = K.sim_arm(w, q, shf, 'rest', aux=vg, decide=dec, idle_only=1, offset=d, size=SIZE)
                st['pd25_differs'] += int(res['PD25'][0] != res['LDG'][0] or res['PD25'][1] != res['LDG'][1])
                for arm, (F, O, pulls) in res.items():
                    st[f'pulls|{arm}|{cfg}'] = st.get(f'pulls|{arm}|{cfg}', 0) + int(pulls)
                    if cfg == 'M50' and arm in ('LDG', 'PD10', 'PD25'):
                        ours.append((arm, d, F, tt))
                    for f in F:
                        i, pt, qq, thr, age, ph = f[0], int(f[1]), f[2], f[3], f[4], f[5]
                        t = float(w['anc'][i])
                        if not (w['ws'] <= t < w['ws'] + 61000.0):
                            continue
                        d_ = GP.depth_of(tfl, P5, opp, side, t, pt)
                        pre = 'n/a'
                        if d_ == 0:
                            ru = QP.run_at(tfl, b0, valid, rs, kind, float(tt[i]), pt)
                            if ru is not None and ru[1] != 'censored':
                                pre = 'PRE' if float(tt[i]) - age < ru[0] else 'POST'
                        ty = 'THROUGH' if thr else (ttype(lrow[i], side, pt) if src[i] else 'n/a')
                        fills.append((int(day), cfg, arm, d, 'WS' if src[i] else 'E2', -9 if d_ is None else d_, bool(thr), age < 100.0, ph == 'pend',
                                      float(qq), sg * (mff[max(int(np.searchsorted(tf, t + 1000.0, side='right')) - 1, 0)] - pt),
                                      sg * (mff[max(int(np.searchsorted(tf, t + 5000.0, side='right')) - 1, 0)] - pt), terc(t), pre, ty))
            if cfg == 'M50':
                Ff, Of, _ = BG.run_lane(w, q, shf, vg, SIZE, 'FAST')
                if (K.L_PLACE, K.L_CANCEL, K.L_PENDCANCEL) != (BG.CLK0['L_PLACE'], BG.CLK0['L_CANCEL'], BG.CLK0['L_PENDCANCEL']):
                    st['clock_restored'] = 0
                ours.append(('FAST', 0, Ff, tt))
        st['fills'] = len(fills)
        # part C: keep this side's context until the window's other side arrives
        ctx_side = dict(w=w, vg=vg, runs=(b0, valid, rs, kind), tfl=tfl, lrow=lrow, src=src, ours=ours)
        pend.setdefault(m, {})[side] = ctx_side
        if len(pend[m]) == 2:
            pair = pend.pop(m)
            ctx = dict(tfl={s_: pair[s_]['tfl'] for s_ in (0, 1)}, lrow={s_: pair[s_]['lrow'] for s_ in (0, 1)}, src={s_: pair[s_]['src'] for s_ in (0, 1)},
                       grp=gfun, ohash=ohash, ohmin=ohmin, ttype=ttype)
            ons = {s_: [float(pair[s_]['w']['tg'][k]) for k in onsets(pair[s_]['vg'])] for s_ in (0, 1)}
            allon = sorted(ons[0] + ons[1]); used = set()

            def starts(s_, kd):
                b0_, valid_, rs_, kind_ = pair[s_]['runs']; tl = pair[s_]['tfl']; out = []
                for j in range(1, len(tl)):
                    if rs_[j] == j and kind_[j] == kd:
                        jn = j + 1
                        while jn < len(tl) and rs_[jn] == j:
                            jn += 1
                        out.append((j, int(b0_[j]), tl[j], tl[jn] if jn < len(tl) else INF))
                return out
            ST = {(s_, kd): starts(s_, kd) for s_ in (0, 1) for kd in (1, 2)}
            wsm = pair[0]['w']['ws']
            for s_ in (0, 1):
                for t0 in ons[s_]:
                    if not (wsm <= t0 < wsm + 60000.0):
                        continue
                    for x, kd, kev in ((s_, 2, 'adverse'), (1 - s_, 1, 'other')):
                        c = [e for e in ST[(x, kd)] if t0 - WIN_PRE < e[2] <= t0 + WIN_POST]
                        if not c or (x, c[0][0]) in used:
                            continue
                        j, p, tb, te_run = c[0]; used.add((x, j)); eid = f'{day}|{m}|{x}|{j}'
                        st['events'][kev] = st['events'].get(kev, 0) + 1
                        ev, fr = level_stats(pair[x]['w'], x, p, tb, te_run, t0, ctx, eid, kev, int(day), pair[x]['ours'])
                        evs.append(ev); frs.extend(fr)
            for x in (0, 1):
                for kd, kev in ((2, 'calm fallback'), (1, 'calm improve')):
                    for j, p, tb, te_run in ST[(x, kd)]:
                        if (x, j) in used or not (wsm <= tb < wsm + 60000.0):
                            continue
                        lo = bisect.bisect_left(allon, tb - WIN_POST)
                        if lo < len(allon) and allon[lo] < tb + WIN_PRE:
                            continue
                        st['events'][kev] = st['events'].get(kev, 0) + 1
                        ev, fr = level_stats(pair[x]['w'], x, p, tb, te_run, tb - PROOF_MS, ctx, f'{day}|{m}|{x}|{j}', kev, int(day), pair[x]['ours'])
                        evs.append(ev); frs.extend(fr)
        w.pop('cache', None)
        if st['window_sides'] % 80 == 0:
            r2 = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r2:.0f} MiB', flush=True)
            if r2 > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r2:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    st['unpaired_windows'] = len(pend)
    if st['ctrl_differ'] or st['align_bad'] or st['align_rows'] != len(zm) or not st['clock_restored']:
        sys.exit(f'PULL-DEPTH CONTROL FAILED on {day} -> no result: {st}')
    st['planted_caught'] = mut
    os.makedirs(OUTD, exist_ok=True)
    pd.DataFrame(fills, columns=['day', 'cfg', 'arm', 'rung', 'src', 'depth', 'thr', 'young', 'pend', 'q', 'v1', 'v5', 'pt', 'pre', 'type']).to_parquet(
        f'{OUTD}/fills_{day}.parquet', index=False)
    pd.DataFrame(evs, columns=['day', 'eid', 'kind', 'side', 'p', 'tb', 'life'] + [f'g{int(g)}' for g in GROW]).to_parquet(f'{OUTD}/events_{day}.parquet', index=False)
    pd.DataFrame(frs, columns=['day', 'eid', 'kind', 'side', 'who', 'rung', 'q', 'v1', 'gap', 'type', 'first', 'proof', 'aged']).to_parquet(
        f'{OUTD}/levelfills_{day}.parquet', index=False)
    json.dump(dict(stat=st, days=len(D['W0']) / 288.0), open(f'{OUTD}/day_{day}.json', 'w'), indent=1, default=float)
    print(f'{day}: {json.dumps({k_: v_ for k_, v_ in st.items() if not isinstance(v_, dict)}, default=float)} events {st["events"]} in {time.time() - T0:.0f} s', flush=True)


# ---------------------------------------------------------------- report
def report():
    FJ = json.load(open(f'{OUTD}/fit.json')); stats = {d: json.load(open(f'{OUTD}/day_{d}.json')) for d in DAYS9}
    A = pd.concat([pd.read_parquet(f'{OUTD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True); A = A[np.isfinite(A.v1)]
    E = pd.concat([pd.read_parquet(f'{OUTD}/events_{d}.parquet') for d in DAYS9], ignore_index=True)
    LF = pd.concat([pd.read_parquet(f'{OUTD}/levelfills_{d}.parquet') for d in DAYS9], ignore_index=True); LF = LF[np.isfinite(LF.v1)]
    RW = pd.concat([pd.read_parquet(f'{OUTD}/rows_{d}.parquet', columns=['day', 'onset', 'mv1000', 'mv250']) for d in DAYS9], ignore_index=True)
    dd = {d: stats[d]['days'] for d in DAYS9}; ndays = sum(dd.values())
    cps = lambda x: float((x.q * x.v1).sum() / x.q.sum()) if x.q.sum() else NAN
    usd = lambda x, d=None: float((x.q * x.v1).sum()) / 100.0 / (dd[d] if d else ndays)
    # control ii: LDG equals TIMING-SPLIT's LDG rows
    T = pd.concat([pd.read_parquet(f'{TSD}/fills_{d}.parquet') for d in DAYS9], ignore_index=True)
    bad = {}
    for cfg in ('M0', 'M50'):
        for d in DAYS9:
            a_ = A[(A.arm == 'LDG') & (A.cfg == cfg) & (A.day == int(d))]; t_ = T[(T.arm == 'LDG') & (T.cfg == cfg) & (T.day == int(d)) & np.isfinite(T.v1)]
            if len(a_) != len(t_) or abs(float((a_.q * a_.v1).sum()) - float((t_.q * t_.v1).sum())) > 1e-6:
                bad[f'{cfg} {d}'] = (len(a_), len(t_))
    if bad or not all(stats[d]['stat']['pd25_differs'] > 0 for d in DAYS9):
        sys.exit(f'PULL-DEPTH REPORT CONTROL FAILED -> no table: LDG vs TIMING-SPLIT {bad}; PD25 differs {[stats[d]["stat"]["pd25_differs"] for d in DAYS9]}')
    R = {}; asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    L = [f'PULL-DEPTH, 9 consumed replay days, first 60 s; as of {asof}; controls passed (tau = 0 = LDG on every window-side, LDG = TIMING-SPLIT,'
         f' P aligned, clocks restored, planted errors caught)', '']
    # part A
    on = RW[RW.onset == 1]
    L.append(f'PART A: guard-on steps {len(RW)} ({len(on)} onsets); move within 1 s at the onset (share): ' +
             ', '.join(f'{j} {float((on.mv1000 == j).mean()):.0%}' for j in range(0, 5)) + f', >= 5 {float((on.mv1000 >= 5).mean()):.0%};'
             f' within 250 ms: 0 {float((on.mv250 == 0).mean()):.0%}, 1 {float((on.mv250 == 1).mean()):.0%}, >= 2 {float((on.mv250 >= 2).mean()):.0%}')
    L.append('  all guard-on steps, P(move >= j within 1 s): ' + ', '.join(f'j{j} {float((RW.mv1000 >= j).mean()):.0%}' for j in JS))
    R['R490'] = {}
    for j in JS:
        per = [FJ['auc'][f'{d}|{j}'] for d in DAYS9]
        pooled = float(np.nanmean(per))
        lab = 'PREDICTABLE' if pooled >= 0.70 and sum(a_ >= 0.65 for a_ in per) >= 7 else 'WEAK'
        R['R490'][j] = dict(auc_mean=pooled, per_day=per, label=lab if j <= 3 else 'printed')
        L.append(f'  R490 j{j}: out-of-day AUC mean {pooled:.3f} (days {min(per):.3f}-{max(per):.3f}) -> {R["R490"][j]["label"]}')
    # part B
    L += ['', 'PART B: $/day w/o young at 1 s (WS-timed | E2 | all), pulls a day; c/sh WS w/o young']
    def cell(arm, cfg, srcs=('WS',), d=None, h='v1'):
        x = A[(A.arm == arm) & (A.cfg == cfg) & ~A.young & A.src.isin(srcs)]
        if d:
            x = x[x.day == int(d)]
        return float((x.q * x[h]).sum()) / 100.0 / (dd[d] if d else ndays)
    for cfg in ('M0', 'M50'):
        for arm in ARMS:
            pl = sum(stats[d]['stat'].get(f'pulls|{arm}|{cfg}', 0) for d in DAYS9) / ndays
            x = A[(A.arm == arm) & (A.cfg == cfg) & ~A.young & (A.src == 'WS')]
            L.append(f'  {cfg} {arm:<5} {cell(arm, cfg):+9.1f} | {cell(arm, cfg, ("E2",)):+9.1f} | {cell(arm, cfg, ("WS", "E2")):+9.1f} | pulls {pl:9.0f} |'
                     f' c/sh {cps(x):+.3f} | WS 5 s {cell(arm, cfg, h="v5"):+9.1f}')
    R['R491'] = {}
    for arm in ('PD10', 'PD25'):
        up = {c: sum(cell(arm, c, d=d) > cell('LDG', c, d=d) for d in DAYS9) for c in ('M0', 'M50')}
        dn = {c: sum(cell(arm, c, d=d) < cell('LDG', c, d=d) for d in DAYS9) for c in ('M0', 'M50')}
        lab = 'HELPS' if min(up.values()) >= 7 else ('HURTS' if min(dn.values()) >= 7 else 'NOT-SHOWN')
        R['R491'][arm] = dict(label=lab, up=up, down=dn, diff={c: cell(arm, c) - cell('LDG', c) for c in ('M0', 'M50')})
        L.append(f'  R491 {arm}: {lab} (WS $/day vs LDG: M 0 {R["R491"][arm]["diff"]["M0"]:+.1f}, higher {up["M0"]}/9; M 50 {R["R491"][arm]["diff"]["M50"]:+.1f},'
                 f' higher {up["M50"]}/9)')
    L.append('  per day WS $/day M 50: LDG | PD10 | PD25 | PDF2 | PDF3')
    for d in DAYS9:
        L.append(f'    {d} ' + ' | '.join(f'{cell(a_, "M50", d=d):+8.1f}' for a_ in ARMS))
    L += ['', 'R492 (M 50, WS, w/o young): at-best fills: PRE share | PRE c/sh | PRE STOP | POST c/sh || deep fills: share at high P | c/sh']
    for arm in ARMS:
        x = A[(A.arm == arm) & (A.cfg == 'M50') & ~A.young & (A.src == 'WS')]; ab = x[x.depth == 0]; pr = ab[ab.pre == 'PRE']; po = ab[ab.pre == 'POST']
        dp = x[x.depth > 0]
        L.append(f'  {arm:<5} {pr.q.sum() / max(ab.q.sum(), 1e-9):5.1%} | {cps(pr):+.3f} | {pr.q[pr.type == "STOP"].sum() / max(pr.q.sum(), 1e-9):4.0%} | {cps(po):+.3f} ||'
                 f' {dp.q[dp.pt == "high"].sum() / max(dp.q[dp.pt != "n/a"].sum(), 1e-9):4.0%} | {cps(dp):+.3f}')
    # part C
    L += ['', 'PART C: levels after a guard onset (adverse = the hit side\'s new best by fallback; other = the other side\'s new best by improvement)'
          ' and calm births; events: ' + json.dumps({k_: int(sum(stats[d]['stat']['events'].get(k_, 0) for d in DAYS9))
                                                      for k_ in ('adverse', 'other', 'calm fallback', 'calm improve')})]
    L.append('  queue at the new level, median shown sh at the birth / +100 / +250 / +500 / +1000 ms (while it stays the best); life median ms')
    for kev in ('adverse', 'calm fallback', 'other', 'calm improve'):
        e = E[E.kind == kev]
        L.append(f'    {kev:<14} n {len(e):>6} | ' + ' / '.join(f'{np.nanmedian(e[f"g{int(g)}"]):.0f}' if e[f'g{int(g)}'].notna().any() else 'n/a' for g in GROW) +
                 f' | life {np.median(e.life[np.isfinite(e.life)]):.0f}')
    real = LF[LF.who.isin(['top 5', 'field'])]
    L.append('  real WS fills at these levels: who | share of shares | first-fill share | first fill ms after birth median | PROOF of first fills | c/sh | STOP'
             ' | c/sh < 100 ms / 100-500 / >= 500 ms')
    tot_ab = {g_: sum(stats[d]['stat']['real_atbest_q'].get(g_, 0.0) for d in DAYS9) for g_ in ('top 5', 'field')}
    top_share_ab = tot_ab['top 5'] / max(sum(tot_ab.values()), 1e-9)
    for kev in ('adverse', 'calm fallback', 'other', 'calm improve'):
        x = real[real.kind == kev]; fx = x[x.first]
        for g_ in ('top 5', 'field'):
            y = x[x.who == g_]; fy = fx[fx.who == g_]
            gb = ' / '.join(f'{cps(y[(y.gap >= lo) & (y.gap < hi)]):+.2f}' for _, lo, hi in GAPB)
            L.append(f'    {kev:<14} {g_:<6} {y.q.sum() / max(x.q.sum(), 1e-9):5.1%} | {len(fy) / max(len(fx), 1):5.1%} | {np.median(fy.gap) if len(fy) else NAN:6.0f} |'
                     f' {fy.proof.mean() if len(fy) else NAN:4.0%} | {cps(y):+.3f} | {y.q[y.type == "STOP"].sum() / max(y.q.sum(), 1e-9):4.0%} | {gb}')
    L.append(f'    (top 5 share of all real at-best WS fill shares, first minute: {top_share_ab:.1%})')
    adv = real[(real.kind == 'adverse') & real.first]
    perd = {}
    for d in DAYS9:
        fa = adv[adv.day == int(d)]; ab_d = stats[d]['stat']['real_atbest_q']; sd = ab_d.get('top 5', 0.0) / max(sum(ab_d.values()), 1e-9)
        perd[d] = (float((fa.who == 'top 5').mean()) if len(fa) else NAN, sd, len(fa))
    ratio = float((adv.who == 'top 5').mean()) / top_share_ab if len(adv) and top_share_ab else NAN
    hi = sum(1 for d in DAYS9 if perd[d][2] >= MIN_FILLS and perd[d][0] > perd[d][1]); lo_ = sum(1 for d in DAYS9 if perd[d][2] >= MIN_FILLS and perd[d][0] < perd[d][1])
    R['R493'] = dict(ratio=ratio, higher_days=hi, lower_days=lo_, label='TOP-FRONT-AFTER-MOVE' if ratio >= 1.2 and hi >= 7 else
                     ('TOP-NOT-FRONT' if ratio <= 1 / 1.2 and lo_ >= 7 else 'PROPORTIONAL'))
    L.append(f'  R493 ADVERSE-FRONT: top 5 first-fill share / at-best share = {ratio:.2f} (higher on {hi}, lower on {lo_} of 9) -> {R["R493"]["label"]}')
    oth = real[real.kind == 'other']; pos = neg = 0
    for d in DAYS9:
        x = oth[oth.day == int(d)]; e_ = x[x.gap < 100.0]; l_ = x[x.gap >= 500.0]
        if len(e_) >= MIN_FILLS and len(l_) >= MIN_FILLS:
            pos += int(cps(e_) > cps(l_)); neg += int(cps(e_) < cps(l_))
    R['R494'] = dict(early=cps(oth[oth.gap < 100.0]), late=cps(oth[oth.gap >= 500.0]), up=pos, down=neg,
                     label='EARLY-PAYS' if pos >= 7 else ('EARLY-DOES-NOT-PAY' if neg >= 7 else 'NOT-SHOWN'))
    L.append(f'  R494 OTHER-SIDE-EARLY: real fills < 100 ms {R["R494"]["early"]:+.3f} vs >= 500 ms {R["R494"]["late"]:+.3f} c/sh (higher {pos}, lower {neg} of 9)'
             f' -> {R["R494"]["label"]}')
    L.append('  ours at these levels (M 50, WS): arm | fills | shares | aged c/sh | STOP | our order live after the birth, ms median | share live < 100 ms')
    for kev in ('adverse', 'calm fallback', 'other', 'calm improve'):
        for arm in (('LDG', 'PD10', 'PD25') if 'fallback' in kev or kev == 'adverse' else ('LDG', 'FAST')):
            y = LF[(LF.kind == kev) & (LF.who == arm) & ((LF.rung == 0) if arm in ('FAST',) or kev in ('other', 'calm improve') else True)]
            ya = y[y.aged]
            L.append(f'    {kev:<14} {arm:<5} {len(y):>6} | {y.q.sum():>8.0f} | {cps(ya):+.3f} | {ya.q[ya.type == "STOP"].sum() / max(ya.q.sum(), 1e-9):4.0%} |'
                     f' {np.median(y.gap) if len(y) else NAN:6.0f} | {(y.q[y.gap < 100.0].sum() / max(y.q.sum(), 1e-9)):4.0%}')
    ob = LF[(LF.kind == 'other') & LF.aged & (LF.rung == 0)]; pos = neg = 0
    for d in DAYS9:
        f_ = ob[(ob.day == int(d)) & (ob.who == 'FAST')]; b_ = ob[(ob.day == int(d)) & (ob.who == 'LDG')]
        if len(f_) >= MIN_FILLS and len(b_) >= MIN_FILLS:
            pos += int(cps(f_) > cps(b_)); neg += int(cps(f_) < cps(b_))
    R['R495'] = dict(fast=cps(ob[ob.who == 'FAST']), base=cps(ob[ob.who == 'LDG']), up=pos, down=neg,
                     label='SPEED-PAYS-HERE' if pos >= 7 else ('SPEED-HURTS-HERE' if neg >= 7 else 'NOT-SHOWN'))
    L.append(f'  R495 OUR-SPEED-THERE: FAST {R["R495"]["fast"]:+.3f} vs BASE {R["R495"]["base"]:+.3f} c/sh (higher {pos}, lower {neg} of 9) -> {R["R495"]["label"]}')
    os.makedirs(OUTD, exist_ok=True)
    json.dump(R, open(f'{OUTD}/pull_depth.json', 'w'), indent=1, default=float)
    open(f'{OUTD}/pull_depth.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        print(f'selftest passed; planted errors caught: {all_selftests()}')
    elif cmd == 'rows' and len(sys.argv) > 2:
        rows_day(sys.argv[2])
    elif cmd == 'fit':
        fit()
    elif cmd == 'eval' and len(sys.argv) > 2:
        eval_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
