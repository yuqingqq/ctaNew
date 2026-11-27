"""Chasing a Binance move on Predict: the top taker's group 3, for us (USER 2026-11-27: "we focus on group 3 first, when we see a move,
combining ob dynamics, we fire, check the performance"). Development evidence on the four consumed days, EVERY market (CHEAP_SIDE_ALL:
'complete' markets are outcome-selected); no orders, no strategy freeze. Declared before the run.
Why: every take test since 2026-11-26 used public_score's decision moments, which drop a move its way in the previous 300 ms; REPRICE took
with the move but only at the aimed quote (no paying through) and was valued at settlement on the complete markets. The top taker's
after-a-move decisions (25%) pay through and are marked +1.24 c/sh at 1 s (STATE_VALUE; reproduced here: a check).
Decision moments (stage points): every Binance futures mid change in direction d (TIMELINE's moves: the opposite best level empties or the
near side improves; E clock) in the first 300 ms of a move; a move = such ticks <= 300 ms apart, its onset = a tick with none in direction
d in the 300 ms before (the group-3 definition). Statuses: the tick and its onset - 300 ms must both have a book (fire_events' 'ok').
We see a tick 1.5 ms after it and decide 2 ms later (TAKE_MODEL's FEED + DECIDE), our order arrives after 22 ms and matches after the
150 ms hold: the match is 175.5 ms after the tick.
Inputs at the tick (causal): public_score3's 62 inputs (Binance best level, its flows against normal, depth, other venues, Predict
fills); the move so far (tick index, time since onset, ticks in the last 50 / 100 ms, ticks against in 300 ms, the quiet before the onset,
the mid's move its way since the onset / over 300 ms / 1 s in bps, aggressive volume its way and against since the onset and over 100 ms);
Predict's book known at the decision (stage labels: mid, spread, best and 5-level sizes both sides, the mid's move its way over 300 ms /
1 s / since the onset's decision, distance from 50, the snapshot's age, spread and updates over 60 s); a fair value from Binance spot
(DRIFT_VALUE's: spot 100 ms late against the registry start, 30 s-return volatility over 15 min, time left): its gap to Predict's mid its
way, delta (c per bp), the move's implied Predict change since the onset (delta x the move in bps) and its gap to Predict's actual change.
The top taker (diagnostic, not an input): its order arrivals (settlement - 150 - c); per move, an arrival in [onset - 300, onset]
(anticipated) or in (onset, last tick + 300 ms] (chased; its lag).
Take (stage labels; TAKE_MARKOUT's): a 50-share IOC on the side its way, limit = the known best + s c (s = 0, 1, 2, 3, 5, 10, 20), every
shown level ('any'), or adaptive (the furthest whole-cent price at which the predicted markout pays the fee and a margin of 0 / 1 c); fill =
the walk of the book at the match (A: the last snapshot at or before it; AB: A's levels at their size in the next snapshot); value per
filled share = the mid h after the match - the price - the fee (2% x min(p, 100 - p) per level), per intended share with misses 0; h = 1 s
(primary), 5 s, 30 s; take back after h (a 50-share IOC decided at match + h through the same 172 ms, every shown level; the rest held);
hold to settlement (ties 50). Rows need the decision >= 1 s into the market's recorded path, a valid known book and a mid 1 s after the
match inside the path; statuses counted; longer horizons count their own.
Actions: at most one per move, at the first tick a selection picks (rows are ticks; values are counted per action).
Selections (stage read), declared: every move at its onset; every move at its second tick; rules at the first qualifying tick: R1 lean
(best-level imbalance its way >= 0.6), R2 sweep (>= 1 BTC aggressive its way in the last 100 ms), R3 fast (>= 3 ticks in the last 50 ms),
R4 stale (implied minus actual Predict change since the onset >= 2 c), R5 = R4 and (R1 or R2); the model: public_score's pinned trees on
every input above predicting the 1 s markout move d x (mid 1 s after the match - the decision mid), leave-one-day-out, edge = prediction -
half the spread - the fee at the known best, the top 10 / 1 / 0.1 % of ticks and edge > 0; the same trees on the 5 s markout (top 1 %, edge
> 0); diagnostics: the moves the top taker chased / anticipated / neither, at their onset; positive control: the oracle (the realised 1 s
markout in place of the prediction, top 1 %), which must pay.
Null (declared, 200 draws): for each model selection, as many random moves (among those with a labelled tick), each at a random labelled
tick of its first 300 ms; mean and 97.5th percentile per fixed order (book A, 1 s).
Paying (computed): value per intended share > 0 pooled AND on >= 3 of 4 days AND >= 150 actions. Point estimates (G = 4 days < 5: no
intervals). Seen days: a positive here needs later untouched days.
Outputs (write-once): OUT/chase_takes/<day>_points.parquet, <day>_labels.parquet + .json; chase_takes.json + OUT/chase_takes/predictions.parquet (read).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_takes.py \
       selftest | points <day> [<dir> <n moves>] | labels <day> [<dir>] | read [<dir>]
"""
import collections
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import before_long as BLm  # noqa: E402
import cross_venue as CV  # noqa: E402
import drift_value as DV  # noqa: E402
import fire_events as FE  # noqa: E402
import public_score as PS  # noqa: E402
import public_score3 as P3  # noqa: E402
import regime_takes as RT  # noqa: E402
import take_markout as TK  # noqa: E402
import take_markout_exit as TE  # noqa: E402
import timeline as TL  # noqa: E402

TM, VR, MM, TR, C, F, WT = TK.TM, TK.VR, TK.MM, TK.TR, TK.C, TK.F, TK.WT
DC = FE.DC
OUT = WT.OUT / 'chase_takes'
DAYS, Q, fee = TK.DAYS, TK.Q, TK.fee
GAP_MS, WIN_MS, TT_AFTER_MS = 300., 300., 300.
NET = 22.
HZ = (1000., 5000., 30000.)
SLIPS = (0., 1., 2., 3., 5., 10., 20.)
MARGINS = (0., 1.)
QTOP = (.1, .01, .001)
N_NULL, MIN_ORDERS, SEED = 200, 150, 20261127
PS_KEEP = ['b_imb1', 'b_lfar1', 'b_lnear1', 'b_spread', 'd_imb20', 'd_lfarcush', 'p_tau', 'p_pf5']
MOVE = ['mv_k', 'mv_since_ms', 'mv_ticks_50', 'mv_ticks_100', 'mv_against_300', 'mv_quiet_before', 'mv_bps', 'mv_bps_300', 'mv_bps_1s',
        'mv_its_btc', 'mv_ag_btc', 'its_100_btc', 'ag_100_btc']
PF = ['pf_mid', 'pf_spread', 'pf_far1', 'pf_near1', 'pf_far5', 'pf_near5', 'pf_mid_ret_300', 'pf_mid_ret_1000', 'pf_mid_ret_onset', 'pf_extreme',
      'pf_its_price', 'pf_age_ms', 'spread_60s', 'book_updates_60s']
FV = ['fv_drift_its', 'fv_vol', 'fv_tau_s', 'fv_z_its', 'fv_gap_its', 'fv_delta', 'chase_implied_c', 'chase_gap_c']
RULES = ('R1_lean', 'R2_sweep', 'R3_fast', 'R4_stale', 'R5_stale_and_lean_or_sweep')
ORDERS_X = ('s0', 's3', 's10', 'any')


# ---------------------------------------------------------------------------------------------------------------- moves
def moves_table(M, gap=GAP_MS):
    """Ticks of one direction (sorted ms) -> per tick: move index, onset, index in the move (1 = onset), the move's last tick, the tick
    before the onset (-inf if none). A tick starts a move when none came in the gap before it."""
    M = np.asarray(M, float)
    prev = np.r_[-np.inf, M[:-1]]
    on = (M - prev) >= gap
    mv = np.cumsum(on) - 1
    first = np.flatnonzero(on)
    last = np.r_[first[1:] - 1, len(M) - 1] if len(first) else np.zeros(0, int)
    return mv, M[first][mv], np.arange(len(M)) - first[mv] + 1, M[last][mv], prev[first][mv]


def first_per_move(sel, key):
    """Rows in time order within each move; a boolean selection -> the first selected row of each move (one action per move)."""
    i = np.flatnonzero(sel)
    _, f = np.unique(key[i], return_index=True)
    return np.sort(i[f])


def move_groups(key, ok):
    """The ok rows grouped by move (a move's rows must be contiguous) -> (rows, first position, count) per move."""
    i = np.flatnonzero(ok)
    ki = key[i]
    u, start, cnt = np.unique(ki, return_index=True, return_counts=True)
    C.require(int(np.r_[True, ki[1:] != ki[:-1]].sum()) == len(u), 'a move\'s rows are not contiguous')
    return i, start, cnt


def random_actions(groups, n, rng):
    """n random moves, each at a random ok row of the move."""
    i, start, cnt = groups
    pick = rng.choice(len(start), n, replace=False)
    return i[start[pick] + np.floor(rng.random(n) * cnt[pick]).astype(int)]


def vol_in(E, cq, lo, hi):
    return cq[np.searchsorted(E, hi, 'right')] - cq[np.searchsorted(E, lo, 'right')]


def mid_series(day):
    """Binance futures mid changes of the day (fire_events' collapse: the last state per ms) -> (E, mid)."""
    mE, mM = [], []
    for h, bk in DC.book_hours(day):
        E, bid, bq, ask, aq = FE.collapse(*bk)
        mid = (bid + ask) / 2.
        k = np.r_[True, np.abs(np.diff(mid)) > 1e-9]
        mE.append(E[k]); mM.append(mid[k])
    E, M = np.concatenate(mE), np.concatenate(mM)
    o = np.argsort(E, kind='stable')
    E, M = E[o], M[o]
    keep = np.r_[np.diff(E) > 0, True]                        # the last state of a ms that two hour files share
    return E[keep], M[keep]


# ---------------------------------------------------------------------------------------------------------------- points
def points(day, od=None, limit=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_points.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    day0 = float(WT.OD_day_ms(day))
    ev, _, _, _ = FE.day_data(day, pd.DataFrame(dict(t=np.zeros(0), d=np.zeros(0))))
    status = collections.Counter()
    parts = []
    for dv in (1., -1.):
        M, Mo = TL.moves(ev, dv), TL.moves(ev, -dv)
        mv, onset, k, end, prev_on = moves_table(M)
        w = (M - onset) <= WIN_MS
        tag = 'up' if dv > 0 else 'down'
        status[f'ticks_{tag}'], status[f'moves_{tag}'], status[f'window_ticks_{tag}'] = int(len(M)), int(mv.max() + 1) if len(M) else 0, int(w.sum())
        Mw = M[w]
        parts.append(pd.DataFrame(dict(t=Mw, d=dv, move=mv[w], onset=onset[w], move_end=end[w], mv_k=k[w].astype(float), mv_since_ms=(Mw - onset[w]),
                                       mv_ticks_50=(np.searchsorted(M, Mw, 'right') - np.searchsorted(M, Mw - 50., 'right')).astype(float),
                                       mv_ticks_100=(np.searchsorted(M, Mw, 'right') - np.searchsorted(M, Mw - 100., 'right')).astype(float),
                                       mv_against_300=(np.searchsorted(Mo, Mw, 'right') - np.searchsorted(Mo, Mw - 300., 'right')).astype(float),
                                       mv_quiet_before=np.log1p(np.minimum(onset[w] - prev_on[w], 60000.)))))
    del ev
    P = pd.concat(parts, ignore_index=True)
    st = FE.day_data(day, pd.DataFrame(dict(t=np.r_[P.t.to_numpy(float), P.onset.to_numpy(float) - GAP_MS], d=np.r_[P.d.to_numpy(float), P.d.to_numpy(float)])))[1]
    ok = (st[:len(P)] == 'ok') & (st[len(P):] == 'ok')
    status['excluded_no_book_or_edge'] = int((~ok).sum())
    P = P[ok].reset_index(drop=True)
    P['mkey'] = P.move.astype(np.int64) * 2 + (P.d > 0).astype(np.int64)
    if limit is not None:                                          # rehearsal: whole moves, seeded
        u = np.unique(P.mkey.to_numpy())
        keep = np.random.default_rng(SEED).choice(u, min(limit, len(u)), replace=False)
        P = P[P.mkey.isin(keep)].reset_index(drop=True)
    P = P.sort_values(['d', 't'], kind='stable').reset_index(drop=True)
    t, d, o, up = P.t.to_numpy(float), P.d.to_numpy(float), P.onset.to_numpy(float), P.d.to_numpy(float) > 0
    # the move so far: the mid and aggressive volume
    E_, M_ = mid_series(day)
    hrs = np.array([(DC.RAW / f'bookTicker/BTCUSDT/{day}_{h:02d}.csv.gz').exists() for h in range(24)])
    m_now = BLm.mid_at(E_, M_, hrs, day0, t)
    for nm, x in (('mv_bps', o - 1.), ('mv_bps_300', t - 300.), ('mv_bps_1s', t - 1000.)):
        P[nm] = d * 1e4 * np.log(m_now / BLm.mid_at(E_, M_, hrs, day0, x))
    del E_, M_
    onset_moved = float(np.mean(P.mv_bps.to_numpy()[P.mv_k.to_numpy() == 1] > 0))
    C.require(onset_moved > .95, f'the mid did not move its way at {100 * (1 - onset_moved):.1f}% of onsets: ticks and mids misaligned')
    tr = DC.read_trades(day).drop_duplicates('id')
    tr = tr[(tr.p > 0) & (tr.q > 0)].sort_values('E', kind='stable')
    buy = tr.m.to_numpy() == DC.BUY_AGGR
    Eb, Es = tr.E.to_numpy(float)[buy], tr.E.to_numpy(float)[~buy]
    cb, cs = np.r_[0., np.cumsum(tr.q.to_numpy(float)[buy])], np.r_[0., np.cumsum(tr.q.to_numpy(float)[~buy])]
    del tr
    vb, vs = vol_in(Eb, cb, o - 1., t), vol_in(Es, cs, o - 1., t)
    P['mv_its_btc'], P['mv_ag_btc'] = np.where(up, vb, vs), np.where(up, vs, vb)
    vb, vs = vol_in(Eb, cb, t - 100., t), vol_in(Es, cs, t - 100., t)
    P['its_100_btc'], P['ag_100_btc'] = np.where(up, vb, vs), np.where(up, vs, vb)
    # the top taker's arrivals (diagnostic)
    c = float(json.loads((H / 'fire_events.json').read_text())['groups']['TOP1']['arrival_offset_ms'])
    A = pd.read_parquet(WT.OUT / 'fire_times_orders.parquet', columns=['day', 'rank', 't_set', 'fire', 'd'])
    A = A[(A.day == day) & (A['rank'] == 1) & A.t_set.notna()]
    tt_next, tt_prev, ant, lag = (np.full(len(P), np.nan) for _ in range(4))
    for dv in (1., -1.):
        m = d == dv
        a = np.sort((A.fire[A.d == dv] - c).to_numpy(float))
        if not m.any() or not len(a):
            continue
        j = np.searchsorted(a, t[m], 'right')
        tt_next[m] = np.where(j < len(a), a[np.minimum(j, len(a) - 1)] - t[m], np.nan)
        tt_prev[m] = np.where(j > 0, t[m] - a[np.maximum(j - 1, 0)], np.nan)
        ant[m] = (np.searchsorted(a, o[m], 'right') - np.searchsorted(a, o[m] - GAP_MS, 'left')) > 0
        jo = np.searchsorted(a, o[m], 'right')
        first = np.where(jo < len(a), a[np.minimum(jo, len(a) - 1)], np.inf)
        lag[m] = np.where(first <= P.move_end.to_numpy(float)[m] + TT_AFTER_MS, first - o[m], np.nan)
    P['tt_next_ms'], P['tt_prev_ms'], P['tt_anticipated'], P['tt_chase_lag_ms'] = tt_next, tt_prev, ant, lag
    # public_score3's inputs at the tick
    Fp = PS.day_features(day, P[['t', 'd']])
    for k_ in PS_KEEP:
        P[k_] = Fp[k_]
    del Fp
    F3 = P3.day_v3(day, P[['t', 'd']])
    for k_ in P3.NEW:
        P[k_] = F3[k_]
    del F3
    C.require(set(P3.V3) <= set(P.columns), 'an input of public_score3 missing')
    for k_ in P3.V3 + MOVE:
        P[k_] = P[k_].astype(np.float32)
    P.insert(0, 'day', day)
    C.require(not P.duplicated(['day', 't', 'd']).any(), 'a tick twice')
    od.mkdir(parents=True, exist_ok=True)
    P.to_parquet(fp, index=False)
    TR.dump(od / f'{day}_points.json', dict(asof='2026-11-27', day=day, rows=int(len(P)), moves=int(P.mkey.nunique()), status=dict(status), onset_moved_its_way=onset_moved,
                                             rehearsal_moves=limit))
    print(day, len(P), P.mkey.nunique(), dict(status), round(onset_moved, 4), 'rss', round(F.rss_mib()), flush=True)


# ---------------------------------------------------------------------------------------------------------------- labels
def labels(day, od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (1500, 1501))
    C.require(day in DAYS, f'{day} is not a declared day')
    od = Path(od) if od else OUT
    fp = od / f'{day}_labels.parquet'
    C.require(not fp.exists(), f'{fp.name} exists (write-once)')
    Z = pd.read_parquet(od / f'{day}_points.parquet', columns=['day', 't', 'd', 'onset'])
    reg = MM.registry(day).set_index('mid')
    rg = pd.read_parquet(MM.S1 / day / 'registry.parquet').set_index('mid')
    ws_to_mid = {int(float(reg.loc[m, 'ws'])): int(m) for m in reg.index}
    Z['mid'] = (np.floor(Z.t / 300000.) * 300).astype(int).map(ws_to_mid)
    status = collections.Counter()
    status['no_market'] = int(Z.mid.isna().sum())
    Z = Z[Z.mid.notna()].reset_index(drop=True)
    Z['mid'] = Z.mid.astype(int)
    by_mid = Z.groupby('mid').indices
    rows, path_st = [], {}
    src = MM.KP.Tape(); off = src.offset(day); n_m = 0
    for h, g in reg.groupby('hour'):
        mids = [int(m) for m in g.index if int(m) in by_mid]
        if not mids:
            continue
        for m, W, lst in VR.worlds_all(day, mids, src, off, reg):
            n_m += 1
            if lst != 'loaded':
                status[f'loader_{lst}'] += len(by_mid[m]); path_st[m] = f'loader_{lst}'
                continue
            ws = float(reg.loc[m, 'ws']) * 1000.
            p, st, info = VR.vm_path(W, ws, 50.)
            path_st[m] = st
            if p is None:
                status[f'path_{st}'] += len(by_mid[m])
                continue
            t0, end = float(info['t_start']), float(info['end_rel'])
            f = np.asarray(p['frames'], float)
            fk = f[:, 1]
            spr = np.where((f[:, 2] >= 0) & (f[:, 12] <= 100) & (f[:, 2] < f[:, 12]), f[:, 12] - f[:, 2], np.nan)
            cs = np.concatenate([[0.], np.cumsum(np.nan_to_num(spr))]); cv = np.concatenate([[0], np.cumsum(np.isfinite(spr))])
            pay = DV.payout(rg.loc[m])
            for i in by_mid[m]:
                t, d, o = float(Z.t.iat[i]), float(Z.d.iat[i]), float(Z.onset.iat[i])
                g_ = t + TM.FEED + TM.DECIDE - t0
                act = g_ + NET + TM.HOLD
                if g_ < 1000. or act + 1000. > end:
                    status['outside_path'] += 1
                    continue
                jk = int(np.searchsorted(fk, g_, 'right') - 1)
                j1 = int(np.searchsorted(fk, g_ - 1000., 'right') - 1)
                if jk < 0 or j1 < 0:
                    status['no_known_frame'] += 1
                    continue
                rk = f[jk]
                bid, ask = rk[2], rk[12]
                if not (0 <= bid < ask <= 100):
                    status['invalid_known_book'] += 1
                    continue
                jv = int(np.searchsorted(f[:, 0], act, 'right') - 1)
                ms = {}
                for hz in HZ:
                    x = act + hz
                    ja = int(np.searchsorted(f[:, 0], x, 'right') - 1)
                    ms[hz] = TM.mid_of(f[ja]) if (x <= end and ja >= 0 and x - f[ja, 0] <= 1000.) else np.nan
                if not np.isfinite(ms[1000.]):
                    status['no_mid_after'] += 1
                    continue
                side = 1 if d > 0 else 0
                md = (bid + ask) / 2.
                j3 = int(np.searchsorted(fk, g_ - 300., 'right') - 1)
                jo = int(np.searchsorted(fk, o + TM.FEED + TM.DECIDE - t0, 'right') - 1)
                m1, m3, mo_ = TM.mid_of(f[j1]), (TM.mid_of(f[j3]) if j3 >= 0 else np.nan), (TM.mid_of(f[jo]) if jo >= 0 else np.nan)
                bs = np.where(rk[2:7] >= 0, rk[7:12], 0.); as_ = np.where(rk[12:17] >= 0, rk[17:22], 0.)
                r = dict(day=day, t=t, d=d, mid=m, lim=ask if d > 0 else bid, md=md, path_status=st, payout=pay, m_act=TM.mid_of(f[jv]),
                         pf_mid=md, pf_spread=ask - bid, pf_far1=as_[0] if d > 0 else bs[0], pf_near1=bs[0] if d > 0 else as_[0],
                         pf_far5=as_.sum() if d > 0 else bs.sum(), pf_near5=bs.sum() if d > 0 else as_.sum(),
                         pf_mid_ret_300=d * (md - m3), pf_mid_ret_1000=d * (md - m1), pf_mid_ret_onset=d * (md - mo_), pf_extreme=abs(md - 50.),
                         pf_its_price=md if d > 0 else 100. - md, pf_age_ms=g_ - f[jk, 0], dtA=act - f[jv, 0])
                for hz in HZ:
                    r[f'm{int(hz / 1000)}s'] = ms[hz]
                    if hz > 1000. and not np.isfinite(ms[hz]):
                        status[f'no_mid_{int(hz / 1000)}s'] += 1
                pa, sa = TK.side_levels(f[jv], side)
                if jv + 1 < len(f):
                    pb, sb = TK.side_levels(f[jv + 1], side)
                    r['dtB'] = f[jv + 1, 0] - act
                else:
                    pb, sb = np.full(5, -1.), np.zeros(5)
                    r['dtB'] = np.nan
                    status['no_frame_after_match'] += 1
                for k in range(5):
                    r[f'pA_{k}'], r[f'sA_{k}'], r[f'pB_{k}'], r[f'sB_{k}'] = pa[k], sa[k], pb[k], sb[k]
                for hz in HZ:
                    hs = f'{int(hz / 1000)}s'
                    xe = act + hz + TE.EXIT_DELAY
                    je = int(np.searchsorted(f[:, 0], xe, 'right') - 1)
                    if xe <= end and je >= 0:
                        pe, se = TK.side_levels(f[je], 0 if d > 0 else 1)          # a long YES exits to the bids, a short to the asks
                    else:
                        pe, se = np.full(5, -1.), np.zeros(5)
                        status[f'no_exit_book_{hs}'] += 1
                    for k in range(5):
                        r[f'eP{hs}_{k}'], r[f'eS{hs}_{k}'] = pe[k], se[k]
                j60 = int(np.searchsorted(fk, g_ - 60000., 'right'))
                span = g_ - max(g_ - 60000., float(fk[0]))
                nv = cv[jk + 1] - cv[j60]
                r['spread_60s'] = (cs[jk + 1] - cs[j60]) / nv if (nv > 0 and span >= 10000.) else np.nan
                r['book_updates_60s'] = (jk + 1 - j60) / (span / 1000.) if span >= 10000. else np.nan
                status['ok'] += 1
                rows.append(r)
            C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    L = pd.DataFrame(rows)
    # the fair value from Binance spot (DRIFT_VALUE's convention: spot known 100 ms late, 30 s-return volatility over 15 min)
    tv, pv, _, _, cov, _ = CV.load('binance_spot', day)
    SV = RT.SpotVol(tv, pv, cov[0], cov[1])
    t, d, md = L.t.to_numpy(float), L.d.to_numpy(float), L.md.to_numpy(float)
    j = np.searchsorted(tv, t - 100., 'right') - 1
    spot = np.where(j >= 0, pv[np.maximum(j, 0)], np.nan)
    drift = 1e4 * np.log(spot / rg.loc[L.mid, 'start_price'].to_numpy(float))
    vol = SV.vol(t)
    tau = (rg.loc[L.mid, 'we'].to_numpy(float) * 1000. - t) / 1000.
    z, fv, delta = DV.fair(drift, vol, tau)
    L['fv_drift_its'], L['fv_vol'], L['fv_tau_s'], L['fv_z_its'] = d * drift, vol, tau, d * z
    L['fv_gap_its'], L['fv_delta'] = d * (fv - md), delta
    status['fv_nan'] = int(np.isnan(L.fv_gap_its.to_numpy()).sum())
    C.require(not L.duplicated(['day', 't', 'd']).any(), 'a tick labelled twice')
    od.mkdir(parents=True, exist_ok=True)
    L.to_parquet(fp, index=False)
    TR.dump(od / f'{day}_labels.json', dict(asof='2026-11-27', day=day, markets=n_m, points=int(len(Z) + status['no_market']), status=dict(status),
                                             path_status=dict(collections.Counter(path_st.values())), dtA_ms_q=[float(x) for x in np.quantile(L.dtA, [.5, .9])]))
    print(day, n_m, dict(status), flush=True)


# ---------------------------------------------------------------------------------------------------------------- read
def frame(src):
    Pp = pd.concat([pd.read_parquet(src / f'{dh}_points.parquet') for dh in DAYS if (src / f'{dh}_labels.parquet').exists()], ignore_index=True)
    Lb = pd.concat([pd.read_parquet(src / f'{dh}_labels.parquet') for dh in DAYS if (src / f'{dh}_labels.parquet').exists()], ignore_index=True)
    C.require(not Pp.duplicated(['day', 't', 'd']).any() and not Lb.duplicated(['day', 't', 'd']).any(), 'a tick twice')
    X = Lb.merge(Pp, on=['day', 't', 'd'], how='inner', validate='1:1')
    C.require(len(X) == len(Lb), 'a labelled tick without its point')
    X['chase_implied_c'] = X.fv_delta * X.mv_bps
    X['chase_gap_c'] = X.chase_implied_c - X.pf_mid_ret_onset
    X['key'] = X.day.astype(str) + ':' + X.mkey.astype(str)
    X = X.sort_values(['day', 'd', 't'], kind='stable').reset_index(drop=True)
    feats = list(P3.V3) + MOVE + PF + FV
    C.require(len(set(feats)) == len(feats) and set(feats) <= set(X.columns), 'inputs missing or twice')
    return X, feats, len(Pp)


def rule_masks(X):
    imb, its100, t50, gap = (X[c].to_numpy(float) for c in ('b_imb1', 'its_100_btc', 'mv_ticks_50', 'chase_gap_c'))
    R4 = gap >= 2.
    return dict(R1_lean=imb >= .6, R2_sweep=its100 >= 1., R3_fast=t50 >= 3., R4_stale=R4, R5_stale_and_lean_or_sweep=R4 & ((imb >= .6) | (its100 >= 1.)))


def read(od=None):
    resource.setrlimit(resource.RLIMIT_CPU, (2400, 2401))
    reh = od is not None
    od = Path(od) if od else None
    fn = (od or H) / 'chase_takes.json'
    fpp = (od or OUT) / 'predictions.parquet'
    C.require(not fn.exists() and not fpp.exists(), 'an output exists (write-once)')
    src = od or OUT
    X, feats, n_points = frame(src)
    d, lim, md, day, mid = X.d.to_numpy(float), X.lim.to_numpy(float), X.md.to_numpy(float), X.day.to_numpy(), X.mid.to_numpy()
    key = X.key.to_numpy()
    cost = d * (lim - md) + fee(lim)
    MK = {h: X[f'm{h}'].to_numpy(float) for h in ('1s', '5s', '30s')}
    MO = {h: np.where(np.isfinite(v), d * (v - md), np.nan) for h, v in MK.items()}
    ok = np.isfinite(MO['1s'])
    C.require(ok.all(), 'a labelled tick without its 1 s mid')
    pay = X.payout.to_numpy(float)
    PA = X[[f'pA_{k}' for k in range(5)]].to_numpy(float); SA = X[[f'sA_{k}' for k in range(5)]].to_numpy(float)
    PB = X[[f'pB_{k}' for k in range(5)]].to_numpy(float); SB = X[[f'sB_{k}' for k in range(5)]].to_numpy(float)
    okB = np.isfinite(X.dtB.to_numpy(float))
    BK = {'A': (PA, SA, np.ones(len(X), bool)), 'AB': TK.minbook(PA, SA, PB, SB) + (okB,)}
    fixed = {f's{int(s)}': lim + d * s for s in SLIPS}
    fixed['any'] = np.where(d > 0, 99., 1.)
    W = {b: {k: TK.vwalk(P_, S_, d, L_) for k, L_ in fixed.items()} for b, (P_, S_, _) in BK.items()}
    EX = {h: (X[[f'eP{h}_{k}' for k in range(5)]].to_numpy(float), X[[f'eS{h}_{k}' for k in range(5)]].to_numpy(float)) for h in ('1s', '5s', '30s')}
    labst = {dh: json.loads((src / f'{dh}_labels.json').read_text()) for dh in DAYS if (src / f'{dh}_labels.json').exists()}
    ptst = {dh: json.loads((src / f'{dh}_points.json').read_text()) for dh in DAYS if (src / f'{dh}_points.json').exists()}
    out = dict(asof='2026-11-27', rehearsal=reh, points=n_points, labelled_ticks=int(len(X)), moves_labelled=int(np.unique(key).size), inputs=len(feats),
               point_status={k: v['status'] for k, v in ptst.items()}, label_status={k: v['status'] for k, v in labst.items()},
               path_status={k: v['path_status'] for k, v in labst.items()}, models={}, selections={}, null={}, by_tick={}, top_taker={})
    P = X[['day', 't', 'd', 'key']].copy()
    preds = {}
    for h in ('1s', '5s'):
        okh = np.isfinite(MO[h])
        pr = TK.fit_lodo(X, feats, MO[h], okh)
        preds[h] = pr
        P[f'pred_{h}'] = pr
        out['models'][h] = dict(n=int(okh.sum()), spearman_lodo=float(spearmanr(pr[okh], MO[h][okh]).correlation), move_all_c=float(MO[h][okh].mean()),
                                edge_at_known_best_all_c=float((MO[h] - cost)[okh].mean()))
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    e1, e5 = preds['1s'] - cost, preds['5s'] - cost
    k_ = X.mv_k.to_numpy(float)
    R = rule_masks(X)
    tt_chased, tt_ant = np.isfinite(X.tt_chase_lag_ms.to_numpy(float)), X.tt_anticipated.to_numpy(float) > 0
    our_arrival = TM.FEED + TM.DECIDE + NET                         # ms after the tick
    ttn, ttp = X.tt_next_ms.to_numpy(float), X.tt_prev_ms.to_numpy(float)
    tt_ahead = (ttp <= GAP_MS) | (ttn <= our_arrival)               # its order (same direction) lands before ours: it matches first
    tt_behind = ~tt_ahead & (ttn <= GAP_MS)

    def topq(score, okh, q):
        thr = np.quantile(score[okh], 1. - q)
        return okh & (score >= thr)

    okv = ok & np.isfinite(e1)
    sels = dict(every_move_onset=('none', first_per_move(k_ == 1, key)), every_move_tick2=('none', first_per_move(k_ == 2, key)))
    for rn, m in R.items():
        sels[rn] = ('none', first_per_move(m & ok, key))
    for q in QTOP:
        sels[f'model1s_top{q * 100:g}'] = ('1s', first_per_move(topq(e1, okv, q), key))
    sels['model1s_edge_pos'] = ('1s', first_per_move(okv & (e1 > 0), key))
    ok5 = np.isfinite(e5) & ok                                     # chosen on what is known at the decision; the 5 s mark is the outcome
    sels['model5s_top1'] = ('5s', first_per_move(topq(e5, ok5, .01), key))
    sels['model5s_edge_pos'] = ('5s', first_per_move(ok5 & (e5 > 0), key))
    sels['top_taker_chased_onset'] = ('none', first_per_move((k_ == 1) & tt_chased, key))
    sels['top_taker_anticipated_onset'] = ('none', first_per_move((k_ == 1) & tt_ant & ~tt_chased, key))
    sels['top_taker_neither_onset'] = ('none', first_per_move((k_ == 1) & ~tt_ant & ~tt_chased, key))
    sels['oracle1s_top1'] = ('oracle', first_per_move(topq(MO['1s'] - cost, ok, .01), key))
    predh = {'1s': preds['1s'], '5s': preds['5s'], 'oracle': MO['1s']}

    def secondary(idx, b):
        o_ = {}
        if b == 'AB':
            idx = idx[okB[idx]]
        for k in ORDERS_X:
            fl, co, fe_ = W[b][k]
            for h in ('5s', '30s'):
                v = d * (np.nan_to_num(MK[h]) * fl - co) - fe_
                okh = idx[np.isfinite(MK[h][idx])]
                o_[f'{k}_mid_{h}'] = dict(TE.summary(v, okh, day), excluded_no_mid=int(len(idx) - len(okh)))
            for h in ('1s', '5s', '30s'):
                back, hold, xq = TE.exit_values(fl, co, fe_, d, EX[h][0], EX[h][1], pay)
                o_[f'{k}_take_back_{h}'] = dict(TE.summary(back, idx, day), exit_filled_share=float(xq[idx].sum() / max(fl[idx].sum(), 1e-9)))
            o_[f'{k}_hold_to_settlement'] = TE.summary(hold, idx, day)
        return o_

    for nm, (ph_key, idx) in sels.items():
        s = dict(actions=int(len(idx)), markets=int(np.unique(mid[idx]).size), tick_index_median=float(np.median(k_[idx])) if len(idx) else None,
                 ms_since_onset_median=float(np.median(X.mv_since_ms.to_numpy(float)[idx])) if len(idx) else None,
                 move_1s_c=float(MO['1s'][idx].mean()) if len(idx) else None, edge_at_known_best_1s_c=float((MO['1s'] - cost)[idx].mean()) if len(idx) else None, orders={})
        if len(idx) == 0:
            out['selections'][nm] = s
            continue
        for b, (P_, S_, okb) in BK.items():
            sb = idx[okb[idx]]
            s['orders'][b] = dict(excluded_no_snapshot_after=int(len(idx) - len(sb)))
            for k in fixed:
                s['orders'][b][k] = TK.cell(sb, np.ones(len(sb), bool), W[b][k], d, MK['1s'], lim, day, mid)
            if ph_key != 'none':
                for mg in MARGINS:
                    al = TK.adaptive_limit(md, d, predh[ph_key], mg)
                    wa = TK.vwalk(P_[sb], S_[sb], d[sb], al[sb])
                    wfull = tuple(np.zeros(len(X)) for _ in range(3))
                    for a_, z_ in zip(wfull, wa):
                        a_[sb] = z_
                    fire = np.where(d[sb] > 0, al[sb] >= lim[sb], al[sb] <= lim[sb])
                    cc = TK.cell(sb, fire, wfull, d, MK['1s'], lim, day, mid)
                    cc['limit_beyond_known_best_c_median'] = float(np.median((d * (al - lim))[sb][fire])) if fire.any() else None
                    s['orders'][b][f'adaptive_m{int(mg)}'] = cc
        s['secondary_A'] = secondary(idx, 'A')
        s['secondary_AB'] = secondary(idx, 'AB')
        fa, fb = W['A']['any'][0][idx], W['AB']['any'][0][idx]
        race = dict(top_taker_ahead_share=float(tt_ahead[idx].mean()), top_taker_behind_share=float(tt_behind[idx].mean()))
        for b in ('A', 'AB'):
            fl, co, fe_ = W[b]['any']
            v = (d * (MK['1s'] * fl - co) - fe_) / Q
            for gnm, gm in (('ahead', tt_ahead[idx]), ('not_ahead', ~tt_ahead[idx])):
                gi = idx[gm & (okB[idx] if b == 'AB' else True)]
                race[f'{b}_any_1s_when_{gnm}_c'] = float(v[gi].mean()) if len(gi) else None
                race[f'{b}_any_fill_when_{gnm}'] = float(fl[gi].sum() / (Q * len(gi))) if len(gi) else None
        s['race'] = race
        s['contested_share_any'] = float(1. - fb.sum() / max(fa.sum(), 1e-9))          # shares A fills that the next snapshot no longer shows
        there = np.where(d > 0, (PA[:, 0] >= 0) & (PA[:, 0] <= lim), (PA[:, 0] >= 0) & (PA[:, 0] >= lim))[idx]
        s['known_best_still_shown_at_match'] = float(there.mean())
        s['predict_moved_its_way_since_onset'] = float((X.pf_mid_ret_onset.to_numpy(float)[idx] > 0).mean())
        s['predict_mid_move_during_hold_c'] = float(np.nanmean(d[idx] * (X.m_act.to_numpy(float)[idx] - md[idx])))
        out['selections'][nm] = s
        C.require(F.rss_mib() < F.RSS_GUARD_MIB, 'RSS over the self-guard')
    # null: random actions of the same count
    rng = np.random.default_rng(SEED)
    groups = move_groups(key, ok)
    VA = {k: (d * (MK['1s'] * W['A'][k][0] - W['A'][k][1]) - W['A'][k][2]) / Q for k in fixed}
    for nm in [x for x in sels if x.startswith('model')]:
        n = len(sels[nm][1])
        if n == 0:
            continue
        o_ = {}
        draws = {k: [] for k in fixed}
        for _ in range(N_NULL if not reh else 20):
            ia = random_actions(groups, n, rng)
            for k in fixed:
                draws[k].append(float(VA[k][ia].mean()))
        for k, v in draws.items():
            v = np.asarray(v)
            o_[k] = dict(mean_c=float(v.mean()), p97_5_c=float(np.quantile(v, .975)), model_c=out['selections'][nm]['orders']['A'][k].get('value_per_intended_c'))
        out['null'][nm] = dict(draws=int(len(v)), actions=int(n), orders=o_)
    # descriptive: every labelled tick by its place in the move (no de-duplication)
    for nm, lo, hi in (('tick1', 1, 1), ('tick2', 2, 2), ('tick3', 3, 3), ('tick4_6', 4, 6), ('tick7+', 7, 1e9)):
        m = (k_ >= lo) & (k_ <= hi)
        out['by_tick'][nm] = dict(ticks=int(m.sum()), move_1s_c=float(MO['1s'][m].mean()) if m.any() else None,
                                  orders={k: TK.cell(np.flatnonzero(m), np.ones(int(m.sum()), bool), W['A'][k], d, MK['1s'], lim, day, mid) for k in ('s0', 's3', 'any')})
    # the top taker's own after-a-move decisions (STATE_VALUE's), reproduced
    SVp = pd.read_parquet(WT.OUT / 'state_value_points.parquet', columns=['day', 'kind', 't', 'd', 'v1', 'momentum'])
    g3 = SVp[(SVp.kind == 'send') & SVp.momentum]
    ref = json.loads((H / 'state_value.json').read_text())['decisions']['by_class']['after_a_move']
    io = sels['every_move_onset'][1]
    out['top_taker'] = dict(after_a_move_decisions=int(len(g3)), after_a_move_v1_c=float(g3.v1.mean()), state_value_ref_v1_c=float(ref['v1']),
                            moves_chased_share=float(tt_chased[io].mean()) if len(io) else None, moves_anticipated_share=float(tt_ant[io].mean()) if len(io) else None,
                            chase_lag_ms_q=[float(x) for x in np.nanquantile(X.tt_chase_lag_ms.to_numpy(float)[io], [.25, .5, .75])] if tt_chased[io].any() else None)
    if not reh:
        C.require(abs(out['top_taker']['after_a_move_v1_c'] - ref['v1']) < 1e-9, 'STATE_VALUE\'s after-a-move value not reproduced')
    # verdict (computed)
    cells = [(nm, b, k, c) for nm, s in out['selections'].items() for b, ex in s['orders'].items() for k, c in ex.items() if isinstance(c, dict) and c.get('orders')]
    held = [x for x in cells if not x[0].startswith(('oracle', 'top_taker'))]
    big = [x for x in held if x[3]['orders'] >= MIN_ORDERS]
    best = max(big, key=lambda x: x[3]['value_per_intended_c']) if big else None
    orc = [x for x in cells if x[0].startswith('oracle')]
    sec = [(nm, b, k, c['value_per_intended_c'], c['paying']) for nm, s in out['selections'].items() if not nm.startswith(('oracle', 'top_taker'))
           for b in ('A', 'AB') if f'secondary_{b}' in s for k, c in s[f'secondary_{b}'].items() if c.get('value_per_intended_c') is not None]
    beats = {nm: [k for k, v in o_['orders'].items() if v['model_c'] is not None and v['model_c'] > v['p97_5_c']] for nm, o_ in out['null'].items()}
    out['verdict'] = dict(
        cells=len(cells), heldout_cells=len(held), heldout_cells_with_min_actions=len(big),
        paying=[dict(selection=a, book=b, order=k, value_c=c['value_per_intended_c'], actions=c['orders'], fill=c['fill'], days_positive=c['days_positive']) for a, b, k, c in held if c['paying']],
        any_paying=bool(any(c['paying'] for *_, c in held)),
        best=dict(selection=best[0], book=best[1], order=best[2], value_c=best[3]['value_per_intended_c'], actions=best[3]['orders'], fill=best[3]['fill'],
                  days_positive=best[3]['days_positive']) if best else None,
        secondary_paying=[x for x in sec if x[4]], secondary_best=max(sec, key=lambda x: x[3]) if sec else None,
        paying_AB=[dict(selection=a, order=k, value_c=c['value_per_intended_c'], actions=c['orders'], fill=c['fill'], days_positive=c['days_positive']) for a, b, k, c in held
                   if c['paying'] and b == 'AB'],
        model_beats_null_orders=beats,
        positive_control_oracle_pays=bool(any(c['paying'] for *_, c in orc)),
        oracle_best=max(((b, k, c['value_per_intended_c']) for _, b, k, c in orc), key=lambda x: x[2]) if orc else None)
    if not reh:
        C.require(out['verdict']['positive_control_oracle_pays'], 'the positive control (oracle) does not pay: the instrument cannot show a paying chase')
    (od or OUT).mkdir(parents=True, exist_ok=True)
    P.to_parquet(fpp, index=False)
    TR.dump(fn, out)
    print(json.dumps(dict(models=out['models'], verdict=out['verdict'], top_taker=out['top_taker']), indent=1, default=str)[:6000])
    for nm, s in out['selections'].items():
        if s.get('orders'):
            print(nm, s['actions'], {k: round(c.get('value_per_intended_c', np.nan), 2) for k, c in s['orders']['A'].items() if isinstance(c, dict)})


# ---------------------------------------------------------------------------------------------------------------- selftest
def selftest():
    M = np.array([0., 100., 250., 600., 1000., 1200., 1499., 1800.])
    mv, on, k, end, prev = moves_table(M)
    assert mv.tolist() == [0, 0, 0, 1, 2, 2, 2, 3], mv
    assert on.tolist() == [0., 0., 0., 600., 1000., 1000., 1000., 1800.] and k.tolist() == [1, 2, 3, 1, 1, 2, 3, 1], (on, k)
    assert end.tolist() == [250., 250., 250., 600., 1499., 1499., 1499., 1800.], end
    assert np.isneginf(prev[0]) and prev[3] == 250. and prev[4] == 600. and prev[7] == 1499., prev
    mv2, on2, k2, _, _ = moves_table(np.array([0., 300., 599.]))                  # exactly 300 ms apart: a new move
    assert mv2.tolist() == [0, 1, 1] and k2.tolist() == [1, 1, 2]
    key = np.array(['a', 'a', 'a', 'b', 'b', 'c'])
    sel = np.array([False, True, True, True, False, False])
    assert first_per_move(sel, key).tolist() == [1, 3]
    assert first_per_move(np.zeros(6, bool), key).tolist() == []
    rng = np.random.default_rng(0)
    okr = np.array([True, True, False, True, True, True])
    gr = move_groups(key, okr)
    seen = set()
    for _ in range(200):
        ia = random_actions(gr, 3, rng)
        assert len(set(key[ia])) == 3 and okr[ia].all(), ia
        seen |= set(ia.tolist())
    assert seen == {0, 1, 3, 4, 5}, seen                                         # every ok row reachable, the excluded one never
    try:
        random_actions(gr, 4, rng)
        raise AssertionError('more actions than moves accepted')
    except ValueError:
        pass
    try:
        move_groups(np.array(['a', 'b', 'a']), np.ones(3, bool))
        raise AssertionError('a non-contiguous move accepted')
    except ValueError as e:
        assert 'not contiguous' in str(e), e
    E = np.array([1., 2., 5.]); cq = np.r_[0., np.cumsum([1., 2., 4.])]
    assert vol_in(E, cq, 1., 5.) == 6. and vol_in(E, cq, 0., 1.) == 1. and vol_in(E, cq, 5., 9.) == 0.
    # rules on a toy frame
    Xr = pd.DataFrame(dict(b_imb1=[.7, .1, .1, .1], its_100_btc=[0., 2., 0., 0.], mv_ticks_50=[1., 1., 3., 1.], chase_gap_c=[0., 2.5, 0., 3.]))
    Rm = rule_masks(Xr)
    assert Rm['R1_lean'].tolist() == [True, False, False, False] and Rm['R2_sweep'].tolist() == [False, True, False, False]
    assert Rm['R3_fast'].tolist() == [False, False, True, False] and Rm['R4_stale'].tolist() == [False, True, False, True]
    assert Rm['R5_stale_and_lean_or_sweep'].tolist() == [False, True, False, False]
    # falsifiers: an undeclared day is refused before any file opens; a tick twice is refused
    for fn_ in (lambda: points('20261022', '/nonexistent'), lambda: labels('19700101', '/nonexistent')):
        try:
            fn_()
            raise AssertionError('an undeclared day accepted')
        except ValueError as e:
            assert 'not a declared day' in str(e), e
    print('chase_takes selftest PASS')


if __name__ == '__main__':
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == 'points':
        points(a[0], a[1] if len(a) > 1 else None, int(a[2]) if len(a) > 2 else None)
    elif cmd == 'labels':
        labels(a[0], a[1] if len(a) > 1 else None)
    elif cmd == 'read':
        read(a[0] if a else None)
    else:
        dict(selftest=selftest)[cmd]()
