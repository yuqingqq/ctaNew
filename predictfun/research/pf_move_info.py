"""MOVE, what explains the move after a fill and what information is missing (2026-11-12, session ce68ebe0). USER: "we need to get
front position in the queue and the prediction model does not perform well enough to keep/pull, need to find ways to improve on these
2 key issues" / "We can test, but finding missing information or explain the move will be more important".

Public tape only: real maker legs and real book frames, no replay fill model (B1a). The 9 consumed replay days (10-23..27, 10-31,
11-01..03), first 60 s of each window. Other venues' public trades (data/exchange_public/README.txt) exist for 8 of the 9 days (not
11-03); no file dated 11-04 or later is opened. Known context: the top makers' edge builds in the first 250 ms after their fills and
is not Binance (22%) nor our guard (B12c); the guard tells activity better than harm (FA); an 11-04 check found other venues' big
trades add ~0.5% (pre-11-05, our replay's fills).

Part A, EXPLAIN (descriptive, conditional on trades). At-best maker legs (the maker side's best in the frame strictly before the
anchor, GAP's depth 0), WS-timed anchors only (E2 anchors are uncertain by about +-250 ms, FINDINGS P12 / C12: a status), one event
per taker order and price. Outcome: the move after the fill, mv_H = sg x (mid(anchor + H) - mid just before), H = 250 ms and 1 s,
mids from frames by venue stamp. Populations: the level still standing in the first frame after the trade (B12c's; primary) and all.
What happens in (anchor, anchor + 250] (and the 250 ms before), four families:
  BN    Binance USD-M BTCUSDT mid return (the guard grid's bn_mid, known at E + 3.5 ms)
  VEN   Coinbase, OKX swap, Bybit perp last-trade returns and signed volume (exchange stamps); 8 days
  TAPE  later Predict takes: same side at / through the price by other taker orders (shares, orders), the same taker account again,
        opposite-side takes
  BOOK  the level's size change after the trade split into trades and cancels, the level gone, the opposite best's move
 Out-of-day R^2 (LightGBM, the guard's pinned parameters pf_model_spec, leave one day out) of mv_250 on each family alone and all
 four (VEN and ALL on the 8 days); the top-5 vs field gap in mv_250 (share-weighted, makers ranked per day as GAP) before and after
 subtracting each family's out-of-day fit (B12c's X).
Part B, MISSING INFORMATION (decision-time rows, not outcome-selected: CLAUDE.md rule 1). Every 500 ms of the first minute, per side,
where the known best exists and the guard value is known (v = -1 rows are a status, P16). Targets in (t + 24, t + 24 + H] (decision
2 ms + cancel 22 ms), H = 250 ms (primary) and 1 s:
  DN_H   the side's best moves worse than the known best at t (frames by venue stamp): for an order at the front this is the harm,
         whether the level is traded through or pulled away
  CLR_H  DN_H with a trade at or through that price (WS anchor in the window, E2 within +-250 ms of it)
  ACT_H  a WS-timed trade at that price on that side (activity)
 Features known at t:
  BASE  the guard's 32 inputs, its value v, S4's 4 extras, side
  TAPE  Predict trades known by t (known = max(anchor + 13.2, first receipt - shift) + 2, the frames' rule; no receipt: anchor + 60 /
        + 300 for E2): own-side and opposite take volume 250 ms / 1 s, taker orders in 1 s, time since the last take, multi-price takes
        in 1 s, and the latest own-side taker account's earlier trades today (count, and the makers' completed 1 s markout on them:
        a causal taker reputation)
  BOOK  known frames: the best's size change over 100 / 250 / 500 ms at an unchanged price, its cancels (decrease beyond trades) and
        adds over 250 ms, time since the best price last changed, the second level's size and its change, the opposite best's move
        over 250 ms, best-price changes in 1 s
  VEN   as Part A, as of t minus an assumed one-way delay to Tokyo (Coinbase 80, OKX 40, Bybit 40 ms), and each minus Binance (250 ms)
 LightGBM binary (pinned parameters), leave one day out; VEN and ALL compared with BASE refitted on the same 8 days. AUC out of day.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; day rule = >= 7 of the days in the comparison):
 R370 EXPLAIN (Part A, standing, mv_250): the family with the largest out-of-day R^2 alone is the DRIVER (label = its name) if that R^2
      >= 0.02 and >= 1.5 x the second largest; else MIXED. All values and mv_1000 / the all-legs population printed.
 R371 GAP (Part A, standing, mv_250): per family and for all four, the share of the top-5 vs field gap removed by its out-of-day fit:
      >= 50% -> EXPLAINS-GAP, 20-50% -> PART, < 20% -> NO (computed on the 8 days for VEN / ALL).
 R372 MISSING-INFO (Part B, DN_250): per family F, AUC(BASE + F) - AUC(BASE), out of day: pooled >= +0.010 and above on >= 7 days ->
      NEW-INFO; pooled >= +0.003 and above on >= 7 days -> SMALL; else NONE.
 R373 HARM-VS-ACTIVITY (Part B): BASE's out-of-day AUC on DN_250 and on ACT_250: TRACKS-ACTIVITY if ACT's is higher on >= 7 of 9 days,
      TRACKS-HARM if DN's is, else MIXED (FA's post hoc reading re-tested on market targets). CLR_250 / DN_1000 printed for BASE and
      BASE + TAPE + BOOK.
Controls (any failure -> no table): K1 synthetic DN cases with known answers, a planted window start at t (not t + 24) must break
them; K2 no TAPE feature uses a leg known after t (asserted on every row), a planted anchor-for-known-time must break the synthetic
case; K3 Part B falsifiers: a planted feature (DN_250 with 20% of rows flipped) must read NEW-INFO and TAPE permuted within each day
must read NONE; K4 Part A falsifiers: a planted feature (mv_250 plus equal-variance noise) must reach R^2 >= 0.3 and permuted TAPE
must stay <= 0.005; K5 AUC inputs carry no NaN score (P17: counted and refused).
Limits: consumed days; descriptive; the venue delays are assumptions; exchange clocks vs Predict's venue clock are not synchronised
(under ~10 ms not resolved); the relay (Predict's reference price) is not used: 1 print per s, ~1.3 s after publication.
Usage: research/run_move.sh; python3 predictfun/research/pf_move_info.py selftest | day <yyyymmdd> | report
"""
import bisect
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402  (the day loader, window contexts, guard features)
import pf_gap_top_makers as GP                            # noqa: E402  (at-best classifier, per-day maker ranks)
import pf_model_spec as MS                                # noqa: E402  (pinned LightGBM parameters)

NAN = float('nan'); INF = float('inf')
DAYS9 = K.DAYS9; VDAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
OUTM = f'{K.IO}/move_20261112'; EX = '/home/yuqing/ctaNew/data/exchange_public'
LAT = {'cb': 80.0, 'okx': 40.0, 'byb': 40.0}; VEN_NAMES = {'cb': 'Coinbase', 'okx': 'OKX swap', 'byb': 'Bybit perp'}
L_ACT = 24.0; HS = (250.0, 1000.0); STEP = 10; E2_TOL = 250.0
MEM_CAP_MIB = 2048.0; FWD_FILE_MS = K._ms('2026-11-04')
MUT_DN_START = MUT_TAPE_ANC = 0                           # planted errors (self-test only)
BASE_N = list(K.G32) + ['v', 'own_d250', 'opp_d250', 'act_rel', 'd_bnimb250', 'side']
TAPE_N = ['tv250_own', 'tv1000_own', 'tn1000_own', 'tsl_own', 'tmp1000_own', 'tv250_opp', 'tv1000_opp', 'tsl_opp', 'rep_n', 'rep_m']
BOOK_N = ['dsz100', 'dsz250', 'dsz500', 'cxl250', 'add250', 'lvl_age', 's2', 'ds2_250', 'opp_mv250', 'pchg1000']
VEN_N = [f'{v}_{k}' for v in LAT for k in ('r100', 'r250', 'r1000', 'sv250', 'sv1000', 'div250')]
A_BN = ['bn_c', 'bn_pre']; A_VEN = [f'{v}_{k}' for v in LAT for k in ('c', 'pre', 'svc')]
A_TAPE = ['tq_own', 'tn_own', 'same_acct', 'tq_opp']; A_BOOK = ['a_dsz', 'a_cxl', 'a_add', 'a_gone', 'a_oppmv']


# ---------------- small instruments ----------------
def dn_target(tf, b, side, t, p, H):
    """1 if a frame stamped in (t + 24, t + 24 + H] shows the side's best worse than p (no best counts as worse)."""
    s0 = t if MUT_DN_START else t + L_ACT
    j0 = bisect.bisect_right(tf, s0); j1 = bisect.bisect_right(tf, t + L_ACT + H)
    if j1 <= j0:
        return 0
    x = b[j0:j1]
    return int(np.any(x < p) if side == 0 else np.any((x > p) | (x < 0)))


def window_sum(kt, cs, t, h):
    """sum of the values whose known time is in (t - h, t]; kt sorted, cs = cumulative sums with a leading 0."""
    return cs[bisect.bisect_right(kt, t)] - cs[bisect.bisect_right(kt, t - h)]


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if np.isnan(s).any():
        raise ValueError(f'NaN scores in an AUC input ({int(np.isnan(s).sum())}): P17')
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return NAN
    r = pd.Series(s).rank(method='average').values
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def selftest():
    bad = []
    tf = [0.0, 10.0, 30.0, 60.0, 100.0]; b = np.array([50, 50, 49, 50, 50])
    if dn_target(tf, b, 0, 0.0, 50, 50.0) != 1 or dn_target(tf, b, 0, 0.0, 50, 5.0) != 0:
        bad.append('dn basic')
    tf2 = [0.0, 10.0, 20.0, 60.0]; b2 = np.array([50, 49, 50, 50])
    if dn_target(tf2, b2, 0, 0.0, 50, 250.0) != 0:
        bad.append('dn starts at t + 24')
    if dn_target([0.0, 40.0], np.array([60, 61]), 1, 0.0, 60, 250.0) != 1 or dn_target([0.0, 40.0], np.array([60, -1]), 1, 0.0, 60, 250.0) != 1:
        bad.append('dn ask side / no best')
    anc = np.array([90.0, 195.0]); kt = np.array([100.0, 210.0]); q = np.array([3.0, 4.0])
    use = anc if MUT_TAPE_ANC else kt
    if window_sum(use, np.r_[0.0, np.cumsum(q)], 200.0, 250.0) != 3.0:
        bad.append('tape counts a leg known after t')
    try:
        auc([1, 0], [NAN, 1.0]); bad.append('auc accepted NaN')
    except ValueError:
        pass
    if abs(auc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) - 1.0) > 1e-12 or abs(auc([1, 0], [0.5, 0.5]) - 0.5) > 1e-12:
        bad.append('auc values')
    return bad


def mutation_check():
    global MUT_DN_START, MUT_TAPE_ANC
    out = {}
    for name in ('MUT_DN_START', 'MUT_TAPE_ANC'):
        globals()[name] = 1
        out[name] = len(selftest()) > 0
        globals()[name] = 0
    return out


# ---------------- other venues ----------------
def _iso(d):
    return f'{d[:4]}-{d[4:6]}-{d[6:]}'


def _guard_file_date(d):
    if K._ms(_iso(d)) >= FWD_FILE_MS:
        sys.exit(f'REFUSED: would open a venue file dated {d} (>= 11-04)')


def load_venues(day):
    if day not in VDAYS:
        return {}
    d0 = K._ms(_iso(day)); d1 = d0 + 86400000.0; V = {}
    _guard_file_date(day)
    x = pd.read_csv(f'{EX}/coinbase/{day}.csv.gz').sort_values(['t_ms', 'id'], kind='stable')
    V['cb'] = (x.t_ms.values.astype(float), x.price.values.astype(float), x.size_btc.values * x.taker.values.astype(float))
    parts = []
    for f in (day, (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime('%Y%m%d')):
        _guard_file_date(f)
        p = f'{EX}/okx_swap/BTC-USDT-SWAP-trades-{_iso(f)}.zip'
        if os.path.exists(p):
            with zipfile.ZipFile(p) as z:
                with z.open(z.namelist()[0]) as fh:
                    y = pd.read_csv(fh, usecols=['side', 'price', 'size', 'created_time'], encoding_errors='replace')
            y = y[(y.created_time >= d0) & (y.created_time < d1)]
            parts.append(pd.DataFrame(dict(t=y.created_time.values.astype(float), p=y.price.values.astype(float),
                                           q=y['size'].values * 0.01 * np.where(y.side.values == 'buy', 1.0, -1.0))))
    y = pd.concat(parts).sort_values('t', kind='stable'); V['okx'] = (y.t.values, y.p.values, y.q.values)
    z = pd.read_csv(f'{EX}/bybit_perp/BTCUSDT{_iso(day)}.csv.gz', usecols=['timestamp', 'side', 'size', 'price'])
    z = z.assign(t=z.timestamp.values * 1000.0).sort_values('t', kind='stable')
    V['byb'] = (z.t.values, z.price.values.astype(float), z['size'].values * np.where(z.side.values == 'Buy', 1.0, -1.0))
    return {k: (t, p, np.r_[0.0, np.cumsum(q)]) for k, (t, p, q) in V.items()}


def vprice(Vk, x):
    t, p, _ = Vk; i = np.searchsorted(t, x, side='right') - 1
    return np.where(i >= 0, p[np.clip(i, 0, None)], np.nan)


def vsvol(Vk, a, b):
    t, _, cs = Vk
    return cs[np.searchsorted(t, b, side='right')] - cs[np.searchsorted(t, a, side='right')]


# ---------------- one day ----------------
def run_day(day):
    mut = mutation_check(); st_bad = selftest()
    if st_bad or not all(mut.values()):
        sys.exit(f'MOVE SELFTEST FAILED: {st_bad} {mut}')
    os.makedirs(OUTM, exist_ok=True); T0 = time.time()
    D = K.load_day(day); st = D['stat']; S1d = f'{K.IO}/step1/{day}'
    full = pd.read_parquet(f'{S1d}/legs_timed2.parquet', columns=['mid', 'signer', 'taker_index', 'first_recv_ns', 'shares', 't_rel'])
    tkf = pd.read_parquet(f'{S1d}/legs_tk.parquet', columns=['tk']).tk.values
    if len(tkf) != len(full):
        sys.exit('REFUSED: legs_tk not aligned with legs_timed2')
    inw = full.mid.isin(list(D['W0'])).values
    G = GP.rank_groups(full.signer.values[inw], full.shares.values[inw].astype(float), full.t_rel.values[inw].astype(float))
    LT = D['LT']; ix = LT.index.values
    LT = LT.assign(signer=full.signer.values[ix].astype('int64'), acct=full.taker_index.values[ix].astype(float),
                   recv=full.first_recv_ns.values[ix].astype(float) / 1e6, tk=np.where(tkf[ix] >= 0, tkf[ix], -1 - ix))
    del full
    wsl = LT.how.values > 0
    kt = np.maximum(LT.anc.values + K.PF_MIN, LT.recv.values - D['SH']) + D['DD']
    kt = np.where(np.isfinite(LT.recv.values), kt, LT.anc.values + np.where(wsl, 60.0, 300.0))
    LT = LT.assign(kt=kt, wsl=wsl, grp=[G.get(int(s_), 'rest') for s_ in LT.signer.values])
    rr = pd.read_parquet(f'{K.IO}/step5/{day}/rows_rl_d2_pk.parquet', columns=['mid', 'tau', 'bn_mid']); rr = rr[rr.mid.isin(D['W0'])]
    BNW = {int(m): (g.tau.values.astype(float), g.bn_mid.values.astype(float)) for m, g in rr.sort_values(['mid', 'tau']).groupby('mid')}
    del rr
    V = load_venues(day); st['venues'] = sorted(V)
    # pass 1: each leg's completed 1 s markout to its maker (taker reputation)
    m1 = np.full(len(LT), np.nan); LTI = LT.groupby('mid', sort=False).indices
    for m, ii in LTI.items():
        if m not in D['BI']:
            continue
        b = D['B'].iloc[D['BI'][m]].sort_values('t_ms', kind='stable'); tf = b.t_ms.values.astype(float)
        ok = (b.b1_p.values >= 0) & (b.a1_p.values > b.b1_p.values)
        mff = pd.Series(np.where(ok, (b.a1_p.values + b.b1_p.values) / 2.0, np.nan)).ffill().values
        x = LT.iloc[ii]; j = np.searchsorted(tf, x.anc.values + 1000.0, side='right') - 1
        mm = np.where(j >= 0, mff[np.clip(j, 0, None)], np.nan)
        m1[ii] = np.where(x.side.values == 0, 1.0, -1.0) * (mm - x.ptick.values)
    mk = np.maximum(LT.kt.values, LT.anc.values + 1060.0)
    REP = {}
    okr = np.isfinite(m1) & np.isfinite(LT.acct.values)
    for a, g in pd.DataFrame(dict(a=LT.acct.values[okr], mk=mk[okr], v=m1[okr] * LT.shares.values[okr], q=LT.shares.values[okr])).groupby('a'):
        o = np.argsort(g.mk.values, kind='stable')
        REP[a] = (g.mk.values[o], np.r_[0.0, np.cumsum(g.v.values[o])], np.r_[0.0, np.cumsum(g.q.values[o])])
    st.update(rows_all=0, rows_no_best=0, rows_v_withheld=0, rows=0, ev_legs_ws=0, ev_legs_e2=0, ev_not_best=0, events=0, ev_no_mid=0,
              tape_leak_checks=0)
    RB = {k: [] for k in BASE_N + TAPE_N + BOOK_N + VEN_N + ['day', 'm', 't'] + [f'{x}_{int(h)}' for x in ('DN', 'CLR', 'ACT') for h in HS]}
    RA = {k: [] for k in A_BN + A_VEN + A_TAPE + A_BOOK + ['mv250', 'mv1000', 'spread', 'standing', 'q', 'q_top5', 'q_field', 'm']}
    nb = len(K.G32)
    for w in K.window_sides(D):
        m, side, sg = w['m'], w['side'], w['sg']; tf = w['tf']; tfl = tf.tolist(); kf = w['kf']; P5, S5, opp = w['P5'], w['S5'], w['opp']
        bb = P5[:, 0]; sb = S5[:, 0]; mff = w['tf_mff'][1]
        x = LT.iloc[LTI[m]] if m in LTI else LT.iloc[0:0]
        own = x[x.side.values == side]; oth = x[x.side.values != side]
        def arr(y):
            o = np.argsort(y.kt.values, kind='stable'); y = y.iloc[o]
            first = ~pd.Series(y.tk.values).duplicated().values
            nprice = pd.Series(y.ptick.values).groupby(y.tk.values).transform('nunique').values > 1
            return dict(kt=y.kt.values, cq=np.r_[0.0, np.cumsum(y.shares.values)], cf=np.r_[0.0, np.cumsum(first)],
                        cmp=np.r_[0.0, np.cumsum(first & nprice)], acct=y.acct.values, pt=y.ptick.values, q=y.shares.values,
                        anc=y.anc.values, wsl=y.wsl.values, tk=y.tk.values, grp=y.grp.values)
        O = arr(own); Q_ = arr(oth)
        run_start = np.zeros(len(bb), 'int64')
        for j in range(1, len(bb)):
            run_start[j] = run_start[j - 1] if bb[j] == bb[j - 1] else j
        X = w['X']; tg = w['tg']; best = w['best']; jk = w['jk']
        bnr250 = X[:, K.G32.index('bn_r250')]
        for i in range(0, w['n'], STEP):
            st['rows_all'] += 1
            t = float(tg[i]); p = best[i]
            if p < 0 or jk[i] < 0:
                st['rows_no_best'] += 1; continue
            v = X[i, nb]
            if not np.isfinite(v) or v == -1.0:
                st['rows_v_withheld'] += 1; continue
            st['rows'] += 1
            for k_, val in zip(BASE_N, list(X[i]) + [float(side)]):
                RB[k_].append(val)
            # targets
            for h in HS:
                dn = dn_target(tfl, bb, side, t, p, h)
                lo, hi = t + L_ACT, t + L_ACT + h
                at = (O['pt'] == p) & O['wsl'] & (O['anc'] > lo) & (O['anc'] <= hi)
                thr = (O['pt'] <= p) if side == 0 else (O['pt'] >= p)
                tol = np.where(O['wsl'], 0.0, E2_TOL)
                clr = dn and bool(np.any(thr & (O['anc'] > lo - tol) & (O['anc'] <= hi + tol)))
                RB[f'DN_{int(h)}'].append(dn); RB[f'CLR_{int(h)}'].append(int(clr)); RB[f'ACT_{int(h)}'].append(int(at.any()))
            # TAPE (known by t)
            n_o = bisect.bisect_right(O['kt'], t); n_q = bisect.bisect_right(Q_['kt'], t)
            if n_o and O['kt'][n_o - 1] > t or n_q and Q_['kt'][n_q - 1] > t:
                sys.exit('MOVE K2 FAILED: a tape feature used a leg known after t')
            st['tape_leak_checks'] += 1
            RB['tv250_own'].append(window_sum(O['kt'], O['cq'], t, 250.0)); RB['tv1000_own'].append(window_sum(O['kt'], O['cq'], t, 1000.0))
            RB['tn1000_own'].append(window_sum(O['kt'], O['cf'], t, 1000.0)); RB['tmp1000_own'].append(window_sum(O['kt'], O['cmp'], t, 1000.0))
            RB['tsl_own'].append(min(t - O['kt'][n_o - 1], 5000.0) if n_o else 5000.0)
            RB['tv250_opp'].append(window_sum(Q_['kt'], Q_['cq'], t, 250.0)); RB['tv1000_opp'].append(window_sum(Q_['kt'], Q_['cq'], t, 1000.0))
            RB['tsl_opp'].append(min(t - Q_['kt'][n_q - 1], 5000.0) if n_q else 5000.0)
            rn, rm = NAN, NAN
            if n_o and np.isfinite(O['acct'][n_o - 1]) and O['acct'][n_o - 1] in REP:
                mk_, cv, cq = REP[O['acct'][n_o - 1]]; c = bisect.bisect_right(mk_, t)
                rn = float(c); rm = cv[c] / cq[c] if cq[c] > 0 else NAN
            RB['rep_n'].append(rn); RB['rep_m'].append(rm)
            # BOOK (known frames)
            j = int(jk[i])
            def back(h):
                return int(np.searchsorted(kf, t - h, side='right') - 1)
            for h, nm in ((100.0, 'dsz100'), (250.0, 'dsz250'), (500.0, 'dsz500')):
                jb = back(h); RB[nm].append(float(sb[j] - sb[jb]) if jb >= 0 and bb[jb] == bb[j] else NAN)
            jb = back(250.0)
            if jb >= 0 and bb[jb] == bb[j]:
                trd = float(np.sum(O['q'][(O['pt'] == bb[j]) & (O['kt'] > t - 250.0) & (O['kt'] <= t)]))
                RB['cxl250'].append(max(0.0, float(sb[jb] - sb[j]) - trd)); RB['add250'].append(max(0.0, float(sb[j] - sb[jb]) + trd))
            else:
                RB['cxl250'].append(NAN); RB['add250'].append(NAN)
            RB['lvl_age'].append(min(t - kf[run_start[j]], 10000.0))
            RB['s2'].append(float(S5[j, 1])); RB['ds2_250'].append(float(S5[j, 1] - S5[jb, 1]) if jb >= 0 and P5[jb, 1] == P5[j, 1] else NAN)
            RB['opp_mv250'].append(sg * float(opp[j] - opp[jb]) if jb >= 0 and opp[j] >= 0 and opp[jb] >= 0 else NAN)
            j1 = back(1000.0); RB['pchg1000'].append(float(np.sum(bb[max(j1, 0) + 1:j + 1] != bb[max(j1, 0):j])) if j1 >= 0 else NAN)
            # VEN (as of t minus the assumed delay)
            for vk in LAT:
                if vk in V:
                    x0 = t - LAT[vk]; p0 = vprice(V[vk], np.array([x0, x0 - 100.0, x0 - 250.0, x0 - 1000.0]))
                    r = 1e4 * np.log(p0[0] / p0[1:])
                    RB[f'{vk}_r100'].append(r[0]); RB[f'{vk}_r250'].append(r[1]); RB[f'{vk}_r1000'].append(r[2])
                    RB[f'{vk}_sv250'].append(float(vsvol(V[vk], x0 - 250.0, x0))); RB[f'{vk}_sv1000'].append(float(vsvol(V[vk], x0 - 1000.0, x0)))
                    RB[f'{vk}_div250'].append(r[1] - bnr250[i])
                else:
                    for k_ in ('r100', 'r250', 'r1000', 'sv250', 'sv1000', 'div250'):
                        RB[f'{vk}_{k_}'].append(NAN)
            RB['day'].append(int(day)); RB['m'].append(m); RB['t'].append(t)
        # Part A: at-best WS-timed events
        tw = (O['anc'] >= w['ws']) & (O['anc'] < w['te'])
        st['ev_legs_e2'] += int(np.sum(tw & ~O['wsl']))
        sel = np.where(tw & O['wsl'])[0]
        st['ev_legs_ws'] += len(sel)
        if not len(sel):
            continue
        grp_ = O['grp']
        evs = {}
        for k in sel:
            dp = GP.depth_of(tfl, P5, opp, side, O['anc'][k], int(O['pt'][k]))
            if dp != 0:
                st['ev_not_best'] += 1; continue
            e = evs.setdefault((O['tk'][k], int(O['pt'][k])), dict(anc=O['anc'][k], q=0.0, q5=0.0, qf=0.0, acct=O['acct'][k]))
            e['anc'] = min(e['anc'], O['anc'][k]); e['q'] += O['q'][k]
            if grp_[k] == 'top 5':
                e['q5'] += O['q'][k]
            else:
                e['qf'] += O['q'][k]
        bt, bm = BNW.get(m, (np.zeros(0), np.zeros(0)))
        def bn_at(xx):
            ii_ = np.searchsorted(bt, xx, side='right') - 1
            return bm[ii_] if ii_ >= 0 else NAN
        for (tk_, p), e in evs.items():
            a = e['anc']; jb_ = bisect.bisect_left(tfl, a) - 1; ja = bisect.bisect_right(tfl, a)
            j25 = bisect.bisect_right(tfl, a + 250.0) - 1; j1k = bisect.bisect_right(tfl, a + 1000.0) - 1
            if jb_ < 0 or not np.isfinite(mff[jb_]) or j25 < 0:
                st['ev_no_mid'] += 1; continue
            st['events'] += 1; mb = mff[jb_]
            RA['mv250'].append(sg * (mff[j25] - mb)); RA['mv1000'].append(sg * (mff[j1k] - mb) if j1k >= 0 else NAN); RA['spread'].append(sg * (mb - p))
            stand = ja < len(bb) and bb[ja] == p; RA['standing'].append(int(stand))
            RA['q'].append(e['q']); RA['q_top5'].append(e['q5']); RA['q_field'].append(e['qf']); RA['m'].append(m)
            b0, b1, bp = bn_at(a), bn_at(a + 250.0), bn_at(a - 250.0)
            RA['bn_c'].append(sg * 1e4 * np.log(b1 / b0) if b0 > 0 and b1 > 0 else NAN)
            RA['bn_pre'].append(sg * 1e4 * np.log(b0 / bp) if b0 > 0 and bp > 0 else NAN)
            for vk in LAT:
                if vk in V:
                    pp = vprice(V[vk], np.array([a - 250.0, a, a + 250.0]))
                    RA[f'{vk}_c'].append(sg * 1e4 * np.log(pp[2] / pp[1])); RA[f'{vk}_pre'].append(sg * 1e4 * np.log(pp[1] / pp[0]))
                    RA[f'{vk}_svc'].append(sg * float(vsvol(V[vk], a, a + 250.0)))
                else:
                    RA[f'{vk}_c'].append(NAN); RA[f'{vk}_pre'].append(NAN); RA[f'{vk}_svc'].append(NAN)
            later = (O['anc'] > a) & (O['anc'] <= a + 250.0) & O['wsl']
            thr = (O['pt'] <= p) if side == 0 else (O['pt'] >= p)
            oth_k = later & thr & (O['tk'] != tk_)
            RA['tq_own'].append(float(np.sum(O['q'][oth_k]))); RA['tn_own'].append(float(len(set(O['tk'][oth_k].tolist()))))
            RA['same_acct'].append(float(np.sum(later & (O['acct'] == e['acct']) & (O['tk'] != tk_))) if np.isfinite(e['acct']) else NAN)
            lq = (Q_['anc'] > a) & (Q_['anc'] <= a + 250.0) & Q_['wsl']; RA['tq_opp'].append(float(np.sum(Q_['q'][lq])))
            if ja < len(bb) and j25 >= ja:
                gone = bb[j25] != p; RA['a_gone'].append(float(gone))
                if bb[ja] == p and not gone:
                    dsz = float(sb[j25] - sb[ja]); trd = float(np.sum(O['q'][(O['pt'] == p) & O['wsl'] & (O['anc'] > tf[ja]) & (O['anc'] <= tf[j25]) & (O['tk'] != tk_)]))
                    RA['a_dsz'].append(dsz); RA['a_cxl'].append(max(0.0, -dsz - trd)); RA['a_add'].append(max(0.0, dsz + trd))
                else:
                    RA['a_dsz'].append(NAN); RA['a_cxl'].append(NAN); RA['a_add'].append(NAN)
                RA['a_oppmv'].append(sg * float(opp[j25] - opp[ja]) if opp[j25] >= 0 and opp[ja] >= 0 else NAN)
            else:
                for k_ in A_BOOK:
                    RA[k_].append(NAN)
        st['window_sides'] = st.get('window_sides', 0) + 1
        if st['window_sides'] % 400 == 0:
            r_ = K.rss_mib(); print(f'{day}: {st["window_sides"]} window-sides, {time.time() - T0:.0f} s, rss {r_:.0f} MiB', flush=True)
            if r_ > MEM_CAP_MIB:
                sys.exit(f'REFUSED: resident memory {r_:.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    lens = {k: len(v_) for k, v_ in RB.items()}
    if len(set(lens.values())) != 1:
        sys.exit(f'MOVE: row columns differ in length {lens}')
    lensa = {k: len(v_) for k, v_ in RA.items()}
    if len(set(lensa.values())) != 1:
        sys.exit(f'MOVE: event columns differ in length {lensa}')
    np.savez_compressed(f'{OUTM}/day_{day}.npz', **{f'B|{k}': np.asarray(v_, np.float64) for k, v_ in RB.items()},
                        **{f'A|{k}': np.asarray(v_, np.float64) for k, v_ in RA.items()})
    json.dump(dict(day=day, stat=st, mutations=mut, ranked_makers=len(G)), open(f'{OUTM}/day_{day}.json', 'w'), default=float)
    print(f'{day}: done in {time.time() - T0:.0f} s; rows {st["rows"]}, events {st["events"]}', flush=True)


# ---------------- report ----------------
def lgb_oof(Xm, y, day, days, objective):
    import lightgbm as lgb
    pr = dict(MS.PARAMS); pr['objective'] = objective
    out = np.full(len(y), np.nan)
    for d in days:
        te = day == d; tr = np.isin(day, days) & ~te
        mdl = lgb.train(pr, lgb.Dataset(Xm[tr], y[tr], free_raw_data=True), MS.ROUNDS)
        out[te] = mdl.predict(Xm[te])
        if K.rss_mib() > MEM_CAP_MIB:
            sys.exit(f'REFUSED: resident memory {K.rss_mib():.0f} MiB > {MEM_CAP_MIB:.0f} MiB')
    return out


def report():
    os.makedirs(OUTM, exist_ok=True); bad = {}; asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    B = {}; A = {}; ST = {}
    for d in DAYS9:
        z = np.load(f'{OUTM}/day_{d}.npz'); ST[d] = json.load(open(f'{OUTM}/day_{d}.json'))
        for k in z.files:
            (B if k.startswith('B|') else A).setdefault(k[2:], []).append(z[k] if k.startswith('B|') else z[k])
        A.setdefault('day', []).append(np.full(len(z['A|mv250']), int(d), float))
    B = {k: np.concatenate(v).astype(np.float32 if k not in ('day', 'm', 't') else np.float64) for k, v in B.items()}
    A = {k: np.concatenate(v) for k, v in A.items()}
    for d in DAYS9:
        if not all(ST[d]['mutations'].values()):
            bad[f'mutations {d}'] = ST[d]['mutations']
    rng = np.random.default_rng(20261112); RES = dict(as_of=asof, days=DAYS9, venue_days=VDAYS)
    L = [f'MOVE public tape, 9 consumed replay days (venues 8), first 60 s; as of {asof}']
    # ---- Part B ----
    bday = B['day'].astype(int); dz = [int(d) for d in DAYS9]; dz8 = [int(d) for d in VDAYS]
    y = B['DN_250'].astype(int)
    plant = np.where(rng.random(len(y)) < 0.2, 1 - y, y).astype(float)
    tp = np.column_stack([B[k] for k in TAPE_N]).copy()
    for d in dz:
        ii = np.where(bday == d)[0]; tp[ii] = tp[rng.permutation(ii)]
    M = lambda names: np.column_stack([B[k] for k in names]).astype(np.float32)
    SETS = {'BASE': (BASE_N, dz), 'BASE+TAPE': (BASE_N + TAPE_N, dz), 'BASE+BOOK': (BASE_N + BOOK_N, dz), 'BASE+TAPE+BOOK': (BASE_N + TAPE_N + BOOK_N, dz),
            'BASE8': (BASE_N, dz8), 'BASE8+VEN': (BASE_N + VEN_N, dz8), 'BASE8+ALL': (BASE_N + TAPE_N + BOOK_N + VEN_N, dz8)}
    P = {}
    for nm, (cols, days) in SETS.items():
        P[('DN_250', nm)] = lgb_oof(M(cols), y, bday, days, 'binary'); print(f'[B] DN_250 {nm} done', flush=True)
    P[('DN_250', 'BASE+PLANT')] = lgb_oof(np.column_stack([M(BASE_N), plant]), y, bday, dz, 'binary')
    P[('DN_250', 'BASE+TAPEperm')] = lgb_oof(np.column_stack([M(BASE_N), tp]), y, bday, dz, 'binary')
    for tg in ('ACT_250', 'CLR_250', 'DN_1000'):
        for nm in ('BASE', 'BASE+TAPE+BOOK'):
            P[(tg, nm)] = lgb_oof(M(SETS[nm][0]), B[tg].astype(int), bday, dz, 'binary'); print(f'[B] {tg} {nm} done', flush=True)
    def aucs(tg, nm, days):
        s = P[(tg, nm)]; yy = B[tg].astype(int); keep = np.isin(bday, days)
        return auc(yy[keep], s[keep]), [auc(yy[bday == d], s[bday == d]) for d in days]
    def comp(nm1, nm0, days, tg='DN_250'):
        a1, d1 = aucs(tg, nm1, days); a0, d0 = aucs(tg, nm0, days)
        up = int(sum(x > z_ for x, z_ in zip(d1, d0))); need = 7
        lab = 'NEW-INFO' if a1 - a0 >= 0.010 and up >= need else ('SMALL' if a1 - a0 >= 0.003 and up >= need else 'NONE')
        return dict(label=lab, auc=a1, auc_base=a0, diff=a1 - a0, days_up=up, n_days=len(days), per_day=[x - z_ for x, z_ in zip(d1, d0)])
    RES['K3'] = dict(plant=comp('BASE+PLANT', 'BASE', dz), tape_perm=comp('BASE+TAPEperm', 'BASE', dz))
    if RES['K3']['plant']['label'] != 'NEW-INFO':
        bad['K3 planted feature did not read NEW-INFO'] = RES['K3']['plant']['diff']
    if RES['K3']['tape_perm']['label'] != 'NONE':
        bad['K3 permuted TAPE did not read NONE'] = RES['K3']['tape_perm']['diff']
    # ---- Part A ----
    aday = A['day'].astype(int); stand = A['standing'] > 0.5
    ya = A['mv250']; oka = np.isfinite(ya)
    FAM = {'BN': A_BN, 'VEN': A_VEN, 'TAPE': A_TAPE, 'BOOK': A_BOOK}
    MA = lambda names: np.column_stack([A[k] for k in names])
    plant_a = ya + rng.normal(0.0, np.nanstd(ya), len(ya))
    tpa = MA(A_TAPE).copy()
    for d in dz:
        ii = np.where(aday == d)[0]; tpa[ii] = tpa[rng.permutation(ii)]
    def r2(pred, yy, msk):
        e = yy[msk] - pred[msk]; return float(1.0 - np.sum(e ** 2) / np.sum((yy[msk] - yy[msk].mean()) ** 2))
    RA_ = {}
    for pop, msk0 in (('standing', stand & oka), ('all', oka)):
        idx = np.where(msk0)[0]; yy = ya[idx]; dd = aday[idx]
        fits = {}
        for nm, cols in list(FAM.items()) + [('ALL', A_BN + A_VEN + A_TAPE + A_BOOK)]:
            days = dz8 if nm in ('VEN', 'ALL') else dz
            if pop == 'all' and nm != 'ALL' and nm != 'BN':
                continue
            pr_ = lgb_oof(MA(cols)[idx], yy, dd, days, 'regression'); fits[nm] = (pr_, days)
            print(f'[A] {pop} {nm} done', flush=True)
        if pop == 'standing':
            fits['PLANT'] = (lgb_oof(plant_a[idx][:, None], yy, dd, dz, 'regression'), dz)
            fits['TAPEperm'] = (lgb_oof(tpa[idx], yy, dd, dz, 'regression'), dz)
        out = {}
        for nm, (pr_, days) in fits.items():
            msk = np.isin(dd, days); out[nm] = dict(r2=r2(pr_, yy, msk), per_day=[r2(pr_, yy, dd == d) for d in days])
            q5 = A['q_top5'][idx][msk]; qf = A['q_field'][idx][msk]; yv = yy[msk]; rv = yv - pr_[msk]
            g_raw = np.sum(q5 * yv) / np.sum(q5) - np.sum(qf * yv) / np.sum(qf); g_res = np.sum(q5 * rv) / np.sum(q5) - np.sum(qf * rv) / np.sum(qf)
            out[nm].update(gap_raw=float(g_raw), gap_resid=float(g_res), gap_share=float(1.0 - g_res / g_raw) if g_raw != 0 else NAN)
        RA_[pop] = out
    RES['A'] = RA_
    if RA_['standing']['PLANT']['r2'] < 0.3:
        bad['K4 planted feature R2 < 0.3'] = RA_['standing']['PLANT']['r2']
    if RA_['standing']['TAPEperm']['r2'] > 0.005:
        bad['K4 permuted TAPE R2 > 0.005'] = RA_['standing']['TAPEperm']['r2']
    if bad:
        json.dump(dict(RES, bad=bad), open(f'{OUTM}/move_failed.json', 'w'), indent=1, default=float)
        sys.exit(f'MOVE CONTROL FAILED -> no table: {bad}')
    # ---- tables and readings ----
    sta = {k: sum(ST[d]['stat'].get(k, 0) for d in DAYS9) for k in ('rows_all', 'rows_no_best', 'rows_v_withheld', 'rows', 'ev_legs_ws', 'ev_legs_e2', 'ev_not_best', 'events', 'ev_no_mid', 'tape_leak_checks')}
    L.append(f'statuses {sta}; controls passed (K1 / K2 synthetic + mutations on every day, K2 leak assertion on {sta["tape_leak_checks"]} rows, K3 / K4 falsifiers, K5 no NaN AUC input)')
    L.append(''); L.append('Part A, explain the move after at-best fills (WS-timed events; c; out-of-day R^2):')
    for pop in ('standing', 'all'):
        idx = (stand if pop == 'standing' else np.ones(len(stand), bool)) & oka
        q5 = A['q_top5'][idx]; qf = A['q_field'][idx]; yv = ya[idx]; y1 = A['mv1000'][idx]; ok1 = np.isfinite(y1)
        L.append(f'  {pop}: events {int(idx.sum())}, mean mv250 {yv.mean():+.3f}, top 5 {np.sum(q5 * yv) / np.sum(q5):+.3f} vs field {np.sum(qf * yv) / np.sum(qf):+.3f} '
                 f'c (share-weighted); mv1000 top 5 {np.sum(q5[ok1] * y1[ok1]) / np.sum(q5[ok1]):+.3f} vs field {np.sum(qf[ok1] * y1[ok1]) / np.sum(qf[ok1]):+.3f}')
        for nm, o in RA_[pop].items():
            L.append(f'    {nm:<9} R^2 {o["r2"]:+.4f} (days {" ".join(f"{x:+.3f}" for x in o["per_day"])}) | gap raw {o["gap_raw"]:+.3f} resid {o["gap_resid"]:+.3f} -> explained {o["gap_share"]:.0%}')
    fam_r2 = {nm: RA_['standing'][nm]['r2'] for nm in FAM}
    srt = sorted(fam_r2.items(), key=lambda kv: -kv[1])
    lab = srt[0][0] if srt[0][1] >= 0.02 and srt[0][1] >= 1.5 * max(srt[1][1], 1e-12) else 'MIXED'
    RES['R370'] = dict(label=lab, r2=fam_r2)
    L.append(''); L.append(f'R370 EXPLAIN: {lab}; R^2 alone ' + ', '.join(f'{k} {v:+.4f}' for k, v in srt))
    gl = {nm: RA_['standing'][nm]['gap_share'] for nm in list(FAM) + ['ALL']}
    RES['R371'] = {nm: dict(share=s_, label='EXPLAINS-GAP' if s_ >= 0.5 else ('PART' if s_ >= 0.2 else 'NO')) for nm, s_ in gl.items()}
    L.append('R371 GAP: ' + ', '.join(f'{nm} {v["share"]:.0%} {v["label"]}' for nm, v in RES['R371'].items()))
    L.append(''); L.append('Part B, missing information (DN_250 = the best moves against the side in (t + 24, t + 274]; out-of-day AUC):')
    yb = B['DN_250']
    L.append(f'  rows {len(yb)}; base rates DN_250 {yb.mean():.3f}, CLR_250 {B["CLR_250"].mean():.3f}, ACT_250 {B["ACT_250"].mean():.3f}, DN_1000 {B["DN_1000"].mean():.3f}')
    RES['R372'] = {}
    for nm1, nm0, days in (('BASE+TAPE', 'BASE', dz), ('BASE+BOOK', 'BASE', dz), ('BASE+TAPE+BOOK', 'BASE', dz), ('BASE8+VEN', 'BASE8', dz8), ('BASE8+ALL', 'BASE8', dz8)):
        c = comp(nm1, nm0, days); RES['R372'][nm1] = c
        L.append(f'  {nm1:<15} vs {nm0:<5}: AUC {c["auc"]:.4f} vs {c["auc_base"]:.4f}, diff {c["diff"]:+.4f}, days above {c["days_up"]} of {c["n_days"]} -> {c["label"]}')
    L.append(f'  falsifiers: planted {RES["K3"]["plant"]["diff"]:+.4f} {RES["K3"]["plant"]["label"]}; TAPE permuted {RES["K3"]["tape_perm"]["diff"]:+.4f} {RES["K3"]["tape_perm"]["label"]}')
    L.append('R372 MISSING-INFO: ' + ', '.join(f'{k} {v["label"]}' for k, v in RES['R372'].items()))
    ad, add_ = aucs('DN_250', 'BASE', dz); aa, aad = aucs('ACT_250', 'BASE', dz)
    na = int(sum(x > z_ for x, z_ in zip(aad, add_))); nd = int(sum(z_ > x for x, z_ in zip(aad, add_)))
    lab = 'TRACKS-ACTIVITY' if na >= 7 else ('TRACKS-HARM' if nd >= 7 else 'MIXED')
    RES['R373'] = dict(label=lab, auc_dn=ad, auc_act=aa, days_act_higher=na, days_dn_higher=nd)
    for tg in ('ACT_250', 'CLR_250', 'DN_1000'):
        a0, _ = aucs(tg, 'BASE', dz); a1, d1 = aucs(tg, 'BASE+TAPE+BOOK', dz); RES[f'B|{tg}'] = dict(base=a0, tape_book=a1)
        L.append(f'  {tg}: BASE {a0:.4f}, BASE+TAPE+BOOK {a1:.4f}')
    L.append(f'R373 HARM-VS-ACTIVITY: {lab}; BASE AUC DN_250 {ad:.4f} vs ACT_250 {aa:.4f}; days ACT higher {na}, DN higher {nd}')
    # feature gain of the best extended model (descriptive)
    json.dump(RES, open(f'{OUTM}/move.json', 'w'), indent=1, default=float)
    open(f'{OUTM}/move.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        b_ = selftest(); mt = mutation_check()
        print('selftest', 'OK' if not b_ else b_, 'mutations caught', mt)
        sys.exit(0 if not b_ and all(mt.values()) else 1)
    elif cmd == 'day':
        run_day(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit(__doc__)
