"""B12f (2026-11-11, USER "Yea check Binance activity during those times, and imbalance etc"). Read-only, descriptive.
220 consumed 11-08 cohort windows, first minute; G = 1 day < 5: point estimates only.

Part A, Binance activity at the top makers' takes. Events exactly as B12e / B12e v2 (SMART = taker orders at the best of
the top-5 makers' addresses; RANDOM = 5 times per SMART order, same window and side, seed 20261111; OTHER = other
takers). Sources (HF collectors, 11-08 hour files, this box's receive clock): trade (recv_ns, E, T, id, price, qty,
is_buyer_maker; aggressor = seller if is_buyer_maker else buyer) and bookTicker (best bid / ask and their sizes).
Features, signed so that + favours the Predict taker (a taker hitting asks buys Up, i.e. bets on BTC up; hitting bids
bets on BTC down): flow_1s / flow_250ms = aggressor buy minus sell BTC over (t - 1050, t - 50] / (t - 300, t - 50];
flow_after = the same over (t - 50, t + 500]; imb = (bid size - ask size) / (bid size + ask size) at the last update <= t
- 50; activity (unsigned) = trades and bookTicker updates over (t - 1050, t - 50].
Part B, Predict book imbalance and queue, all at-best maker fills (guard ON / OFF at lag 50, value known). Side class
from the frame before: thin (the maker side's size at the best <= half the other side's), balanced, heavy (>= twice).
 B1 incremental to our guard: within cells guard x quintile of our guard value v (at anc - 50), the volume-weighted gap
    of thin-side fills against the rest (ranks >= 1 except ours): D_thin = sum_c w_thin(c) (v_thin(c) - v_rest(c)).
 B2 the thin side while the guard is ON (where v3 quotes): its share of guard-ON at-best volume and its net.
 B3 queue on the heavy side: heavy-side fills from trades that took more than half the shown level (reach r = taker's
    shares at the level / shown size > 0.5: what an order at the back is reached by) against r <= 0.5.

Readings DECLARED BEFORE THE FIRST RUN (evaluated in code; nulls 500 draws, seed 20261111):
 R80 flow lead    FLOW-LEAD if mean flow_1s at SMART minus RANDOM >= 0.2 x RANDOM's SD of flow_1s and above its null
                  p99 (SMART / RANDOM label permuted within window); else NO-FLOW-LEAD.
 R81 book lead    BN-BOOK-LEAD if mean imb at SMART minus RANDOM >= +0.05 and above its null p99; else NO-BN-BOOK-LEAD.
 R82 activity     BN-ACTIVE if mean trades over the second before at SMART >= 1.5 x RANDOM's; else BN-NORMAL.
 R85 incremental  IMBALANCE-INCREMENTAL if D_thin <= -0.30 c/sh and below its null p01 (thin / rest label permuted within
                  window, side and cell); else NOT-INCREMENTAL.
 R86 thin when ON THIN-LOSES-WHEN-ON if the thin side's net at 1 s with the guard ON is < 0; else THIN-OK-WHEN-ON.
 R87 heavy queue  BACK-OF-HEAVY-LOSES if heavy-side fills with r > 0.5 net < 0 and those with r <= 0.5 net > 0; else
                  NO-QUEUE-SPLIT.
What the outcomes mean (fixed in advance): FLOW-LEAD / BN-BOOK-LEAD = Binance order flow or its top-of-book leans their
way before they take on Predict, even with the price still: a Binance signal our guard could use. BN-ACTIVE = they take
when Binance is busy. IMBALANCE-INCREMENTAL = Predict's own book imbalance marks bad fills beyond what our guard
value knows: a guard input to test. THIN-LOSES-WHEN-ON = quoting the thin side loses even when our guard says quote.
BACK-OF-HEAVY-LOSES = on heavy levels the trades that reach the back lose, the front earns: the heavy-side loss is
queue place.
Controls (any failure -> no table): B12e's controls; the trade / bookTicker window sums and imbalance on planted rows,
including the aggressor sign; the R85 null on planted fills (thin worse by 1 c -> IMBALANCE-INCREMENTAL, no difference ->
NOT-INCREMENTAL); the label rules; K-check: the event counts equal b12e.json's.
Limits fixed in advance: one consumed Sunday; Predict trade times are reports (~0.2 s after the match); the Binance
receive clock is this box's, assumed the same as the Predict collector's; pro rata fill values rank moments, they do not
price an order we can place (B1a: the fill model is not validated at the order level); three SMART addresses.
Usage: predictfun/research/run_light.sh b12f_activity_imbalance -- python3 -m predictfun.research.pf_b12f_activity_imbalance <out_dir>
       python3 -m predictfun.research.pf_b12f_activity_imbalance --selftest
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB
from . import pf_b12c_model_or_speed as C1
from . import pf_b12d_bursts as DB
from . import pf_b12e_smart_takes as E1
from . import pf_b12e2_binance_raw as E2

TR = '/home/yuqing/ctaNew/data/mm_hf/raw/trade/BTCUSDT'
N_NULL = 500; SEED = 20261111
NAN = float('nan')


def window_sum(t, x, a, b):
    """Sum of x over (a, b] for time-sorted t (vectorised over arrays a, b)."""
    c = np.r_[0.0, np.cumsum(x)]
    return c[np.searchsorted(t, b, side='right')] - c[np.searchsorted(t, a, side='right')]


def aggressor_sign(is_buyer_maker):
    """+1 when the buyer was the aggressor (is_buyer_maker 0), -1 when the seller was (1)."""
    return np.where(np.asarray(is_buyer_maker) == 1, -1.0, 1.0)


def thin_class(q, qo):
    r = q / qo
    return np.where(~np.isfinite(r), 'n/a', np.where(r <= 0.5, 'thin', np.where(r >= 2.0, 'heavy', 'balanced')))


def d_thin(cell, thin, sh, val, n):
    wT = np.bincount(cell[thin], sh[thin], n); wR = np.bincount(cell[~thin], sh[~thin], n); b = (wT > 0) & (wR > 0)
    if not b.any():
        return NAN
    vT = np.divide(np.bincount(cell[thin], sh[thin] * val[thin], n), wT, out=np.zeros(n), where=b)
    vR = np.divide(np.bincount(cell[~thin], sh[~thin] * val[~thin], n), wR, out=np.zeros(n), where=b)
    w = np.where(b, wT, 0.0)
    return float(np.sum(w / w.sum() * (vT - vR)))


def null_d_thin(cell, thin, sh, val, n, block, draws, seed):
    rng = np.random.default_rng(seed); o = np.argsort(block, kind='stable'); res = np.empty(draws)
    for d in range(draws):
        o2 = np.lexsort((rng.random(len(block)), block)); l2 = np.empty_like(thin); l2[o] = thin[o2]
        res[d] = d_thin(cell, l2, sh, val, n)
    return res


def label_r80(d, sd, p99):
    return 'n/a' if not BB.fin(d, sd, p99) else ('FLOW-LEAD' if (d >= 0.2 * sd and d > p99) else 'NO-FLOW-LEAD')


def label_r81(d, p99):
    return 'n/a' if not BB.fin(d, p99) else ('BN-BOOK-LEAD' if (d >= 0.05 and d > p99) else 'NO-BN-BOOK-LEAD')


def label_r82(ms, mr):
    return 'n/a' if not BB.fin(ms, mr) or mr <= 0 else ('BN-ACTIVE' if ms >= 1.5 * mr else 'BN-NORMAL')


def label_r85(d, p01):
    return 'n/a' if not BB.fin(d, p01) else ('IMBALANCE-INCREMENTAL' if (d <= -0.30 and d < p01) else 'NOT-INCREMENTAL')


def label_r86(v):
    return 'n/a' if not BB.fin(v) else ('THIN-LOSES-WHEN-ON' if v < 0 else 'THIN-OK-WHEN-ON')


def label_r87(vb, vf):
    return 'n/a' if not BB.fin(vb, vf) else ('BACK-OF-HEAVY-LOSES' if (vb < 0 and vf > 0) else 'NO-QUEUE-SPLIT')


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    oke, repe = E1.controls(); chk('B12e_controls', oke)
    t = np.array([0.0, 100.0, 200.0, 300.0]); x = np.array([1.0, 2.0, 3.0, 4.0])
    chk('A1_window_sum', list(window_sum(t, x, np.array([0.0, -1.0, 150.0]), np.array([200.0, 0.0, 300.0]))) == [5.0, 1.0, 7.0])
    chk('A2_aggressor', list(aggressor_sign([0, 1]) * np.array([0.5, 0.5])) == [0.5, -0.5])
    chk('A3_classes', list(thin_class(np.array([10.0, 50.0, 300.0, 5.0]), np.array([100.0, 60.0, 100.0, np.nan]))) == ['thin', 'balanced', 'heavy', 'n/a'])
    rng = np.random.default_rng(41); n = 6000; cell = rng.integers(0, 10, n); thin = rng.random(n) < 0.3; sh = 1.0 + rng.random(n); blk = rng.integers(0, 60, n)
    res = []
    for plant in (-1.0, 0.0):
        val = rng.normal(0, 1, n) + np.where(thin, plant, 0.0); d = d_thin(cell, thin, sh, val, 10)
        nul = null_d_thin(cell, thin, sh, val, 10, blk * 10 + cell, 200, 3); res.append(label_r85(d, float(np.percentile(nul, 1))))
    chk('A4_null', res == ['IMBALANCE-INCREMENTAL', 'NOT-INCREMENTAL'])
    chk('A5_rules', label_r80(1.0, 2.0, 0.5) == 'FLOW-LEAD' and label_r80(0.1, 2.0, 0.05) == 'NO-FLOW-LEAD' and label_r81(0.06, 0.02) == 'BN-BOOK-LEAD'
        and label_r81(0.02, 0.01) == 'NO-BN-BOOK-LEAD' and label_r82(3.0, 1.5) == 'BN-ACTIVE' and label_r82(2.0, 1.5) == 'BN-NORMAL'
        and label_r86(-0.1) == 'THIN-LOSES-WHEN-ON' and label_r87(-0.5, 0.3) == 'BACK-OF-HEAVY-LOSES' and label_r87(-0.5, -0.1) == 'NO-QUEUE-SPLIT'
        and label_r85(NAN, 0.0) == 'n/a' and label_r82(1.0, 0.0) == 'n/a')
    rep['B12e_controls_detail'] = repe
    return ok, rep


def load_binance(W0):
    """Trades and bookTicker sizes for the cohort windows' first minutes (+/- 3 s), from the 11-08 hour files."""
    spans = sorted((ws - 3000.0, ws + 63000.0, m) for m, ws in W0.items())
    hours = sorted({int(((ws / 1000.0) % 86400) // 3600) for ws in W0.values()} | {int((((ws + 63000.0) / 1000.0) % 86400) // 3600) for ws in W0.values()})
    trd = {m: [] for m in W0}; btk = {m: [] for m in W0}; stat = dict(trade_files=0, bt_files=0, missing=[])
    for h in hours:
        ft = f'{TR}/20261108_{h:02d}.csv.gz'; fb = f'{E2.BT}/20261108_{h:02d}.csv.gz'
        if os.path.exists(ft):
            d = pd.read_csv(ft, header=None, usecols=[0, 5, 6], names=['recv_ns', 'qty', 'ibm'], dtype={'recv_ns': 'int64', 'qty': 'float64', 'ibm': 'int64'})
            stat['trade_files'] += 1; t = d.recv_ns.values / 1e6; s = aggressor_sign(d.ibm.values) * d.qty.values
            for a, b, m in spans:
                i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
                if i1 > i0:
                    trd[m].append((t[i0:i1], s[i0:i1]))
        else:
            stat['missing'].append(f'trade {h}')
        if os.path.exists(fb):
            d = pd.read_csv(fb, header=None, usecols=[0, 5, 7], names=['recv_ns', 'bq', 'aq'], dtype={'recv_ns': 'int64', 'bq': 'float64', 'aq': 'float64'})
            stat['bt_files'] += 1; t = d.recv_ns.values / 1e6; imb = (d.bq.values - d.aq.values) / (d.bq.values + d.aq.values)
            for a, b, m in spans:
                i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
                if i1 > i0:
                    btk[m].append((t[i0:i1], imb[i0:i1]))
        else:
            stat['missing'].append(f'bookTicker {h}')

    def cat(parts):
        if not parts:
            return None
        tt = np.concatenate([p[0] for p in parts]); vv = np.concatenate([p[1] for p in parts]); o = np.argsort(tt, kind='stable')
        return tt[o], vv[o]
    return {m: cat(p) for m, p in trd.items()}, {m: cat(p) for m, p in btk.items()}, stat


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1)); print('selftest', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    b12e = json.load(open(E2.B12E))
    X = BB.build(); R = X['R']; W0 = X['W0']; OWN = X['OWN']; G = X['G']; LINE = X['LINE']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    # ---- events exactly as B12e ----
    M = pd.read_parquet(E1.MATCH, columns=['mid', 'exec_s', 'taker', 'maker', 'ptick', 'shares'])
    M = M[M.mid.isin(W0)].copy(); M['exec_s'] = M.exec_s.astype('int64'); M['sh6'] = M.shares.round(6)
    key = ['mid', 'exec_s', 'ptick', 'sh6']
    Mu = M.merge(M.groupby(key).size().rename('n').reset_index().query('n == 1')[key], on=key)
    L = R[['mid', 'exec_s', 'ptick', 'shares', 'signer', 'grp']].copy(); L['exec_s'] = L.exec_s.astype('int64'); L['sh6'] = L.shares.round(6)
    Lu = L.merge(L.groupby(key).size().rename('nl').reset_index().query('nl == 1')[key], on=key)
    J = Lu.merge(Mu[key + ['taker', 'maker']], on=key)
    addr = J.groupby('signer').maker.agg(lambda s: s.value_counts().index[0]).str.lower()
    tk = J.groupby('grp').taker.agg(lambda s: s.value_counts().index[0]).str.lower()
    if not (OWN in addr.index and addr[OWN] == str(json.load(open(E1.WHO))['our_address']).lower()):
        sys.exit('CONTROL FAILED (KJ): own address')
    top5 = sorted(int(s) for s in R.signer[R.g == 'top 5'].unique()); smart_addr = {addr[s]: s for s in top5 if s in addr.index}
    R['taker'] = tk.reindex(R.grp.values).values
    A = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & R.taker.notna()]
    TO = A.groupby('grp').agg(mid=('mid', 'first'), side=('side', 'first'), anc=('anc', 'first'), taker=('taker', 'first')).reset_index()
    TO['smart'] = TO.taker.isin(set(smart_addr))
    rng = np.random.default_rng(E1.SEED); Sm = TO[TO.smart]
    rr = pd.DataFrame(dict(mid=np.repeat(Sm.mid.values, E1.K_RAND), side=np.repeat(Sm.side.values, E1.K_RAND)))
    rr['anc'] = rr.mid.map(W0).values + rng.uniform(1050.0, 59000.0, len(rr))
    EV = pd.concat([Sm[['mid', 'side', 'anc']].assign(kind='SMART'), TO[~TO.smart][['mid', 'side', 'anc']].assign(kind='OTHER'), rr.assign(kind='RANDOM')], ignore_index=True)
    counts = {k: int((EV.kind == k).sum()) for k in ('SMART', 'OTHER', 'RANDOM')}
    if counts != b12e['counts']:
        sys.exit(f'CONTROL FAILED (K-check counts) -> no table: {counts}')
    # ---- Part A: Binance activity ----
    TRd, BTd, bst = load_binance(W0); sgT = E1.tsign(EV.side.values)
    f = {k: np.full(len(EV), np.nan) for k in ('flow_1s', 'flow_250ms', 'flow_after', 'imb', 'n_trades_1s', 'n_bt_1s')}
    for m, ii in EV.groupby('mid', sort=False).indices.items():
        t = EV.anc.values[ii]
        if TRd.get(m) is not None:
            tt, ss = TRd[m]
            f['flow_1s'][ii] = window_sum(tt, ss, t - 1050.0, t - 50.0); f['flow_250ms'][ii] = window_sum(tt, ss, t - 300.0, t - 50.0)
            f['flow_after'][ii] = window_sum(tt, ss, t - 50.0, t + 500.0); f['n_trades_1s'][ii] = window_sum(tt, np.ones(len(tt)), t - 1050.0, t - 50.0)
        if BTd.get(m) is not None:
            tb, ib = BTd[m]; j = np.searchsorted(tb, t - 50.0, side='right') - 1; jj = np.clip(j, 0, None)
            f['imb'][ii] = np.where((j >= 0) & (t - 50.0 - tb[jj] <= E2.STALE), ib[jj], np.nan)
            f['n_bt_1s'][ii] = window_sum(tb, np.ones(len(tb)), t - 1050.0, t - 50.0)
    for k in ('flow_1s', 'flow_250ms', 'flow_after', 'imb'):
        EV[k] = f[k] * sgT
    EV['n_trades_1s'] = f['n_trades_1s']; EV['n_bt_1s'] = f['n_bt_1s']
    SR = EV[EV.kind.isin(['SMART', 'RANDOM'])]; lab = SR.kind.values == 'SMART'; blk = pd.factorize(SR.mid)[0]
    tabA = {}
    for k in ('flow_1s', 'flow_250ms', 'flow_after', 'imb', 'n_trades_1s', 'n_bt_1s'):
        row = {kd: dict(n=int(np.isfinite(EV[k][EV.kind == kd]).sum()), mean=float(np.nanmean(EV[k][EV.kind == kd])), median=float(np.nanmedian(EV[k][EV.kind == kd])),
                        share_pos=float(np.nanmean(EV[k][EV.kind == kd] > 0)), sd=float(np.nanstd(EV[k][EV.kind == kd]))) for kd in ('SMART', 'OTHER', 'RANDOM')}
        d, p99, p = E1.diff_null(SR[k].values.astype(float), lab, blk, N_NULL, SEED); row['SMART_minus_RANDOM'] = dict(diff=d, null_p99=p99, p_one_sided=p)
        tabA[k] = row
    r80 = label_r80(tabA['flow_1s']['SMART_minus_RANDOM']['diff'], tabA['flow_1s']['RANDOM']['sd'], tabA['flow_1s']['SMART_minus_RANDOM']['null_p99'])
    r81 = label_r81(tabA['imb']['SMART_minus_RANDOM']['diff'], tabA['imb']['SMART_minus_RANDOM']['null_p99'])
    r82 = label_r82(tabA['n_trades_1s']['SMART']['mean'], tabA['n_trades_1s']['RANDOM']['mean'])
    # ---- Part B: Predict imbalance and queue ----
    FR = DB.frames(W0)
    V = R[R.val_ok & (R.status == 'ok') & (R.depth == 0) & R.g50.isin([B.G_ON, B.G_OFF])].copy(); qo = np.full(len(V), np.nan); v50 = np.full(len(V), np.nan)
    for (m, side), ii in V.groupby(['mid', 'side'], sort=False).indices.items():
        tf, S = FR[m]; j0 = np.searchsorted(tf, V.anc.values[ii], side='left') - 1; jj = np.clip(j0, 0, None)
        qo[ii] = np.where(j0 >= 0, S[1 - side][1][jj, 0], np.nan)
        tau, vb, va = G[m]; v50[ii] = C1.vlook(tau, (vb, va)[side], V.anc.values[ii] - 50.0, LINE)
    V['q_other'] = qo; V['v50'] = v50; V['cls'] = thin_class(V.Q.values, qo); V['r'] = V.s_lvl.values / V.Q.values
    F = V[(V.g != 'ours') & (V.cls != 'n/a') & np.isfinite(V.v50.values)].copy()
    e5 = C1.qe(F.v50.values, 5); F['cell'] = (5 * (F.g50.values == B.G_OFF) + BB.bucket(F.v50.values, e5)).astype(np.int64)
    thin = F.cls.values == 'thin'; sh = F.shares.values; val = F.v1.values
    dth = d_thin(F.cell.values, thin, sh, val, 10)
    nb = pd.factorize(pd.Series(list(zip(F.mid.values, F.side.values, F.cell.values))))[0]
    nul = null_d_thin(F.cell.values, thin, sh, val, 10, nb, N_NULL, SEED); p01 = float(np.percentile(nul, 1))
    r85 = label_r85(dth, p01)
    tabB = {}
    for gs, gv in (('ON', B.G_ON), ('OFF', B.G_OFF)):
        x = F[F.g50.values == gv]; s_ = x.shares.values; tot = s_.sum(); tabB[gs] = {}
        for c in ('thin', 'balanced', 'heavy'):
            mk = x.cls.values == c
            tabB[gs][c] = dict(legs=int(mk.sum()), share=float(s_[mk].sum() / tot) if tot else NAN, net1=B.wmean(s_[mk], x.v1.values[mk]), net5=B.wmean(s_[mk], x.v5.values[mk]))
    r86 = label_r86(tabB['ON']['thin']['net1'])
    H = F[(F.cls.values == 'heavy') & (F.sweep.values >= 0) & np.isfinite(F.r.values)]
    back = H.r.values > 0.5
    vb_ = B.wmean(H.shares.values[back], H.v1.values[back]); vf_ = B.wmean(H.shares.values[~back], H.v1.values[~back])
    r87 = label_r87(vb_, vf_)
    hq = dict(back=dict(legs=int(back.sum()), net1=vb_), front=dict(legs=int((~back).sum()), net1=vf_))
    # ours, descriptive
    U = V[(V.g == 'ours') & (V.cls != 'n/a')]; ours = {}
    for gs, gv in (('ON', B.G_ON), ('OFF', B.G_OFF)):
        x = U[U.g50.values == gv]
        ours[gs] = {c: dict(legs=int((x.cls == c).sum()), net1=B.wmean(x.shares.values[x.cls.values == c], x.v1.values[x.cls.values == c])) for c in ('thin', 'balanced', 'heavy')}
    Uh = U[U.cls == 'heavy']; ours['heavy_median_reach'] = BB.wmedian(Uh.r.values[np.isfinite(Uh.r.values)], Uh.shares.values[np.isfinite(Uh.r.values)])
    ours['field_heavy_median_reach'] = BB.wmedian(H.r.values, H.shares.values)
    readings = dict(R80=r80, R81=r81, R82=r82, R85=r85, R86=r86, R87=r87)
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K0=k0['ok'], counts=counts,
               binance_files=bst, partA=tabA, partB=dict(D_thin=dth, null_p01=p01, null_p50=float(np.percentile(nul, 50)), by_guard=tabB, heavy_queue=hq, ours=ours),
               readings=readings)
    json.dump(res, open(f'{out}/b12f.json', 'w'), indent=1, default=float)
    g = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.3f}'
    Lt = [f"B12f Binance activity at the top makers' takes, and Predict imbalance / queue. {res['as_of_note']}",
          f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; counts {counts}; Binance files {bst}", '',
          "Part A (flows in BTC, imbalance -1..1, + = the Predict taker's favour; activity unsigned): feature | SMART mean / median / share > 0 | OTHER mean | RANDOM mean / sd / share > 0 | SMART - RANDOM (null p99, p)"]
    for k, r in tabA.items():
        s, o, q = r['SMART'], r['OTHER'], r['RANDOM']
        Lt.append(f"{k} | {g(s['mean'])} / {g(s['median'])} / {s['share_pos'] * 100:.0f}% | {g(o['mean'])} | {g(q['mean'])} / {g(q['sd'])} / {q['share_pos'] * 100:.0f}% | "
                  f"{g(r['SMART_minus_RANDOM']['diff'])} ({g(r['SMART_minus_RANDOM']['null_p99'])}, {r['SMART_minus_RANDOM']['p_one_sided']:.3f})")
    Lt += ['', f"Part B: D_thin within guard x v-quintile cells {g(dth)} c/sh (null p01 {g(p01)}, p50 {g(float(np.percentile(nul, 50)))})",
           'by guard state, field at-best fills: class share / net 1 s / net 5 s / legs']
    for gs, x in tabB.items():
        Lt.append(f"  {gs}: " + '; '.join(f"{c} {v['share'] * 100:.0f}% {g(v['net1'])} {g(v['net5'])} ({v['legs']})" for c, v in x.items()))
    Lt.append(f"heavy side by reach: back (r > 0.5) {g(hq['back']['net1'])} ({hq['back']['legs']}); front (r <= 0.5) {g(hq['front']['net1'])} ({hq['front']['legs']}); "
              f"median reach on heavy levels: field {ours['field_heavy_median_reach']:.3f}, ours {ours['heavy_median_reach']:.3f}")
    Lt.append('ours: ' + '; '.join(f"{gs} " + ', '.join(f"{c} {g(v['net1'])} ({v['legs']})" for c, v in ours[gs].items()) for gs in ('ON', 'OFF')))
    Lt += ['', 'Readings: ' + json.dumps(readings)]
    open(f'{out}/b12f.txt', 'w').write('\n'.join(Lt) + '\n'); print('\n'.join(Lt))


if __name__ == '__main__':
    main()
