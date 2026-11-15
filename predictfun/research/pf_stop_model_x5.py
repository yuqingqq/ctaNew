"""STOP-MODEL EXTRA INPUTS x5 (2026-11-15, session aea5bfb9): the order-flow inputs the input review found missing.

USER (2026-11-15): "If we include sufficient inputs like imbalance, flow etc". The review: imbalance is in and strong (Binance top-of-book
imbalance change = the top input at 250 ms / 1 s; Predict L1-3 imbalance 8-11% of SHAP); Binance taker flow is in but weak; Predict trade
prints cannot be a sub-second input (they reach us 2.5 / 6.5 / 13 s after execution, p05 / p50 / p95: a polled API - the book frames carry
the same trades ~53 ms after the venue, already in as L1 size depletion and settlement-pending flows). MISSING: order-flow imbalance (OFI)
from book updates on Predict, and Binance's book beyond L1 (20-level snapshots every ~100 ms are collected and unused).

FAMILIES (all known by t):
 PF_OFI   Predict, from the frames KNOWN at t (frame index <= jk; known time kf, the model's clock):
          adv_ofi250 / adv_ofi1000 = the L1 order-flow imbalance of Cont, Kukanov & Stoikov (e_n = 1{b_n >= b_n-1} qb_n - 1{b_n <= b_n-1} qb_n-1
          - 1{a_n <= a_n-1} qa_n + 1{a_n >= a_n-1} qa_n-1 over consecutive valid frames; > 0 = buying pressure) summed over the frames known in
          (t - h, t], / W30, x adv (asks +1, bids -1: > 0 = against our side);
          adv_dd3_250 / adv_dd3_1000 = the change of (bid size - ask size) within the best 3 levels between the frame known at t - h and at t,
          / W30, x adv.
 BN_DEPTH Binance depth20 snapshots, known at event time + 1.5 + 2 ms (the bookTicker clock):
          adv_bnd_imb5 / adv_bnd_imb20 = (bid - ask) / (bid + ask) size within 5 / 20 levels of the last known snapshot, x adv;
          adv_d_bnd_imb5_250 = its change over 250 ms; adv_bnd_ofi250 / adv_bnd_ofi1000 = multi-level OFI (Cont's e per level 1-5, summed)
          over the snapshots known in (t - h, t], / the mean 5-level depth of the snapshots known in the last 30 s, x adv; bnd_age = t - the
          last snapshot's known time.
OUTPUT rows_<day>_x5.parquet: m, side, k in the rows file's order + the 10 inputs (statuses counted in rows_<day>_x5.json).
CONTROLS (any failure -> no output): synthetic selftests (OFI = a brute-force sum; the depth parser; known-time lookups) with planted errors
(OFI reading frame jk + 1; depth reading the snapshot after t) caught by their own checks; on every day (i) PAIR falsifier: frames after jk
perturbed on both sides (prices +-1 tick, sizes x U(0.2, 3)) change no PF_OFI input at k; (ii) DEPTH falsifier: snapshots known after t
perturbed change no BN_DEPTH input at t; (iii) keys equal the rows file's.
Day rules: the loader's boundary-hour filter (D12) is on (via pf_stop_model); Binance hours = the day's 24 and the previous hour unless sealed.
Usage (run_light, PF_SLOT=claude): pf_stop_model_x5.py selftest | build <day>
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_stop_model as M                                 # noqa: E402

PF_OFI = ['adv_ofi250', 'adv_ofi1000', 'adv_dd3_250', 'adv_dd3_1000']
BN_DEPTH = ['adv_bnd_imb5', 'adv_bnd_imb20', 'adv_d_bnd_imb5_250', 'adv_bnd_ofi250', 'adv_bnd_ofi1000', 'bnd_age']
X5_F = PF_OFI + BN_DEPTH
DEPTH_DIR = '/home/yuqing/ctaNew/data/mm_hf/raw/depth20/BTCUSDT'
SCALE_MS = 30000.0; FZ_EVERY = 20; FZ_STEP = 41; FD_N = 400
MUT_OFI_NEXT = MUT_BND_NEXT = 0


def adv_of(side):
    return 1.0 if side == 1 else -1.0


def pair_ofi(W, jk=None):
    """Per step: L1 OFI over the frames known in (t - h, t] and the best-3 depth-imbalance change between the frames known at t - h and t
    (raw sign: > 0 = buying pressure). Frames are both sides' (one shared frame list); a transition counts only between two valid frames."""
    w0, w1 = W[0], W[1]; kf = np.asarray(w0['kf'], float); tg = np.asarray(w0['tg'], float)
    jk = np.asarray(w0['jk'], 'int64') if jk is None else jk
    hi = np.minimum(jk + 1, len(kf) - 1) if MUT_OFI_NEXT else jk
    Pb, Sb = np.asarray(w0['P5']), np.asarray(w0['S5'], float); Pa, Sa = np.asarray(w1['P5']), np.asarray(w1['S5'], float)
    b, a, qb, qa = Pb[:, 0].astype(float), Pa[:, 0].astype(float), Sb[:, 0], Sa[:, 0]
    valid = (b >= 0) & (a >= 0) & (a > b)
    e = np.zeros(len(kf))
    if len(kf) > 1:
        pv = valid[1:] & valid[:-1]
        e[1:] = np.where(pv, (b[1:] >= b[:-1]) * qb[1:] - (b[1:] <= b[:-1]) * qb[:-1] - (a[1:] <= a[:-1]) * qa[1:] + (a[1:] >= a[:-1]) * qa[:-1], 0.0)
    ce = np.cumsum(e)
    d3 = np.where(valid, sum(np.where(Pb[:, i] >= 0, Sb[:, i], 0.0) for i in range(3)) - sum(np.where(Pa[:, i] >= 0, Sa[:, i], 0.0) for i in range(3)), np.nan)
    out = {}
    for h in (250, 1000):
        lo = np.searchsorted(kf, tg - h, side='right') - 1
        ok = hi >= 0
        out[f'ofi{h}'] = np.where(ok, ce[np.clip(hi, 0, None)] - np.where(lo >= 0, ce[np.clip(lo, 0, None)], 0.0), np.nan)
        out[f'dd3_{h}'] = np.where(ok & (lo >= 0), d3[np.clip(hi, 0, None)] - d3[np.clip(lo, 0, None)], np.nan)
    return out


def parse_depth(df):
    """depth20 rows -> arrays: bid / ask prices and sizes of levels 1-20 (best first)."""
    def lv(col):
        s = np.nan_to_num(df[col].str.replace('@', '|', regex=False).str.split('|', expand=True).iloc[:, :40].astype(float).to_numpy(), nan=0.0)
        return s[:, 0::2], s[:, 1::2]
    bp, bq = lv('bids'); ap, aq = lv('asks')
    return bp, bq, ap, aq


def depth_arrays(hours, b_in):
    """The day's Binance depth snapshots, globally sorted by known time: known time, imbalance within 5 / 20 levels, cumulative multi-level OFI
    (levels 1-5) and cumulative 5-level depth."""
    ks, i5, i20, es, d5 = [], [], [], [], []; st = dict(files=0, missing=[], snapshots=0)
    prev = None
    for h in hours:
        p = f'{DEPTH_DIR}/{h}.csv.gz'
        if not os.path.exists(p):
            st['missing'].append(h); continue
        df = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bids', 'asks'], usecols=['E', 'u', 'bids', 'asks'])
        df = df.dropna().drop_duplicates('u', keep='first'); st['files'] += 1
        bp, bq, ap, aq = parse_depth(df); k = df.E.values.astype(np.float64) + b_in
        ks.append(k); sb5, sa5 = bq[:, :5].sum(1), aq[:, :5].sum(1); sb20, sa20 = bq.sum(1), aq.sum(1)
        i5.append((sb5 - sa5) / np.maximum(sb5 + sa5, 1e-12)); i20.append((sb20 - sa20) / np.maximum(sb20 + sa20, 1e-12)); d5.append(sb5 + sa5)
        es.append((bp[:, :5], bq[:, :5], ap[:, :5], aq[:, :5])); del df
    if not ks:
        return None, st
    k = np.concatenate(ks); o = np.argsort(k, kind='stable'); k = k[o]
    bp = np.concatenate([x[0] for x in es])[o]; bq = np.concatenate([x[1] for x in es])[o]; ap = np.concatenate([x[2] for x in es])[o]
    aq = np.concatenate([x[3] for x in es])[o]; del es
    e = np.zeros(len(k))
    if len(k) > 1:
        e[1:] = ((bp[1:] >= bp[:-1]) * bq[1:] - (bp[1:] <= bp[:-1]) * bq[:-1] - (ap[1:] <= ap[:-1]) * aq[1:] + (ap[1:] >= ap[:-1]) * aq[:-1]).sum(1)
    st['snapshots'] = int(len(k))
    return dict(k=k, imb5=np.concatenate(i5)[o], imb20=np.concatenate(i20)[o], ce=np.cumsum(e), cd=np.cumsum(np.concatenate(d5)[o])), st


def depth_feats(DD, t):
    """BN_DEPTH raw inputs at decision times t (raw sign) from snapshots known by t."""
    k = DD['k']; i = np.searchsorted(k, t, side='right') - 1
    if MUT_BND_NEXT:
        i = np.minimum(i + 1, len(k) - 1)
    ok = i >= 0; ic = np.clip(i, 0, None); out = {}
    out['bnd_imb5'] = np.where(ok, DD['imb5'][ic], np.nan); out['bnd_imb20'] = np.where(ok, DD['imb20'][ic], np.nan)
    out['bnd_age'] = np.where(ok, t - k[ic], np.nan)
    j = np.searchsorted(k, t - 250.0, side='right') - 1
    out['d_bnd_imb5_250'] = np.where(ok & (j >= 0), out['bnd_imb5'] - DD['imb5'][np.clip(j, 0, None)], np.nan)
    j30 = np.searchsorted(k, t - SCALE_MS, side='right') - 1; nn = i - j30
    sc = np.where(ok & (nn > 0), (DD['cd'][ic] - np.where(j30 >= 0, DD['cd'][np.clip(j30, 0, None)], 0.0)) / np.maximum(nn, 1), np.nan)
    for h in (250, 1000):
        jh = np.searchsorted(k, t - h, side='right') - 1
        of = np.where(ok, DD['ce'][ic] - np.where(jh >= 0, DD['ce'][np.clip(jh, 0, None)], 0.0), np.nan)
        out[f'bnd_ofi{h}'] = np.where(sc > 0, of / sc, np.nan)
    return out


def side_inputs(po, dfe, w30, s, ks):
    """The 10 inputs of one window-side's rows (steps ks) from the pair OFI arrays, the depth inputs at those steps and the W30 scale."""
    a = adv_of(s); sc = np.where(w30[ks] > 0, w30[ks], np.nan); out = {}
    for h in (250, 1000):
        out[f'adv_ofi{h}'] = a * po[f'ofi{h}'][ks] / sc; out[f'adv_dd3_{h}'] = a * po[f'dd3_{h}'][ks] / sc
    out['adv_bnd_imb5'] = a * dfe['bnd_imb5']; out['adv_bnd_imb20'] = a * dfe['bnd_imb20']; out['adv_d_bnd_imb5_250'] = a * dfe['d_bnd_imb5_250']
    out['adv_bnd_ofi250'] = a * dfe['bnd_ofi250']; out['adv_bnd_ofi1000'] = a * dfe['bnd_ofi1000']; out['bnd_age'] = dfe['bnd_age']
    return out


def _synth_pair(rng, nf=60, n=40):
    kf = np.sort(rng.uniform(0, 3000, nf)); tg = np.linspace(500, 3000, n); jk = np.searchsorted(kf, tg, side='right') - 1
    W = {}
    for s in (0, 1):
        base = 48 if s == 0 else 52
        P5 = np.stack([base + (-1 if s == 0 else 1) * i + rng.integers(-1, 2, nf) for i in range(5)], 1).astype('int64')
        S5 = rng.uniform(1, 100, (nf, 5))
        W[s] = dict(kf=kf, tg=tg, jk=jk, P5=P5, S5=S5)
    return W


def _ofi_brute(W, k, h):
    kf, tg = W[0]['kf'], W[0]['tg']; hi = int(W[0]['jk'][k]); t = tg[k]; tot = 0.0
    for nidx in range(1, hi + 1):
        if not (kf[nidx] > t - h):
            continue
        b1, b0 = W[0]['P5'][nidx, 0], W[0]['P5'][nidx - 1, 0]; a1, a0 = W[1]['P5'][nidx, 0], W[1]['P5'][nidx - 1, 0]
        if not (b1 >= 0 and a1 > b1 and b0 >= 0 and a0 > b0):
            continue
        qb1, qb0, qa1, qa0 = W[0]['S5'][nidx, 0], W[0]['S5'][nidx - 1, 0], W[1]['S5'][nidx, 0], W[1]['S5'][nidx - 1, 0]
        tot += (b1 >= b0) * qb1 - (b1 <= b0) * qb0 - (a1 <= a0) * qa1 + (a1 >= a0) * qa0
    return tot


def perturb_after(W, rng):
    """f(cut): copies of both sides with every frame after index cut perturbed (prices +-1 tick where a level exists, sizes x U(0.2, 3))."""
    def f(cut):
        W2 = {s: dict(W[s]) for s in (0, 1)}
        for s in (0, 1):
            P5 = np.array(W[s]['P5'], copy=True); S5 = np.array(W[s]['S5'], dtype=float, copy=True)
            if cut + 1 < len(P5):
                P5[cut + 1:] = np.where(P5[cut + 1:] >= 0, P5[cut + 1:] + rng.integers(-1, 2, P5[cut + 1:].shape), P5[cut + 1:])
                S5[cut + 1:] *= rng.uniform(0.2, 3.0, S5[cut + 1:].shape)
            W2[s]['P5'] = P5; W2[s]['S5'] = S5
        return W2
    return f


def falsify_pair(W, po, rng, st, step=FZ_STEP):
    f = perturb_after(W, rng)
    for k in range(0, len(W[0]['tg']), step):
        jk = int(W[0]['jk'][k])
        if jk < 0:
            continue
        p2 = pair_ofi(f(jk)); st['pair_checked'] += 1
        st['pair_differ'] += int(any(not (np.isclose(p2[c][k], po[c][k], equal_nan=True)) for c in po))


def falsify_depth(DD, t, rng, st):
    """Snapshots known after each sampled t perturbed (imbalances, OFI increments, depth) must not change the inputs at t."""
    base = depth_feats(DD, t)
    for x in rng.choice(len(t), min(FD_N, len(t)), replace=False):
        cut = int(np.searchsorted(DD['k'], t[x], side='right'))
        D2 = {c: np.array(v, dtype=float, copy=True) for c, v in DD.items()}
        if cut < len(D2['k']):
            for c in ('imb5', 'imb20'):
                D2[c][cut:] = rng.uniform(-1, 1, len(D2[c]) - cut)
            for c in ('ce', 'cd'):
                D2[c][cut:] = D2[c][cut:] + np.cumsum(rng.uniform(0, 50, len(D2[c]) - cut))
        o2 = depth_feats(D2, t[x:x + 1]); st['depth_checked'] += 1
        st['depth_differ'] += int(any(not np.isclose(o2[c][0], base[c][x], equal_nan=True) for c in base))


def selftest():
    bad = []

    def chk(nm, c):
        if not c:
            bad.append(nm)
    rng = np.random.default_rng(5); W = _synth_pair(rng); po = pair_ofi(W)
    nb = sum(1 for k in range(len(W[0]['tg'])) for h in (250, 1000) if not np.isclose(po[f'ofi{h}'][k], _ofi_brute(W, k, h)) and W[0]['jk'][k] >= 0)
    chk(f'OFI = brute force ({nb} differ)', nb == 0)
    st = dict(pair_checked=0, pair_differ=0); falsify_pair(W, po, rng, st, step=3)
    chk(f'pair falsifier clean {st}', st['pair_checked'] > 5 and st['pair_differ'] == 0)
    df = pd.DataFrame({'E': [0.0, 100.0], 'u': [1, 2], 'bids': ['10.0@2|9.0@1', '10.0@3|9.0@1'], 'asks': ['11.0@1|12.0@1', '11.0@1|12.0@2']})
    df['bids'] = df.bids + '|' + '|'.join(['1.0@0'] * 18); df['asks'] = df.asks + '|' + '|'.join(['99.0@0'] * 18)
    bp, bq, ap, aq = parse_depth(df)
    chk('depth parse', bp.shape == (2, 20) and bq[1, 0] == 3.0 and ap[0, 1] == 12.0 and aq[1, 1] == 2.0)
    kk = np.array([10.0, 110.0, 210.0, 310.0]); DD = dict(k=kk, imb5=np.array([0.1, 0.2, 0.3, 0.4]), imb20=np.array([0.0, 0.1, 0.2, 0.3]),
                                                       ce=np.array([0.0, 1.0, 3.0, 6.0]), cd=np.array([10.0, 20.0, 30.0, 40.0]))
    o = depth_feats(DD, np.array([300.0]))                 # last known snapshot k = 210; 250 ms earlier k = 10; 30 s scale over 3 snapshots
    chk(f'depth lookup {o}', o['bnd_imb5'][0] == 0.3 and o['bnd_age'][0] == 90.0 and np.isclose(o['d_bnd_imb5_250'][0], 0.2) and
        np.isclose(o['bnd_ofi250'][0], 3.0 / 10.0) and np.isnan(depth_feats(DD, np.array([5.0]))['bnd_imb5'][0]))
    sd = dict(depth_checked=0, depth_differ=0); falsify_depth(DD, np.array([150.0, 250.0, 320.0]), rng, sd)
    chk(f'depth falsifier clean {sd}', sd['depth_checked'] == 3 and sd['depth_differ'] == 0)
    return bad


def all_selftests():
    M.all_selftests()
    b = selftest()
    if b:
        sys.exit(f'X5 SELFTEST FAILED: {b}')
    out = {}
    for nm in ('MUT_OFI_NEXT', 'MUT_BND_NEXT'):
        globals()[nm] = 1
        try:
            out[nm] = len(selftest()) > 0
        finally:
            globals()[nm] = 0
    if not all(out.values()):
        sys.exit(f'X5 PLANTED ERROR NOT CAUGHT: {out}')
    print(f'x5 selftests passed; planted errors caught: {out}', flush=True)


def depth_hours(day):
    d0 = pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]}', tz='UTC'); hs = []
    for i in range(-1, 24):
        h = d0 + pd.Timedelta(hours=i)
        if M.K.part_allowed(f'part_{h.strftime("%Y%m%d_%H")}.parquet', day):
            hs.append(h.strftime('%Y%m%d_%H'))
    return hs


def build(day):
    all_selftests()
    if day not in M.DAYS9:
        sys.exit(f'REFUSED: {day} is not a consumed replay day')
    T0 = time.time(); rng = np.random.default_rng(int(day) + 5)
    ck = json.load(open(f'{M.K.IO}/step1/{day}/clock_rl_d2.json')); b_in = float(ck['bn_in']) + float(ck['d_dec'])
    keys = M.day_cols(day, ['m', 'side', 'k']); D = M.K.load_day(day); cut = M.boundary_cut(day)
    DD, dst = depth_arrays(depth_hours(day), b_in)
    if DD is None:
        sys.exit(f'REFUSED: no depth snapshots for {day}')
    M.mem_guard(f'x5 {day} depth')
    idx = {}
    for i, (m, s) in enumerate(zip(keys['m'].tolist(), keys['side'].tolist())):
        idx.setdefault((m, s), []).append(i)
    out = {c: np.full(len(keys['k']), np.nan, np.float32) for c in X5_F}; filled = np.zeros(len(keys['k']), bool)
    st = dict(windows=0, pair_checked=0, pair_differ=0, depth_checked=0, depth_differ=0, depth=dst)
    tall = np.full(len(keys['k']), np.nan)
    for W in M.P1.window_pairs(D):
        if W[0]['ws'] < cut:
            continue
        st['windows'] += 1; po = pair_ofi(W); w30, _ = M.pair_scales(W)
        if st['windows'] % FZ_EVERY == 1:
            falsify_pair(W, po, rng, st)
        m = int(W[0]['m']); tg = np.asarray(W[0]['tg'], float)
        for s in (0, 1):
            ii = idx.get((m, s))
            if not ii:
                continue
            ii = np.asarray(ii); ks = keys['k'][ii]; t = tg[ks]; tall[ii] = t
            o = side_inputs(po, depth_feats(DD, t), w30, s, ks)
            for c in X5_F:
                out[c][ii] = o[c].astype(np.float32)
            filled[ii] = True
        if st['windows'] % 60 == 0:
            M.mem_guard(f'x5 {day}')
    falsify_depth(DD, tall[np.isfinite(tall)], rng, st)
    st['rows'] = int(len(filled)); st['rows_filled'] = int(filled.sum())
    if st['pair_differ'] or st['depth_differ'] or not filled.all():
        sys.exit(f'REFUSED (no output for {day}): falsifier differences or rows without inputs {st}')
    R = pd.DataFrame({'m': keys['m'], 'side': keys['side'], 'k': keys['k'], **out})
    R.to_parquet(f'{M.OUTD}/rows_{day}_x5.parquet', index=False)
    st['nan_share'] = {c: float(np.isnan(out[c]).mean()) for c in X5_F}; st['seconds'] = round(time.time() - T0, 1)
    json.dump(st, open(f'{M.OUTD}/rows_{day}_x5.json', 'w'), indent=1, default=str)
    print(f'x5 {day}: rows {st["rows"]}, windows {st["windows"]}, pair falsifier {st["pair_checked"]} / {st["pair_differ"]}, depth falsifier '
          f'{st["depth_checked"]} / {st["depth_differ"]}, snapshots {dst["snapshots"]} (missing hours {dst["missing"]}), NaN {st["nan_share"]}, '
          f'{st["seconds"]} s', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    if cmd == 'selftest':
        all_selftests()
    elif cmd == 'build':
        build(sys.argv[2])
    else:
        sys.exit(f'unknown command {cmd}')
