"""B12e v2 (2026-11-11). B12e (pf_b12e_smart_takes.py 04422a15) read its Binance features from the research grid
(step5 rows_rl_d2.bn_mid), which a post-run check found coarse (90% of 1 s intervals unchanged). USER "Can you still check
Binance data, whether we have that data": the HF collectors on this box recorded Binance BTCUSDT best bid / ask for
2026-11-08 (data/mm_hf/raw/bookTicker/BTCUSDT/20261108_HH.csv.gz: recv_ns, E ms, T ms, update id, bid, bid qty, ask,
ask qty; sub-second-reliable era). v2 recomputes B12e's Binance features on that source with B12e's readings and adds a
resolution check that the old grid would fail. Declared before this script's first run. Read-only, descriptive.

Events: exactly B12e's (same build, address join, SMART / OTHER, RANDOM times from the same seed); K-checks: the counts
and B12e's Predict feature means are reproduced. Binance mid = (bid + ask) / 2 at the latest update with recv time <= t
(this box's clock, the clock of the research Predict times if both collectors run here: a stated assumption), missing if
older than 1 s. Signed returns in bps as B12e (+ = the taker's favour), over (t - 1050, t - 50], (t - 300, t - 50],
(t - 50, t + 500], (t - 50, t + 2000].

Readings DECLARED BEFORE THE FIRST RUN:
 R76 resolution  BN-RESOLUTION-OK if at RANDOM times the share of exactly-zero changes over (t - 1050, t - 50] is <= 0.5;
                 else BN-RESOLUTION-WEAK, and R70v2 / R74v2 read n/a.
 R70v2 lead      B12e's R70 rule on this source: BINANCE-LEAD if mean (before) at SMART minus RANDOM >= +0.3 bps and above
                 its null p99 (500 permutations within window, seed 20261111); else NO-BINANCE-LEAD.
 R74v2 after     B12e's R74 rule: BINANCE-FOLLOWS if mean (t - 50, t + 2000] at SMART minus RANDOM >= +0.3 bps and above
                 its null p99; else NO-BINANCE-FOLLOW.
 R71v2 lag       B12e's R71 rule under R70v2 (PREDICT-STALE if the Predict move before at SMART minus RANDOM <= +0.1 c).
Descriptive: the share of events with a Binance move of at least 1 bps in the taker's favour over the second before;
the same for the old grid, side by side. Our own at-best fills (and the top 5's and O's) split by the book on their side
in the frame before: thin side (size <= half the other side's), balanced, heavy side (>= twice), with net value at 1 s.
Controls: B12e's controls (which include B12b's and B12's); the bookTicker lookup on planted rows (latest update, stale
after 1 s); K-checks against b12e.json; every cohort hour has its file (status otherwise).
Limits: one consumed Sunday; the clocks of the two collectors are assumed to be the same box clock; recv time trails
Binance's event time by our network latency (~77 ms in the first rows); three addresses.
Usage: predictfun/research/run_light.sh b12e2_binance_raw -- python3 -m predictfun.research.pf_b12e2_binance_raw <out_dir>
       python3 -m predictfun.research.pf_b12e2_binance_raw --selftest
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from . import pf_b12_makers as B
from . import pf_b12b_reason as BB
from . import pf_b12c2_binance as C2
from . import pf_b12d_bursts as DB
from . import pf_b12e_smart_takes as E1

BT = '/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/BTCUSDT'
B12E = f'{BB.LV}/b12e_20261111/b12e.json'
STALE = 1000.0
FEAT = E1.FEAT_BN


def bt_look(tms, mid, t):
    j = np.searchsorted(tms, t, side='right') - 1; jj = np.clip(j, 0, None)
    return np.where((j >= 0) & (t - tms[jj] <= STALE), mid[jj], np.nan)


def label_r76(z):
    return 'n/a' if not BB.fin(z) else ('BN-RESOLUTION-OK' if z <= 0.5 else 'BN-RESOLUTION-WEAK')


def controls():
    rep = {}; ok = True

    def chk(name, cond):
        nonlocal ok
        rep[name] = bool(cond); ok = ok and bool(cond)
    oke, repe = E1.controls(); chk('B12e_controls', oke)
    tms = np.array([0.0, 10.0, 2000.0]); mid = np.array([100.0, 101.0, 102.0])
    x = bt_look(tms, mid, np.array([-1.0, 5.0, 1500.0, 2500.0]))
    chk('G1_lookup', np.isnan(x[0]) and x[1] == 100.0 and np.isnan(x[2]) and x[3] == 102.0)
    chk('G2_rule', label_r76(0.9) == 'BN-RESOLUTION-WEAK' and label_r76(0.2) == 'BN-RESOLUTION-OK' and label_r76(float('nan')) == 'n/a')
    rep['B12e_controls_detail'] = repe
    return ok, rep


def load_bt(W0):
    """Best bid / ask mids for the cohort windows' first minutes (+/- 3 s), from the 11-08 hour files."""
    lo = {m: ws - 3000.0 for m, ws in W0.items()}; hi = {m: ws + 63000.0 for m, ws in W0.items()}
    spans = sorted((lo[m], hi[m], m) for m in W0); out = {m: [] for m in W0}; stat = dict(files=0, missing_hours=[])
    hours = sorted({int(((ws / 1000.0) % 86400) // 3600) for ws in W0.values()} | {int((((ws + 63000.0) / 1000.0) % 86400) // 3600) for ws in W0.values()})
    for h in hours:
        fn = f'{BT}/20261108_{h:02d}.csv.gz'
        if not os.path.exists(fn):
            stat['missing_hours'].append(h); continue
        d = pd.read_csv(fn, header=None, usecols=[0, 4, 6], names=['recv_ns', 'bid', 'ask'], dtype={'recv_ns': 'int64', 'bid': 'float64', 'ask': 'float64'})
        stat['files'] += 1; t = d.recv_ns.values / 1e6; mid = (d.bid.values + d.ask.values) / 2.0
        for a, b, m in spans:
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
            if i1 > i0:
                out[m].append((t[i0:i1], mid[i0:i1]))
        del d
    res = {}
    for m, parts in out.items():
        if parts:
            tt = np.concatenate([p[0] for p in parts]); mm = np.concatenate([p[1] for p in parts]); o = np.argsort(tt, kind='stable'); res[m] = (tt[o], mm[o])
    stat['windows_with_data'] = len(res)
    return res, stat


def main():
    if '--selftest' in sys.argv:
        ok, rep = controls(); print(json.dumps({k: v for k, v in rep.items() if not k.endswith('detail')}, indent=1)); print('selftest', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    ok, ctl = controls()
    if not ok:
        sys.exit(f'CONTROL FAILED -> no table: {ctl}')
    b12e = json.load(open(B12E))
    X = BB.build(); R = X['R']; W0 = X['W0']; OWN = X['OWN']
    k0 = BB.reconcile(R, X['days'])
    if not k0['ok']:
        sys.exit(f'CONTROL FAILED (K0) -> no table: {k0}')
    # events exactly as B12e
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
    kc = dict(counts=counts, counts_ok=counts == b12e['counts'])
    sgT = E1.tsign(EV.side.values)
    # Predict move before (to reproduce B12e) and the two Binance sources
    FR = DB.frames(W0); BTd, bst = load_bt(W0)
    BNd = pd.read_parquet(C2.BN, columns=['mid', 'tau', 'bn_mid']); BNd = BNd[BNd.mid.isin(W0)].sort_values(['mid', 'tau'], kind='stable')
    BG = {m: (g.tau.values.astype(float), g.bn_mid.values.astype(float)) for m, g in BNd.groupby('mid')}; del BNd
    new = {k: np.full(len(EV), np.nan) for k in FEAT}; old = np.full(len(EV), np.nan); pfb = np.full(len(EV), np.nan)
    for m, ii in EV.groupby('mid', sort=False).indices.items():
        t = EV.anc.values[ii]
        if m in BTd:
            tms, mid = BTd[m]
            for k, (a, b) in FEAT.items():
                new[k][ii] = C2.sret(np.ones(len(ii)), bt_look(tms, mid, t + a), bt_look(tms, mid, t + b))
        tau, bn = BG[m]; old[ii] = C2.sret(np.ones(len(ii)), C2.bnlook(tau, bn, t - 1050.0), C2.bnlook(tau, bn, t - 50.0))
        tf, S = FR[m]; bb = S[0][0][:, 0]; ba = S[1][0][:, 0]; mff = pd.Series(np.where((bb >= 0) & (ba > bb), (ba + bb) / 2.0, np.nan)).ffill().values
        j1 = np.searchsorted(tf, t - 50.0, side='right') - 1; j2 = np.searchsorted(tf, t - 1050.0, side='right') - 1
        pfb[ii] = np.where((j1 >= 0) & (j2 >= 0), mff[np.clip(j1, 0, None)] - mff[np.clip(j2, 0, None)], np.nan)
    for k in FEAT:
        EV[k] = new[k] * sgT
    EV['old_before'] = old * sgT; EV['pf_before'] = pfb * sgT
    kc['pf_before_mean_SMART'] = float(np.nanmean(EV.pf_before[EV.kind == 'SMART'])); kc['pf_before_ok'] = abs(kc['pf_before_mean_SMART'] - b12e['table']['pf_before']['SMART']['mean']) < 1e-9
    if not (kc['counts_ok'] and kc['pf_before_ok']):
        sys.exit(f'CONTROL FAILED (K-check against b12e.json) -> no table: {kc}')
    rnd = EV[EV.kind == 'RANDOM']; zero_new = float(np.mean(rnd.bn_before.dropna() == 0)); zero_old = float(np.mean(rnd.old_before.dropna() == 0))
    SR = EV[EV.kind.isin(['SMART', 'RANDOM'])]; lab = SR.kind.values == 'SMART'; blk = pd.factorize(SR.mid)[0]
    table = {}
    for k in list(FEAT) + ['old_before']:
        row = {kd: dict(n=int(np.isfinite(EV[k][EV.kind == kd]).sum()), mean=float(np.nanmean(EV[k][EV.kind == kd])), median=float(np.nanmedian(EV[k][EV.kind == kd])),
                        share_pos=float(np.nanmean(EV[k][EV.kind == kd] > 0)), share_ge_1bp=float(np.nanmean(EV[k][EV.kind == kd] >= 1.0)),
                        share_le_m1bp=float(np.nanmean(EV[k][EV.kind == kd] <= -1.0))) for kd in ('SMART', 'OTHER', 'RANDOM')}
        d, p99, p = E1.diff_null(SR[k].values.astype(float), lab, blk, E1.N_NULL, E1.SEED)
        row['SMART_minus_RANDOM'] = dict(diff=d, null_p99=p99, p_one_sided=p); table[k] = row
    r76 = label_r76(zero_new)
    if r76 == 'BN-RESOLUTION-OK':
        r70 = E1.label_r70(table['bn_before']['SMART_minus_RANDOM']['diff'], table['bn_before']['SMART_minus_RANDOM']['null_p99'])
        r74 = E1.label_r74(table['bn_after_2000']['SMART_minus_RANDOM']['diff'], table['bn_after_2000']['SMART_minus_RANDOM']['null_p99'])
        r71 = E1.label_r71(r70, b12e['table']['pf_before']['SMART_minus_RANDOM']['diff'])
    else:
        r70 = r74 = r71 = 'n/a (resolution)'
    # our own (and others') at-best fills by the book on their side
    V = R[R.val_ok & (R.status == 'ok') & (R.depth == 0)].copy(); qo = np.full(len(V), np.nan)
    for (m, side), ii in V.groupby(['mid', 'side'], sort=False).indices.items():
        tf, S = FR[m]; j0 = np.searchsorted(tf, V.anc.values[ii], side='left') - 1; jj = np.clip(j0, 0, None)
        qo[ii] = np.where(j0 >= 0, S[1 - side][1][jj, 0], np.nan)
    V['q_other'] = qo; r_ = V.Q.values / V.q_other.values
    V['book'] = np.where(~np.isfinite(r_), 'n/a', np.where(r_ <= 0.5, 'thin side', np.where(r_ >= 2.0, 'heavy side', 'balanced')))
    book = {}
    for g in BB.GR:
        x = BB.sel(V, g); s_ = x.shares.values; tot = s_.sum(); book[g] = {}
        for c in ('thin side', 'balanced', 'heavy side'):
            mk = x.book.values == c
            book[g][c] = dict(legs=int(mk.sum()), share=float(s_[mk].sum() / tot) if tot else np.nan, net1=B.wmean(s_[mk], x.v1.values[mk]))
    res = dict(as_of_note='point estimates, one consumed partial day (220 windows of 2026-11-08, 0.764 window-days)', controls=ctl, K0=k0['ok'], Kcheck=kc,
               bookticker=bst, zero_share_random_new=zero_new, zero_share_random_old=zero_old, table=table, our_fills_by_book=book,
               readings=dict(R76=r76, R70v2=r70, R71v2=r71, R74v2=r74))
    json.dump(res, open(f'{out}/b12e2.json', 'w'), indent=1, default=float)
    f = lambda v: 'n/a' if not BB.fin(v) else f'{v:+.3f}'
    Lt = [f"B12e v2: Binance at the top makers' takes, from the raw bookTicker. {res['as_of_note']}",
          f"Controls: {json.dumps({k: v for k, v in ctl.items() if not k.endswith('detail')})}; K-check {kc}; bookTicker {bst}",
          f"Resolution at RANDOM times: share of zero changes over the second before: raw bookTicker {zero_new * 100:.1f}%, old grid {zero_old * 100:.1f}%", '',
          'feature (bps, + = taker\'s favour) | SMART mean / median / >= +1 bp / <= -1 bp | OTHER mean / >= +1 bp | RANDOM mean / >= +1 bp / <= -1 bp | SMART - RANDOM (null p99, p)']
    for k, r in table.items():
        s, o, q = r['SMART'], r['OTHER'], r['RANDOM']
        Lt.append(f"{k} | {f(s['mean'])} / {f(s['median'])} / {s['share_ge_1bp'] * 100:.0f}% / {s['share_le_m1bp'] * 100:.0f}% | {f(o['mean'])} / {o['share_ge_1bp'] * 100:.0f}% | "
                  f"{f(q['mean'])} / {q['share_ge_1bp'] * 100:.0f}% / {q['share_le_m1bp'] * 100:.0f}% | {f(r['SMART_minus_RANDOM']['diff'])} ({f(r['SMART_minus_RANDOM']['null_p99'])}, {r['SMART_minus_RANDOM']['p_one_sided']:.3f})")
    Lt += ['', 'At-best fills by the book on the maker\'s side (frame before; thin = size <= half the other side, heavy = >= twice): share of shares / net 1 s / legs']
    for g, x in book.items():
        Lt.append(f"{g}: " + '; '.join(f"{c} {v['share'] * 100:.0f}% {f(v['net1'])} ({v['legs']})" for c, v in x.items()))
    Lt += ['', 'Readings: ' + json.dumps(res['readings'])]
    open(f'{out}/b12e2.txt', 'w').write('\n'.join(Lt) + '\n'); print('\n'.join(Lt))


if __name__ == '__main__':
    main()
