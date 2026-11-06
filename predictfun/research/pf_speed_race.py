"""Speed race on Predict.fun: avoiding bad fills and getting to the front (declared in STATUS.yml 2026-11-03 23:14Z; user:
"check how do we avoid bad fills like top makers, and how do we become the first/front in the queue"). Descriptive, 8 read
days, nothing selected.

Lag: time since BTCUSDT futures (Binance E clock) crossed X bp from its mid at the event, in the given direction, within the
last 1 s ('none' if it did not move that far), against Predict venue time (two venue clocks; classes under 10 ms are not
resolved). Our reach from a Binance crossing (pf_clock_rl / exec-port measurements): known E + 3.5 ms (1.5 feed + 2 decision),
a cancel lands E + 25.5 ms (+22), a new order is live E + 43.5 ms (+40).
  part 1  spread width in ticks, time share, live window
  part 2  new best prices (the best improves to a price that was not the best in the frame before): lag since Binance moved
          that way (bid up / ask down), first fill at that price while it rests (who, how soon, net 1 s), and the share
          created later than E + 43.5 ms (an order sent on the crossing would have been there first)
  part 3  fills by lag since an adverse Binance move (maker bid: BTC fell; maker ask: BTC rose), per group (top 5 / 6-20 /
          rest by day volume; ours: ladder lad1q arm, single quote S60 + CAP10): volume share and net 1 s; swept-through
          fills (transaction rule; ours: the replay's through flag) split into no trigger / trigger <= 25.5 ms (too fast for
          our cancel) / earlier (our cancel could have landed)
  part 4  the market's reaction: after Binance crosses 1 bp within 250 ms (first crossing, 1 s refractory), the resting size at
          risk (best and 2 ticks behind on the side the move hurts) in later frames: share traded vs share cancelled (lower
          bound: Q0 - Q(tau) - traded) by tau
Statuses: legs with a 1 s stamp only, fills with no frame before / an empty side / inside the spread, events with no book.
Usage: python3 -m predictfun.research.pf_speed_race          (selftest: ... --selftest)
"""
import os, sys
import numpy as np, pandas as pd

LAG_EDGES = np.array([10.0, 25.5, 43.5, 100.0, 250.0, 1000.0 + 1e-9]); LAG_LAB = ['<10 ms', '10-25 ms', '25-44 ms', '44-100 ms', '100-250 ms', '250-1000 ms', 'none']
CANCEL_AT, LIVE_AT = 25.5, 43.5; TAUS = (0.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0)


def since_cross(K, LM, t, d, X):
    """Per time t[n]: ms since the Binance log-mid LM (update times K) crossed X bp toward its value at t[n] in direction d[n]
    (+1 = it rose into t, -1 = it fell), looking back 1 s; NaN if it did not."""
    out = np.full(len(t), np.nan)
    i1 = np.searchsorted(K, t, side='right') - 1; i0 = np.maximum(np.searchsorted(K, t - 1000.0, side='right') - 1, 0)
    thr = X * 1e-4
    for n in range(len(t)):
        b = i1[n]
        if b < 0:
            continue
        a = i0[n]; seg = LM[a:b + 1]
        idx = np.flatnonzero(d[n] * (seg[-1] - seg) >= thr - 1e-12)
        if len(idx):
            out[n] = t[n] - K[a + idx[-1] + 1]
    return out


def lag_class(x):
    lab = np.array(LAG_LAB, dtype=object)[np.minimum(np.searchsorted(LAG_EDGES, np.nan_to_num(x, nan=1e9), side='right'), 6)]
    return np.where(np.isnan(x), 'none', lab)


def bn_events(K, LM, X=1.0, W=250.0, REF=1000.0):
    """First crossings of X bp within W ms (vs the mid in force W ms earlier), 1 s refractory: (times, signs)."""
    j = np.searchsorted(K, K - W, side='right') - 1; ok = j >= 0
    mv = np.where(ok, LM - LM[np.clip(j, 0, None)], 0.0)
    out_t, out_s, last = [], [], -np.inf
    for i in np.flatnonzero(np.abs(mv) >= X * 1e-4 - 1e-12):
        if K[i] - last >= REF:
            out_t.append(K[i]); out_s.append(1 if mv[i] > 0 else -1); last = K[i]
    return np.array(out_t), np.array(out_s, dtype=int)


def selftest():
    K = np.array([0., 100., 200., 300., 400.]); LM = np.log(np.array([100.0, 100.0, 100.006, 100.006, 100.012]))
    # 1) at t 450 the mid (100.012) is 0.6 bp above 100.006 and 1.2 bp above 100.0: crossing 0.5 bp up happened at 400 -> 50 ms;
    #    1.0 bp up: last update >= 1 bp below is 100 (t 100) -> crossing at 200 -> 250 ms
    a = since_cross(K, LM, np.array([450.0]), np.array([1]), 0.5); b = since_cross(K, LM, np.array([450.0]), np.array([1]), 1.0)
    assert abs(a[0] - 50) < 1e-9 and abs(b[0] - 250) < 1e-9, (a, b)
    # 2) downward direction finds nothing in a rising series; 3) a 2 bp threshold is never reached
    assert np.isnan(since_cross(K, LM, np.array([450.0]), np.array([-1]), 0.5)[0]) and np.isnan(since_cross(K, LM, np.array([450.0]), np.array([1]), 2.0)[0])
    # 4) lookback is 1 s: from t 1250 only updates since 250 (in force: 200) count -> 1.2 bp vs 100 at t 0..100 is out of reach
    assert abs(since_cross(K, LM, np.array([1250.0]), np.array([1]), 0.5)[0] - 850) < 1e-9 and np.isnan(since_cross(K, LM, np.array([1250.0]), np.array([1]), 1.0)[0])
    # 5) lag classes: 20 -> 10-25, 30 -> 25-44, NaN -> none, 1000 -> 250-1000
    assert list(lag_class(np.array([20.0, 30.0, np.nan, 1000.0]))) == ['10-25 ms', '25-44 ms', 'none', '250-1000 ms']
    # 6) events: a 1 bp jump at 300 within 250 ms, a second one 500 ms later is refractory
    K2 = np.array([0., 100., 300., 800., 2000.]); LM2 = np.log(np.array([100.0, 100.0, 100.011, 100.022, 99.99]))
    et, es = bn_events(K2, LM2); assert list(et) == [300.0, 2000.0] and list(es) == [1, -1], (et, es)
    print('selftest PASS (6 cases)')


if __name__ == '__main__' and '--selftest' in sys.argv:
    selftest(); sys.exit(0)

from . import pf_io as io

DAYS = os.environ.get('PF_SR_DAYS', '20261023,20261024,20261025,20261026,20261027,20261031,20261101,20261102').split(','); FRESH = ('20261101', '20261102')
S4 = f'{io.OUT}/step4'; OUTD = f"{S4}/speed_race{os.environ.get('PF_SR_SFX', '')}"; os.makedirs(OUTD, exist_ok=True)
COLS = ['mid', 't_ms'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
TK = ['mid', 'exec_s', 'first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
OURS = [('ours: ladder', lambda d: f"{S4}/ladder_{'preds_c3' if d in FRESH else 'preds_o3'}_lad1q/{d}/fills/K3_z5_S60_TH0.5_CAP50_BUD75_RF1000_THIN30_CD500.parquet"),
        ('ours: single quote', lambda d: f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d in FRESH else 'preds_o3'}_lc/{d}/fills/"
                                         "MICRO_G4_G2_G6_G5_G3_G1_M025_XNA_XMIN1_XS0_XV_S60_CAP10__mid_szA1-5-25-0.parquet")]


def load_bn(day):
    ks, ms = [], []
    for h in io.hours_of(day, extra_after=1, extra_before=1):
        p_ = f'{io.BN_DIR}/{h}.csv.gz'
        if not os.path.exists(p_):
            continue
        df = pd.read_csv(p_, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['E', 'bid', 'ask'])
        df = df.apply(pd.to_numeric, errors='coerce').dropna(); df = df[(df.bid > 0) & (df.ask > df.bid)]
        k = df.E.values.astype(np.float64); m = (df.bid.values + df.ask.values) / 2.0; del df
        o = np.argsort(k, kind='stable'); k, m = k[o], m[o]; keep = np.r_[True, m[1:] != m[:-1]]
        ks.append(k[keep]); ms.append(m[keep])
    K = np.concatenate(ks); M = np.concatenate(ms); o = np.argsort(K, kind='stable')
    return K[o], np.log(M[o])


def lag_u(K, LM, t, d, X):
    """since_cross on unique (t, d) pairs."""
    key = np.stack([t, d.astype(float)], 1); u, inv = np.unique(key, axis=0, return_inverse=True)
    return since_cross(K, LM, u[:, 0], u[:, 1].astype(int), X)[inv.ravel()]


SP, NB, FL, EV, st = [], [], [], [], dict(legs_1s_only=0, fills_unusable=0, events_no_book=0, nb_events=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=COLS) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
    bk.sort_values(['mid', 't_ms'], inplace=True, kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws']).set_index('mid')
    lt = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=TK + ['side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 'signer', 't_rel'])
    lt = lt[lt.mid.isin(reg.index)].copy()
    lo = lt.groupby(TK, sort=False).ptick.transform('min').values; hi = lt.groupby(TK, sort=False).ptick.transform('max').values
    lt['thr'] = np.where(lt.side.values == 0, lt.ptick.values > lo, lt.ptick.values < hi)
    lt = lt[(lt.t_rel >= 0) & (lt.t_rel < 240)]
    vol = lt.groupby('signer').shares.sum().sort_values(ascending=False); rk = pd.Series(np.arange(1, len(vol) + 1), index=vol.index)
    r_ = lt.signer.map(rk).fillna(1e9).values; lt['grp'] = np.where(r_ <= 5, 'top 5', np.where(r_ <= 20, '6-20', 'rest'))
    timed = (lt.how.values > 0) | np.isfinite(lt.t_e2.values); st['legs_1s_only'] += int((~timed).sum()); lt = lt[timed].copy()
    lt['ta'] = np.where(lt.how.values > 0, lt.t_ms.values, lt.t_e2.values)
    parts = [lt[['mid', 'side', 'ptick', 'shares', 'ta', 't_rel', 'grp', 'thr']]]
    for g_, f_ in OURS:
        f = pd.read_parquet(f_(d), columns=['mid', 't', 'side', 'pt', 'q', 'through'])
        parts.append(pd.DataFrame(dict(mid=f.mid.values, side=f.side.values, ptick=f.pt.values.astype(int), shares=f.q.values,
                                       ta=reg.loc[f.mid.values].ws.values * 1000.0 + f.t.values * 1000.0, t_rel=f.t.values, grp=g_, thr=f.through.values)))
    U = pd.concat(parts, ignore_index=True); UB = {m: x for m, x in U.groupby('mid', sort=False)}
    MK = {k: x for k, x in lt.groupby(['mid', 'side'], sort=False)}
    K, LM = load_bn(d)
    ev_t, ev_s = bn_events(K, LM)
    for mid, g in bk.groupby('mid', sort=False):
        if mid not in reg.index:
            continue
        ws = float(reg.loc[mid].ws) * 1000.0; tf = g.t_ms.values.astype(np.float64)
        b1 = g.b1_p.values; a1 = g.a1_p.values; vm = (b1 >= 0) & (a1 > b1); tv = tf[vm]; mv = ((b1 + a1) / 2.0)[vm]
        lw = (tf >= ws) & (tf < ws + 240000.0)
        if lw.sum() < 2:
            continue
        # part 1: spread width, time share (frame holds until the next frame)
        dt = np.diff(np.r_[tf, tf[-1]]); spr = np.where(vm, a1 - b1, -1)
        for w_, lab_ in ((60000.0, 'w60'), (240000.0, 'w240')):
            m_ = (tf >= ws) & (tf < ws + w_)
            SP.append(pd.DataFrame(dict(part=lab_, spread=np.clip(spr[m_], -1, 4), dt=dt[m_])).groupby(['part', 'spread']).dt.sum().reset_index())

        def mid_at(t, strict):
            i = np.searchsorted(tv, t, side='left' if strict else 'right') - 1
            return np.where(i >= 0, mv[np.clip(i, 0, None)], np.nan)
        # part 2: new best prices
        for side in (0, 1):
            pre = 'b' if side == 0 else 'a'
            P = g[[f'{pre}{i}_p' for i in range(1, 6)]].values.astype(int); S = g[[f'{pre}{i}_s' for i in range(1, 6)]].values
            best = P[:, 0]
            imp = np.r_[False, (best[1:] >= 0) & (best[:-1] >= 0) & ((best[1:] > best[:-1]) if side == 0 else (best[1:] < best[:-1]))] & lw
            ks_ = np.flatnonzero(imp)
            if not len(ks_):
                continue
            st['nb_events'] += len(ks_)
            pres = np.zeros((len(tf), 101), bool)
            for i in range(5):
                ok_ = (P[:, i] >= 0) & (S[:, i] > 0); pres[np.flatnonzero(ok_), P[ok_, i]] = True
            tc = tf[ks_]; pc = best[ks_]
            lag5 = lag_u(K, LM, tc, np.full(len(tc), 1 if side == 0 else -1), 0.5); lag10 = lag_u(K, LM, tc, np.full(len(tc), 1 if side == 0 else -1), 1.0)
            h = MK.get((mid, side)); ff_t = np.full(len(ks_), np.nan); ff_g = np.full(len(ks_), '', dtype=object); ff_net = np.full(len(ks_), np.nan); ff_q = np.zeros(len(ks_))
            for n, (k, p) in enumerate(zip(ks_, pc)):
                gone = np.flatnonzero(~pres[k + 1:, p]); kend = k + 1 + gone[0] if len(gone) else len(tf)
                tend = tf[kend] if kend < len(tf) else np.inf
                if h is None:
                    continue
                hp = h[(h.ptick.values == p) & (h.ta.values >= tf[k]) & (h.ta.values <= tend)]
                if len(hp):
                    i = int(np.argmin(hp.ta.values)); t1 = hp.ta.values[i]; first = hp[hp.ta.values == t1]
                    ff_t[n] = t1 - tf[k]; ff_g[n] = first.grp.values[np.argmax(first.shares.values)]; ff_q[n] = first.shares.sum()
                    sg = 1.0 if side == 0 else -1.0; m0 = mid_at(np.array([t1]), True)[0]; m1 = mid_at(np.array([t1 + 1000.0]), False)[0]
                    ff_net[n] = sg * (m0 - p) + sg * (m1 - m0)
            NB.append(pd.DataFrame(dict(day=d, side=side, t_rel=(tc - ws) / 1000.0, jump=np.abs(pc - best[ks_ - 1]), lag05=lag5, lag10=lag10,
                                        ff_ms=ff_t, ff_grp=ff_g, ff_q=ff_q, ff_net=ff_net)))
        # part 3: fills by adverse lag
        u = UB.get(mid)
        if u is not None:
            ta = u.ta.values.astype(np.float64); p = u.ptick.values; side = u.side.values; sg = np.where(side == 0, 1.0, -1.0)
            j0 = np.searchsorted(tf, ta, side='left') - 1; ok = j0 >= 0; j0c = np.clip(j0, 0, None)
            best0 = np.where(side == 0, b1[j0c], a1[j0c]); dep = np.where(side == 0, best0 - p, p - best0); ok &= (best0 >= 0) & (dep >= 0)
            m0 = mid_at(ta, True); m1 = mid_at(ta + 1000.0, False); ok &= np.isfinite(m0) & np.isfinite(m1)
            st['fills_unusable'] += int((~ok).sum())
            dd = np.where(side == 0, -1, 1)
            FL.append(pd.DataFrame(dict(day=d, grp=u.grp.values, w60=u.t_rel.values < 60, q=u.shares.values, thr=u.thr.values.astype(bool),
                                        lag05=lag_u(K, LM, ta, dd, 0.5), lag10=lag_u(K, LM, ta, dd, 1.0), net=sg * (m0 - p) + sg * (m1 - m0),
                                        behind=dep > 0))[ok])
        # part 4: the market's reaction to Binance crossings while this market is live
        sel = (ev_t >= ws) & (ev_t < ws + 240000.0)
        for te, sgn in zip(ev_t[sel], ev_s[sel]):
            side = 0 if sgn < 0 else 1; pre = 'b' if side == 0 else 'a'
            i0 = np.searchsorted(tf, te, side='right') - 1
            if i0 < 0:
                st['events_no_book'] += 1; continue
            P = g[[f'{pre}{i}_p' for i in range(1, 6)]].values; S = g[[f'{pre}{i}_s' for i in range(1, 6)]].values
            bst = P[i0, 0]
            if bst < 0:
                st['events_no_book'] += 1; continue
            prices = [bst, bst - 1, bst - 2] if side == 0 else [bst, bst + 1, bst + 2]
            q_at = lambda i: float(np.where(np.isin(P[i], prices), S[i], 0.0).sum())
            Q0 = q_at(i0)
            if Q0 <= 0:
                continue
            h = MK.get((mid, side)); rec = dict(day=d, Q0=Q0)
            for tau in TAUS:
                i = np.searchsorted(tf, te + tau, side='right') - 1
                V = float(h.shares.values[np.isin(h.ptick.values, prices) & (h.ta.values > te) & (h.ta.values <= te + tau)].sum()) if h is not None else 0.0
                rec[f'Q_{tau:g}'] = q_at(i); rec[f'V_{tau:g}'] = V
            EV.append(rec)
    del bk, lt, U, UB, MK, K, LM
    io.mem_check(f'day {d}')

SPd = pd.concat(SP).groupby(['part', 'spread']).dt.sum(); NBd = pd.concat(NB, ignore_index=True); FLd = pd.concat(FL, ignore_index=True); EVd = pd.DataFrame(EV)
NBd.to_parquet(f'{OUTD}/new_best.parquet', index=False); FLd.to_parquet(f'{OUTD}/fills.parquet', index=False); EVd.to_parquet(f'{OUTD}/bn_events.parquet', index=False)
print(f'statuses {st}')
print('part 1: spread width, share of live time (ticks; -1 = one side empty / crossed)')
for part in ('w60', 'w240'):
    x = SPd.loc[part]; print(f"  {part}: " + '  '.join(f"{int(k)}: {v / x.sum():.1%}" for k, v in x.items()))
print(f'\npart 2: new best prices in the live window (0-240 s): {len(NBd)} events; lag since Binance moved that way')
rep = {}
for X, col in ((0.5, 'lag05'), (1.0, 'lag10')):
    c = pd.Series(lag_class(NBd[col].values)).value_counts(normalize=True).reindex(LAG_LAB).fillna(0)
    bd = NBd[NBd[col].notna()]; later = float((bd[col] >= LIVE_AT).mean()) if len(bd) else np.nan
    print(f"  X {X} bp: " + '  '.join(f"{k} {v:.1%}" for k, v in c.items()) + f" | Binance-driven created later than E + 43.5 ms: {later:.1%}")
    rep[f'lag_X{X}'] = dict(classes=c.round(4).to_dict(), later_than_our_live=later)
f = NBd[NBd.ff_grp != '']
print(f"  first fill while the new best rests: {len(f) / len(NBd):.1%} of events; time to it: " + '  '.join(f"<{x} ms {(f.ff_ms < x).mean():.0%}" for x in (50, 100, 250, 1000)))
for cls in (['<10 ms', '10-25 ms', '25-44 ms'], ['44-100 ms', '100-250 ms', '250-1000 ms'], ['none']):
    s_ = f[pd.Series(lag_class(f.lag05.values), index=f.index).isin(cls)]
    if len(s_):
        print(f"  created {'/'.join(cls):34s}: first fill by top 5 {np.average(s_.ff_grp == 'top 5', weights=s_.ff_q):.0%}, 6-20 {np.average(s_.ff_grp == '6-20', weights=s_.ff_q):.0%}, "
              f"rest {np.average(s_.ff_grp == 'rest', weights=s_.ff_q):.0%}; net 1 s of the first fill {np.average(s_.ff_net.fillna(0), weights=s_.ff_q):+.2f} c/sh; median time to it {s_.ff_ms.median():.0f} ms")
print('\npart 3: fills by lag since an adverse Binance move of 0.5 bp (first 60 s); share of the group\'s volume (net 1 s c/sh)')
GR = ['top 5', '6-20', 'rest', 'ours: ladder', 'ours: single quote']; out3 = {}
for gname in GR:
    x = FLd[(FLd.grp == gname) & FLd.w60]
    if not len(x):
        continue
    cl = pd.Series(lag_class(x.lag05.values), index=x.index); v = x.q.sum(); row = {}
    cells = []
    for lab in LAG_LAB:
        m = cl == lab; row[lab] = dict(share=float(x.q[m].sum() / v), net=float(np.average(x.net[m], weights=x.q[m])) if m.any() else np.nan)
        cells.append(f"{lab} {row[lab]['share']:5.1%} ({row[lab]['net']:+.2f})")
    sw = x[x.thr]; cls_sw = pd.Series(lag_class(sw.lag05.values), index=sw.index); vs = sw.q.sum()
    fast = cls_sw.isin(['<10 ms', '10-25 ms']); none = cls_sw == 'none'; slow = ~fast & ~none
    row['swept'] = dict(share=float(vs / v), none=float(sw.q[none].sum() / vs), within_cancel=float(sw.q[fast].sum() / vs), after_cancel=float(sw.q[slow].sum() / vs),
                        net_none=float(np.average(sw.net[none], weights=sw.q[none])) if none.any() else np.nan,
                        net_within=float(np.average(sw.net[fast], weights=sw.q[fast])) if fast.any() else np.nan,
                        net_after=float(np.average(sw.net[slow], weights=sw.q[slow])) if slow.any() else np.nan)
    out3[gname] = row
    print(f"  {gname:20s} " + ' | '.join(cells))
    r = row['swept']
    print(f"  {'':20s} swept {r['share']:.0%} of volume: no trigger {r['none']:.0%} ({r['net_none']:+.2f}), trigger <= 25.5 ms {r['within_cancel']:.0%} ({r['net_within']:+.2f}), "
          f"earlier {r['after_cancel']:.0%} ({r['net_after']:+.2f})")
print(f'\npart 4: the market\'s reaction to {len(EVd)} Binance crossings of 1 bp within 250 ms (live markets): size at risk Q0 (best + 2 ticks, hurt side)')
q0 = EVd.Q0.sum(); out4 = {}
for tau in TAUS[1:]:
    V = EVd[f'V_{tau:g}'].sum() / q0; C = np.maximum(EVd.Q0 - EVd[f'Q_{tau:g}'] - EVd[f'V_{tau:g}'], 0).sum() / q0
    out4[f'{tau:g}'] = dict(traded=float(V), cancelled_lb=float(C))
    print(f"  +{tau:5.0f} ms: traded {V:5.1%} of Q0, cancelled >= {C:5.1%}")
io.write_json(f'{OUTD}/report.json', dict(statuses=st, spread={f'{k[0]}|{k[1]}': float(v) for k, v in SPd.items()},
              new_best=rep, fills=out3, reaction=out4))
