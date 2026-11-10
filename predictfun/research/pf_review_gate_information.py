"""Review diagnostic D-B (2026-11-10, strategy review). Descriptive only.

Question: is the frozen per-side guard (v_side vs the minute-0 PM90 line) informative about which real touch trades are good for
the maker BEFORE the trade, independent of where our own order was? D-A (queue misses by order age) is selected on our order
surviving the trade, i.e. on the level not being cleared, so it cannot answer this.

Population: every timed touch trade in 0-60 s of the 220 same-window cohort windows of 2026-11-08 (cut 23:00Z, consumed day; the
guard models and lines were fit on 10-23..11-02, so this day is out of sample for the guard, but it is ONE partial day):
touch_trade_reasons.parquet (trade = maker fills at the pre-trade best, time W = live receipt of the WS pending message).
Guard value: research/step5/20261108/preds_t60k.parquet (50 ms grid), maker side's v at W - lag, fresh <= 100 ms.
Outcome: maker markout from the trade price (c/share, net of spread) at 1 s / 5 s / 60 s / settlement, share-weighted.
No raw capture, no protected day, no interval, no inferential claim (G < 5). Exclusions are statuses and are counted.

Readings DECLARED BEFORE THE TABLES WERE SEEN  (D(lag) = mo1(gate on) - mo1(gate off) with the gate read at W - lag):
  R1 information : PRE-TRADE if D(250) >= +0.30 and D(500) >= +0.20 ; REACTIVE if D(50) >= +0.30 and D(250) < +0.15 ;
                   UNINFORMATIVE if D(50) < +0.15 ; else MIXED.
  R2 buffer zone : trades while the gate has been off for <= 100 ms (the state a pull buffer keeps an order in):
                   LIKE-ON if mo1 >= mo1(on) - 0.20 ; LIKE-PULLED if mo1 <= mo1(off > 2 s) + 0.10 ; else BETWEEN.
  R3 persistence : share of gate-on touch volume that trades after the gate has been continuously on >= 500 ms (50 ms grid,
                   strict): PERSISTENT if >= 50% ; FLICKER if <= 20% ; else BETWEEN.
Caveat fixed in advance: these are pro-rata values of the flow (what the makers who were filled got). A maker who joins at the
back of the queue does not receive the pro-rata flow.

Self-test (falsifier): (a) the recomputed all-status first-minute mo1 must equal the direct share-weighted mean; (b) the gate
state at lag 50 must agree with the existing status for the two statuses that are pure functions of it ('not quoting: prediction:
guard value below the pull line' -> off) on >= 99% of those trades; (c) with the guard column replaced by its sign-flipped
copy, agreement in (b) must collapse (< 60%).
Usage: predictfun/research/run_light.sh review_gate_information -- python3 predictfun/research/pf_review_gate_information.py <out.json>
"""
import json
import sys

import numpy as np
import pandas as pd

RP = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108'
OUT = f'{RP}/live_vs_replay'
LAGS = (50, 250, 500, 1000)


def wm(d, col):
    ok = d[col].notna()
    return float((d.loc[ok, 'shares'] * d.loc[ok, col]).sum() / d.loc[ok, 'shares'].sum()) if ok.any() else float('nan')


def row(name, d, tot):
    held = d.shares[d.mo1 >= 0.5 - 1e-9].sum() / d.shares[d.mo1.notna()].sum() if d.mo1.notna().any() else float('nan')
    return dict(bucket=name, trades=len(d), shares=round(float(d.shares.sum()), 0), share_pct=round(100 * d.shares.sum() / tot, 2), mo1=round(wm(d, 'mo1'), 3),
                mo5=round(wm(d, 'mo5'), 3), mo60=round(wm(d, 'mo60'), 3), settle=round(wm(d, 'settle'), 2), held_1s_pct=round(100 * float(held), 1), windows=int(d.mid.nunique()))


def show(title, rows):
    print(f'\n== {title} =='); print(pd.DataFrame(rows).to_string(index=False))


def main():
    sc = json.load(open(f'{RP}/scope.json')); coh = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}
    LINE = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']['90'][0]
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet')
    n_all = len(R); n_unt = int((R.status == 'untimed').sum())
    V = R[(R.status != 'untimed')].copy(); V['mid'] = V.mid.astype('int64')
    T = V[V.t_in < 60].copy(); T['W'] = T.mid.map(coh) + T.t_in * 1000.0
    print(f'rows {n_all}; untimed {n_unt}; timed {len(V)}; timed 0-60 s {len(T)} trades, {T.shares.sum():.0f} shares, {T.mid.nunique()} windows; line {LINE}')
    P = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    P = P[P.mid.isin(set(T.mid))].sort_values(['mid', 'tau'])
    G = {m: (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in P.groupby('mid')}
    del P

    def gate(flip=False):
        """Per trade and lag: guard value of the maker side at W - lag (NaN = no fresh value); plus on-run and off-run lengths at lag 50."""
        v = {lag: np.full(len(T), np.nan) for lag in LAGS}; on_ms = np.full(len(T), np.nan); off_ms = np.full(len(T), np.nan)
        on_ms_g100 = np.full(len(T), np.nan)
        idx = {m: np.flatnonzero(T.mid.values == m) for m in G}
        W = T.W.values; up = T.up.values
        for m, ii in idx.items():
            tau, vb, va = G[m]
            if flip:
                vb, va = -vb, -va
            for side_up in (True, False):
                k = ii[up[ii] == side_up]
                if not len(k):
                    continue
                vs = vb if side_up else va
                for lag in LAGS:
                    tm = W[k] - lag; j = np.searchsorted(tau, tm, side='right') - 1
                    ok = (j >= 0) & (tm - tau[np.clip(j, 0, None)] <= 100.0)
                    x = np.where(ok, vs[np.clip(j, 0, None)], np.nan); v[lag][k] = x
                # run lengths on the grid at lag 50: consecutive grid points (step <= 60 ms) with the same on / off state
                on = vs > LINE; fin = np.isfinite(vs)
                step_ok = np.r_[False, np.diff(tau) <= 60.0]
                same = np.r_[False, (on[1:] == on[:-1]) & fin[1:] & fin[:-1]] & step_ok
                start = np.arange(len(tau)); brk = np.flatnonzero(~same); start = brk[np.searchsorted(brk, start, side='right') - 1]
                # gap-tolerant on-run: an off spell of <= 100 ms (<= 2 grid points) between on points does not break it
                t_last_off_long = np.full(len(tau), -np.inf); cur = -np.inf; off_run = 0
                for q in range(len(tau)):
                    if (not fin[q]) or (q and tau[q] - tau[q - 1] > 60.0):
                        cur = tau[q]; off_run = 0
                    elif not on[q]:
                        off_run += 1
                        if off_run > 2:
                            cur = tau[q]
                    else:
                        off_run = 0
                    t_last_off_long[q] = cur
                tm = W[k] - 50.0; j = np.searchsorted(tau, tm, side='right') - 1
                ok = (j >= 0) & (tm - tau[np.clip(j, 0, None)] <= 100.0); jj = np.clip(j, 0, None)
                dur = tm - tau[start[jj]]
                on_ms[k] = np.where(ok & fin[jj] & on[jj], dur, np.nan)
                off_ms[k] = np.where(ok & fin[jj] & ~on[jj], dur, np.nan)
                on_ms_g100[k] = np.where(ok & fin[jj] & on[jj], tm - t_last_off_long[jj], np.nan)
        return v, on_ms, off_ms, on_ms_g100

    v, on_ms, off_ms, on_g = gate()
    for lag in LAGS:
        T[f'v{lag}'] = v[lag]
    T['on_ms'] = on_ms; T['off_ms'] = off_ms; T['on_g'] = on_g
    tot = T.shares.sum()
    # ---- self-test
    direct = float((T.shares * T.mo1).sum() / T.shares[T.mo1.notna()].sum())
    a_ok = abs(direct - wm(T, 'mo1')) < 1e-12
    pulled = T[T.status == 'not quoting: prediction: guard value below the pull line']
    agree = float(((pulled.v50 <= LINE)).mean()) if len(pulled) else float('nan')
    vf, _, _, _ = gate(flip=True); agree_flip = float((pd.Series(vf[50], index=T.index)[pulled.index] <= LINE).mean()) if len(pulled) else float('nan')
    print(f'selftest: (a) {a_ok}; (b) status-pulled trades with gate off at lag 50: {100 * agree:.2f}% of {len(pulled)}; (c) with flipped guard: {100 * agree_flip:.2f}%')
    if not (a_ok and agree >= 0.99 and agree_flip < 0.60):
        print('SELFTEST FAILED -> no table'); return 1
    # ---- first-minute by existing status
    show('0-60 s by existing status (W - 50 ms), top 12 by shares', [row(k, d, tot) for k, d in sorted(T.groupby('status'), key=lambda kv: -kv[1].shares.sum())][:12])
    # ---- R1: gate on / off at each lag
    D = {}; rows = []
    for lag in LAGS:
        x = T[f'v{lag}']; on = T[x > LINE]; off = T[x <= LINE]; na = T[x.isna()]
        rows += [row(f'lag {lag}: on', on, tot), row(f'lag {lag}: off', off, tot), row(f'lag {lag}: no fresh value', na, tot)]
        D[lag] = wm(on, 'mo1') - wm(off, 'mo1')
    show('gate state of the maker side at W - lag (all touch trades 0-60 s)', rows)
    print('D(lag) = mo1(on) - mo1(off):', {k: round(v_, 3) for k, v_ in D.items()})
    r1 = 'PRE-TRADE' if (D[250] >= 0.30 and D[500] >= 0.20) else ('REACTIVE' if (D[50] >= 0.30 and D[250] < 0.15) else ('UNINFORMATIVE' if D[50] < 0.15 else 'MIXED'))
    print('R1 reading =', r1)
    # ---- dose response at lag 250
    d = T.v250 - LINE; bins = [-np.inf, -0.30, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, np.inf]
    lab = ['<-0.30', '-0.30..-0.15', '-0.15..-0.10', '-0.10..-0.05', '-0.05..-0.02', '-0.02..0', '0..0.02', '0.02..0.05', '0.05..0.10', '>0.10']
    T['dbin'] = pd.cut(d, bins, labels=lab)
    dr = [row(str(k), g, tot) for k, g in T.groupby('dbin', observed=True)]
    show('dose response: v - line at W - 250 ms (c)', dr)
    m1 = [r['mo1'] for r in dr]; rho = float(pd.Series(m1).corr(pd.Series(range(len(m1))), method='spearman'))
    print(f'Spearman(bucket order, mo1) = {rho:+.3f} over {len(m1)} buckets')
    # ---- R2: buffer zone = time since the gate was last on (state at lag 50)
    onT = T[T.v50 > LINE]; offT = T[T.v50 <= LINE]
    zb = [(0, 100, 'off <= 100 ms'), (100, 500, 'off 100-500 ms'), (500, 2000, 'off 0.5-2 s'), (2000, np.inf, 'off > 2 s')]
    rows = [row('on', onT, tot)] + [row(n, offT[(offT.off_ms > a) & (offT.off_ms <= b)] if a else offT[offT.off_ms <= b], tot) for a, b, n in zb]
    show('time since the gate was last on (strict 50 ms grid, state at W - 50 ms)', rows)
    z = offT[offT.off_ms <= 100]; far = offT[offT.off_ms > 2000]
    mz, mon, mfar = wm(z, 'mo1'), wm(onT, 'mo1'), wm(far, 'mo1')
    r2 = 'LIKE-ON' if mz >= mon - 0.20 else ('LIKE-PULLED' if mz <= mfar + 0.10 else 'BETWEEN')
    print(f'R2: mo1 zone {mz:+.3f} | on {mon:+.3f} | off > 2 s {mfar:+.3f} -> reading = {r2}')
    # the buffer's own band: off <= 100 ms AND v within 0.01 c below the line
    for band in (0.005, 0.01, 0.02, 0.04):
        b = z[z.v50 > LINE - band]
        print(f'   band {band}: trades {len(b)}, shares {b.shares.sum():.0f} ({100 * b.shares.sum() / tot:.2f}% of touch volume), mo1 {wm(b, "mo1"):+.3f}, mo5 {wm(b, "mo5"):+.3f}, held 1 s {100 * b.shares[b.mo1 >= 0.5 - 1e-9].sum() / max(b.shares.sum(), 1e-9):.1f}%')
    # ---- R3: persistence of the on state before the trade
    pb = [(0, 100, 'on < 100 ms'), (100, 250, 'on 100-250 ms'), (250, 500, 'on 250-500 ms'), (500, 1000, 'on 0.5-1 s'), (1000, 2000, 'on 1-2 s'), (2000, np.inf, 'on >= 2 s')]
    ton = onT.shares.sum()
    show('gate-on touch volume by how long the gate had been on (strict)', [row(n, onT[(onT.on_ms >= a) & (onT.on_ms < b)], ton) for a, b, n in pb])
    show('same, gaps <= 100 ms bridged', [row(n, onT[(onT.on_g >= a) & (onT.on_g < b)], ton) for a, b, n in pb])
    s500 = float(onT.shares[onT.on_ms >= 500].sum() / ton); s500g = float(onT.shares[onT.on_g >= 500].sum() / ton)
    r3 = 'PERSISTENT' if s500 >= 0.5 else ('FLICKER' if s500 <= 0.2 else 'BETWEEN')
    print(f'R3: share of gate-on touch volume with the gate on >= 500 ms: strict {100 * s500:.1f}% (bridged {100 * s500g:.1f}%) -> reading = {r3}')
    # ---- grid facts: on-time share and spell lengths in 0-60 s
    tsh = []; spells = []
    for m, (tau, vb, va) in G.items():
        t0 = coh[m]; k = (tau >= t0) & (tau < t0 + 60_000)
        for vs in (vb, va):
            x = vs[k]; f = np.isfinite(x); on = (x > LINE) & f
            tsh.append((on.sum(), f.sum()))
            e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0])); spells += list((e[1::2] - e[0::2]) * 50.0)
    tsh = np.array(tsh); sp = np.array(spells)
    print(f'\ngrid 0-60 s: gate-on time share {100 * tsh[:, 0].sum() / tsh[:, 1].sum():.2f}% of side-time; spells {len(sp)}; length ms p50 {np.median(sp):.0f} p75 {np.quantile(sp, .75):.0f} '
          f'p90 {np.quantile(sp, .9):.0f}; share of on-time in spells >= 500 ms {100 * sp[sp >= 500].sum() / sp.sum():.1f}%, >= 1 s {100 * sp[sp >= 1000].sum() / sp.sum():.1f}%, >= 2 s {100 * sp[sp >= 2000].sum() / sp.sum():.1f}%')
    print(f'gate-on share of touch VOLUME at lag 50: {100 * onT.shares.sum() / tot:.2f}% (time share above); touch volume per window 0-60 s: {tot / T.mid.nunique():.0f} sh; gate-on volume per window: {onT.shares.sum() / T.mid.nunique():.0f} sh')
    json.dump(dict(D=D, R1=r1, R2=r2, R3=r3, zone=dict(mo1=mz, on=mon, far=mfar), on_share_volume=float(onT.shares.sum() / tot), s500=s500, s500_bridged=s500g, spearman=rho),
              open(sys.argv[1], 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
