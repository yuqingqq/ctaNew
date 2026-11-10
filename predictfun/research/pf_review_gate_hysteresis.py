"""Review diagnostic D-D (2026-11-10, strategy review). Descriptive only; same consumed cohort as D-B / D-C
(220 windows of 2026-11-08, cut 23:00Z; 0-60 s touch trades; guard out of sample, ONE partial day; no interval, no inference).
Derived cohort outputs only (touch_trade_reasons.parquet, preds_t60k.parquet, minute_lines_t60k.json, scope.json).

Question: v3 quotes a side only while its guard value v is above the minute's pull line (PM90). A two-line rule - ENTER when
v > L_in, STAY until v <= L_out - would rest longer (queue priority) but also sits through flow the gate now avoids. What is
the public touch flow worth, pro rata, while such a state is on, and how long would the state rest?
This values FLOW (every touch trade, share-weighted, markout from the trade price), not our fills: it needs no fill model and
says nothing about the queue position we would reach.

Arms (fixed before the tables were seen; H1 is the only primary):
  H0  L_in = L_out = PM90 line        = v3's gate (self-test reference)
  H1  L_in = PM90 line, L_out = -0.10 c  (about line - 0.15: D-B's dose table turns negative below it. That edge was read
      from THIS day, so H1 is in-sample for its exit line: an illustration to be re-declared and tested on untouched days.)
  H2  L_in = PM90 line, L_out = PM50 line (an existing line, fitted on the 8 read days)
  H3  L_in = PM80 line, L_out = -0.10 c
State per (window, side) on the 50 ms grid: on when v > L_in; kept while v > L_out; off when v <= L_out, when the value is
missing, or across a grid gap > 60 ms. Read for a trade at W - 250 ms (primary) and W - 500 ms, fresh <= 100 ms.

Readings DECLARED BEFORE THE TABLES WERE SEEN (H1, state read at W - 250 ms):
  R6 hold-state flow (state on while v <= L_in):  HOLD-OK if mo1 >= +0.10 c/sh and mo5 >= 0 ; HOLD-ADVERSE if mo1 <= 0 or
     mo5 <= -0.20 ; else NEUTRAL.
  R7 residence: share of state-on touch volume whose state has been on >= 1 s:  RESIDENT if >= 60% ; else NOT.
Self-test (falsifier): H0 must reproduce D-C's lag-250 gate-on row (shares and mo1) exactly; with trade times shuffled
within the window the H1 on-minus-off mo1 contrast must collapse (|D| < 0.15).
Usage: predictfun/research/run_light.sh review_gate_hysteresis -- python3 predictfun/research/pf_review_gate_hysteresis.py <out.json> <D-C out.json>
"""
import json
import sys

import numpy as np
import pandas as pd

RP = '/home/yuqing/pmmm-data/review-exports/live-samewindow-20261108'
OUT = f'{RP}/live_vs_replay'


def wm(d, col):
    ok = d[col].notna()
    return float((d.loc[ok, 'shares'] * d.loc[ok, col]).sum() / d.loc[ok, 'shares'].sum()) if ok.any() else float('nan')


def row(name, d, tot):
    return dict(bucket=name, trades=len(d), shares=round(float(d.shares.sum()), 0), share_pct=round(100 * d.shares.sum() / tot, 2), mo1=round(wm(d, 'mo1'), 3),
                mo5=round(wm(d, 'mo5'), 3), mo60=round(wm(d, 'mo60'), 3), settle=round(wm(d, 'settle'), 2),
                usd_1s=round(float((d.shares * d.mo1).sum()) / 100, 0), usd_5s=round(float((d.shares * d.mo5).sum()) / 100, 0), usd_60s=round(float((d.shares * d.mo60).sum()) / 100, 0),
                windows=int(d.mid.nunique()))


def show(title, rows):
    print(f'\n== {title} =='); print(pd.DataFrame(rows).to_string(index=False))


def hyst(tau, v, lin, lout):
    """Two-line state on the grid and, per grid point, the start time of the current on-run (nan when off)."""
    fin = np.isfinite(v)
    gap = np.r_[True, np.diff(tau) > 60.0]
    s = np.full(len(v), np.nan)
    s[fin & (v > lin)] = 1.0
    s[~fin | (v <= lout) | gap & ~(fin & (v > lin))] = 0.0          # a gap ends a kept state; a fresh entry may start on it
    s = pd.Series(s).ffill().fillna(0.0).values.astype(bool)
    st = np.r_[s[0], s[1:] & ~s[:-1]]
    start = np.where(s, pd.Series(np.where(st, tau, np.nan)).ffill().values, np.nan)
    return s, start


def read(G, S, T, Wcol, lag):
    """Per trade: guard value, state and state age of the maker side at W - lag (nan / False when no fresh value)."""
    n = len(T); v = np.full(n, np.nan); on = np.zeros(n, bool); age = np.full(n, np.nan)
    W = T[Wcol].values; up = T.up.values; mids = T.mid.values
    for m in np.unique(mids):
        tau, vb, va = G[m]; ii = np.flatnonzero(mids == m)
        for side_up in (True, False):
            k = ii[up[ii] == side_up]
            if not len(k):
                continue
            vs = vb if side_up else va; s, start = S[m][0 if side_up else 1]
            tm = W[k] - lag; j = np.searchsorted(tau, tm, side='right') - 1; jj = np.clip(j, 0, None)
            ok = (j >= 0) & (tm - tau[jj] <= 100.0) & np.isfinite(vs[jj])
            v[k] = np.where(ok, vs[jj], np.nan); on[k] = ok & s[jj]; age[k] = np.where(ok & s[jj], tm - start[jj], np.nan)
    return v, on, age


def main():
    sc = json.load(open(f'{RP}/scope.json')); coh = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}
    LN = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']
    P90, P80, P50 = LN['90'][0], LN['80'][0], LN['50'][0]
    ARMS = {'H0': (P90, P90), 'H1': (P90, -0.10), 'H2': (P90, P50), 'H3': (P80, -0.10)}
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet')
    T = R[(R.status != 'untimed') & (R.t_in < 60)].copy(); T['mid'] = T.mid.astype('int64'); T['W'] = T.mid.map(coh) + T.t_in * 1000.0
    P = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    P = P[P.mid.isin(set(T.mid))].sort_values(['mid', 'tau'])
    G = {m: (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in P.groupby('mid')}
    del P
    tot = T.shares.sum(); days = len(sc['windows']) / 288.0
    ST = {a: {m: (hyst(tau, vb, li, lo), hyst(tau, va, li, lo)) for m, (tau, vb, va) in G.items()} for a, (li, lo) in ARMS.items()}
    # ---- self-test
    ref = json.load(open(sys.argv[2]))                                  # D-C's saved output
    v0, on0, _ = read(G, ST['H0'], T, 'W', 250.0); h0 = T[on0]
    rng = np.random.default_rng(7); T['Wsh'] = T.groupby('mid').W.transform(lambda s: rng.permutation(s.values))
    _, ons, _ = read(G, ST['H1'], T, 'Wsh', 250.0); Dsh = wm(T[ons], 'mo1') - wm(T[~ons], 'mo1')
    print(f"selftest: H0 state-on at lag 250: shares {h0.shares.sum():.1f}, mo1 {wm(h0, 'mo1'):+.6f} vs D-C gate-on {ref['gate_on_250']['shares']:.1f}, {ref['gate_on_250']['mo1']:+.6f}; "
          f'H1 on-minus-off with trade times shuffled within window {Dsh:+.3f}')
    if abs(h0.shares.sum() - ref['gate_on_250']['shares']) > 1e-6 or abs(wm(h0, 'mo1') - ref['gate_on_250']['mo1']) > 1e-9 or abs(Dsh) >= 0.15:
        print('SELFTEST FAILED -> no table'); return 1
    out = {}
    for a, (li, lo) in ARMS.items():
        print(f'\n################ {a}: enter v > {li:+.4f}, stay while v > {lo:+.4f}')
        for lag in (250.0, 500.0):
            v, on, age = read(G, ST[a], T, 'W', lag); v = pd.Series(v, index=T.index); on = pd.Series(on, index=T.index); age = pd.Series(age, index=T.index)
            fresh = v.notna()
            rows = [row('state on', T[on], tot), row('  on, v > L_in (gate)', T[on & (v > li)], tot), row('  on, v <= L_in (hold)', T[on & (v <= li)], tot),
                    row('state off, v > L_out', T[~on & fresh & (v > lo)], tot), row('state off, v <= L_out', T[~on & fresh & (v <= lo)], tot), row('no fresh value', T[~fresh], tot)]
            show(f'{a}: touch flow by state of the maker side at W - {lag:.0f} ms (USD = cohort total, pro rata; {days:.2f} window-days)', rows)
            if lag == 250.0:
                ab = [(-1, 500, '< 0.5 s'), (500, 1000, '0.5-1 s'), (1000, 2000, '1-2 s'), (2000, 5000, '2-5 s'), (5000, np.inf, '>= 5 s')]
                son = T[on]; ts = son.shares.sum()
                show(f'{a}: state-on touch flow by how long the state has been on (share of state-on volume)', [row(n, son[(age[son.index] > x) & (age[son.index] <= y)], ts) for x, y, n in ab])
                s1 = float(son[age[son.index] >= 1000].shares.sum() / ts) if ts else float('nan')
                hold = T[on & (v <= li)]
                out[a] = dict(on_share=float(son.shares.sum() / tot), on_mo1=wm(son, 'mo1'), on_mo5=wm(son, 'mo5'), on_mo60=wm(son, 'mo60'), hold_share=float(hold.shares.sum() / tot),
                              hold_mo1=wm(hold, 'mo1'), hold_mo5=wm(hold, 'mo5'), hold_mo60=wm(hold, 'mo60'), age_ge_1s=s1,
                              usd_day_1s=float((son.shares * son.mo1).sum()) / 100 / days, usd_day_5s=float((son.shares * son.mo5).sum()) / 100 / days,
                              usd_day_60s=float((son.shares * son.mo60).sum()) / 100 / days)
        # grid facts 0-60 s
        sp = []; onb = ona = both = n = 0
        for m, (tau, vb, va) in G.items():
            t0 = coh[m]; k = (tau >= t0) & (tau < t0 + 60_000)
            sb, sa = ST[a][m][0][0][k], ST[a][m][1][0][k]
            onb += int(sb.sum()); ona += int(sa.sum()); both += int((sb & sa).sum()); n += int(k.sum())
            for s in (sb, sa):
                e = np.flatnonzero(np.diff(np.r_[0, s.astype(int), 0])); sp += list((e[1::2] - e[0::2]) * 50.0)
        sp = np.array(sp)
        print(f'{a} grid 0-60 s: state on {100 * (onb + ona) / (2 * n):.1f}% of side-time; spells per side-minute {len(sp) / (2 * len(G)):.1f}; length ms p50 {np.median(sp):.0f} p75 {np.quantile(sp, .75):.0f} '
              f'p90 {np.quantile(sp, .9):.0f}; on-time in spells >= 1 s {100 * sp[sp >= 1000].sum() / sp.sum():.1f}%, >= 2 s {100 * sp[sp >= 2000].sum() / sp.sum():.1f}%, >= 5 s {100 * sp[sp >= 5000].sum() / sp.sum():.1f}%; '
              f"both sides on {100 * both / n:.2f}% of time = {100 * both / (0.5 * (onb + ona)):.1f}% of a side's on-time")
        out[a].update(on_time=float((onb + ona) / (2 * n)), spells_per_side_min=float(len(sp) / (2 * len(G))), spell_p50_ms=float(np.median(sp)), both_share_of_on=float(both / (0.5 * (onb + ona))))
    h = out['H1']
    r6 = 'HOLD-OK' if (h['hold_mo1'] >= 0.10 and h['hold_mo5'] >= 0) else ('HOLD-ADVERSE' if (h['hold_mo1'] <= 0 or h['hold_mo5'] <= -0.20) else 'NEUTRAL')
    r7 = 'RESIDENT' if h['age_ge_1s'] >= 0.60 else 'NOT'
    print(f"\nR6 (H1 hold state at W - 250): mo1 {h['hold_mo1']:+.3f}, mo5 {h['hold_mo5']:+.3f} -> {r6}")
    print(f"R7 (H1 share of state-on volume with the state on >= 1 s): {100 * h['age_ge_1s']:.1f}% -> {r7}")
    print('\nper arm, state-on flow at W - 250 (pro rata, whole market; USD per window-day):')
    print(pd.DataFrame(out).T[['on_time', 'on_share', 'on_mo1', 'on_mo5', 'on_mo60', 'usd_day_1s', 'usd_day_5s', 'usd_day_60s', 'hold_share', 'hold_mo1', 'hold_mo5', 'hold_mo60', 'age_ge_1s',
                               'spells_per_side_min', 'spell_p50_ms', 'both_share_of_on']].round(3).to_string())
    json.dump(dict(arms=out, R6=r6, R7=r7), open(sys.argv[1], 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
