"""Review diagnostic D-C (2026-11-10, strategy review). Descriptive only; follow-up to D-B on the same consumed cohort
(220 windows of 2026-11-08, cut 23:00Z; 0-60 s touch trades; guard out of sample, ONE partial day; no interval, no inference).
Derived cohort outputs only (touch_trade_reasons.parquet, preds_t60k.parquet, minute_lines_t60k.json, scope.json).

Why: D-B's "gate off <= 100 ms" bucket was read at W - 50 ms, where W (receipt of the pending message) lags the match by an
unknown amount, so a trade can itself have switched the gate off. Re-read the zone at larger leads.

Readings DECLARED BEFORE THE TABLES WERE SEEN:
  R2' grace zone, read at W - 250 ms (gate went off 250-350 ms before the trade report):
        GRACE-FAVOURABLE if mo1 >= +0.20 c/sh ; GRACE-ADVERSE if mo1 <= 0 ; else NEUTRAL.
  R4  two-sidedness on the 50 ms grid, 0-60 s: share of a side's gate-on time during which the other side is also on:
        TWO-SIDED if >= 30% ; ONE-SIDED if <= 10% ; else BETWEEN.
  R5  persistence of a low "stay" line (v > -0.10 c, i.e. PM90 line - 0.15): share of its on-time in spells >= 2 s:
        PERSISTENT if >= 60% ; else NOT.
  Line frontier (PM50..PM95, read at W - 250 ms) is a description of ONE consumed day: choosing a line from it would be
  selection on seen data and is not done here.
Self-test (falsifier): the lag-50 on / off rows must reproduce D-B's saved numbers (review_gate_information.json) to 1e-9;
with trade times shuffled within window the D(50) contrast must collapse (|D| < 0.15).
Usage: predictfun/research/run_light.sh review_gate_followup -- python3 predictfun/research/pf_review_gate_followup.py <out.json> <D-B out.json>
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


def state_at(G, T, Wcol, lag, line):
    """Maker side's guard value at W - lag and how long it has been on / off (strict 50 ms grid). NaN = no fresh value."""
    v = np.full(len(T), np.nan); dur = np.full(len(T), np.nan)
    W = T[Wcol].values; up = T.up.values; mids = T.mid.values
    for m in np.unique(mids):
        tau, vb, va = G[m]; ii = np.flatnonzero(mids == m)
        for side_up in (True, False):
            k = ii[up[ii] == side_up]
            if not len(k):
                continue
            vs = vb if side_up else va; on = vs > line; fin = np.isfinite(vs)
            same = np.r_[False, (on[1:] == on[:-1]) & fin[1:] & fin[:-1] & (np.diff(tau) <= 60.0)]
            brk = np.flatnonzero(~same); start = brk[np.searchsorted(brk, np.arange(len(tau)), side='right') - 1]
            tm = W[k] - lag; j = np.searchsorted(tau, tm, side='right') - 1; jj = np.clip(j, 0, None)
            ok = (j >= 0) & (tm - tau[jj] <= 100.0) & fin[jj]
            v[k] = np.where(ok, vs[jj], np.nan); dur[k] = np.where(ok, tm - tau[start[jj]], np.nan)
    return v, dur


def main():
    sc = json.load(open(f'{RP}/scope.json')); coh = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}
    LN = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']; LINE = LN['90'][0]
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet')
    T = R[(R.status != 'untimed') & (R.t_in < 60)].copy(); T['mid'] = T.mid.astype('int64'); T['W'] = T.mid.map(coh) + T.t_in * 1000.0
    P = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    P = P[P.mid.isin(set(T.mid))].sort_values(['mid', 'tau'])
    G = {m: (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in P.groupby('mid')}
    del P
    tot = T.shares.sum()
    # ---- self-test
    ref = json.load(open(sys.argv[2]))
    v50, d50 = state_at(G, T, 'W', 50.0, LINE)
    D50 = wm(T[v50 > LINE], 'mo1') - wm(T[v50 <= LINE], 'mo1')
    rng = np.random.default_rng(7); T['Wsh'] = T.groupby('mid').W.transform(lambda s: rng.permutation(s.values))
    vs_, _ = state_at(G, T, 'Wsh', 50.0, LINE); Dsh = wm(T[vs_ > LINE], 'mo1') - wm(T[vs_ <= LINE], 'mo1')
    print(f'selftest: D(50) {D50:+.6f} vs saved {ref["D"]["50"]:+.6f}; with trade times shuffled within window D = {Dsh:+.3f}')
    if abs(D50 - ref['D']['50']) > 1e-9 or abs(Dsh) >= 0.15:
        print('SELFTEST FAILED -> no table'); return 1
    # ---- R2': zone at larger leads
    zb = [(-1, 100, 'off <= 100 ms'), (100, 500, 'off 100-500 ms'), (500, 2000, 'off 0.5-2 s'), (2000, np.inf, 'off > 2 s')]
    res = {}
    for lag in (50.0, 150.0, 250.0, 500.0):
        v, dur = state_at(G, T, 'W', lag, LINE); on = T[v > LINE]; off = T[v <= LINE]; do = pd.Series(dur, index=T.index)
        rows = [row(f'lag {lag:.0f}: on', on, tot)] + [row(f'lag {lag:.0f}: {n}', off[(do[off.index] > a) & (do[off.index] <= b)], tot) for a, b, n in zb]
        show(f'time since the gate was last on, state read at W - {lag:.0f} ms', rows)
        z = off[do[off.index] <= 100]; res[lag] = wm(z, 'mo1')
        if lag == 250.0:
            g250 = dict(shares=float(on.shares.sum()), mo1=wm(on, 'mo1'), mo5=wm(on, 'mo5'), mo60=wm(on, 'mo60'))
        for band in (0.01, 0.04):
            b = z[pd.Series(v, index=T.index)[z.index] > LINE - band]
            print(f'   within {band} c below the line: trades {len(b)}, shares {b.shares.sum():.0f} ({100 * b.shares.sum() / tot:.2f}%), mo1 {wm(b, "mo1"):+.3f}, mo5 {wm(b, "mo5"):+.3f}, mo60 {wm(b, "mo60"):+.3f}')
    r2 = 'GRACE-FAVOURABLE' if res[250.0] >= 0.20 else ('GRACE-ADVERSE' if res[250.0] <= 0 else 'NEUTRAL')
    print(f"\nR2': zone mo1 by lead {dict((int(k), round(x, 3)) for k, x in res.items())} -> reading at 250 = {r2}")
    # ---- line frontier at W - 250 ms (description of one consumed day)
    v250, _ = state_at(G, T, 'W', 250.0, -np.inf); v250 = pd.Series(v250, index=T.index)
    rows = []
    for name in ('50', '60', '70', '80', '90', '95'):
        ln = LN[name][0]; rows.append(row(f'v > PM{name} ({ln:+.4f})', T[v250 > ln], tot))
    rows.append(row('v > -0.10 (stay line used in R5)', T[v250 > -0.10], tot)); rows.append(row('all with a fresh value', T[v250.notna()], tot))
    show('touch flow above each minute-0 line, guard read at W - 250 ms (USD = whole cohort, pro-rata, 220 windows)', rows)
    # complement: flow between lines
    edges = [('below PM50', -np.inf, LN['50'][0]), ('PM50-60', LN['50'][0], LN['60'][0]), ('PM60-70', LN['60'][0], LN['70'][0]), ('PM70-80', LN['70'][0], LN['80'][0]),
             ('PM80-90', LN['80'][0], LN['90'][0]), ('PM90-95', LN['90'][0], LN['95'][0]), ('above PM95', LN['95'][0], np.inf)]
    show('touch flow between lines, guard read at W - 250 ms', [row(n, T[(v250 > a) & (v250 <= b)], tot) for n, a, b in edges])
    # ---- grid facts: two-sidedness, correlation, spells per line
    both = 0; onb = 0; ona = 0; n = 0; vv = []
    spells = {k: [] for k in ('PM90', 'PM70', 'PM50', 'stay -0.10')}; tsh = {k: [0, 0] for k in spells}
    lines = {'PM90': LN['90'][0], 'PM70': LN['70'][0], 'PM50': LN['50'][0], 'stay -0.10': -0.10}
    for m, (tau, vb, va) in G.items():
        t0 = coh[m]; k = (tau >= t0) & (tau < t0 + 60_000); b, a = vb[k], va[k]; f = np.isfinite(b) & np.isfinite(a)
        both += int(((b > LINE) & (a > LINE) & f).sum()); onb += int(((b > LINE) & f).sum()); ona += int(((a > LINE) & f).sum()); n += int(f.sum())
        vv.append(np.c_[b[f], a[f]][::10])
        for name, ln in lines.items():
            for x in (b, a):
                fx = np.isfinite(x); on = (x > ln) & fx; tsh[name][0] += int(on.sum()); tsh[name][1] += int(fx.sum())
                e = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0])); spells[name] += list((e[1::2] - e[0::2]) * 50.0)
    vv = np.vstack(vv); share_other = both / (0.5 * (onb + ona))
    r4 = 'TWO-SIDED' if share_other >= 0.30 else ('ONE-SIDED' if share_other <= 0.10 else 'BETWEEN')
    print(f'\ngrid 0-60 s: bid on {100 * onb / n:.2f}%, ask on {100 * ona / n:.2f}%, both on {100 * both / n:.3f}% of time (independent would be {100 * (onb / n) * (ona / n):.3f}%); '
          f"share of a side's on-time with the other side also on {100 * share_other:.2f}% -> R4 = {r4}; corr(v_bid, v_ask) = {np.corrcoef(vv[:, 0], vv[:, 1])[0, 1]:+.3f}")
    for name in spells:
        sp = np.array(spells[name]); nw = len(G) * 2
        print(f'{name:>10}: on {100 * tsh[name][0] / tsh[name][1]:5.1f}% of side-time; spells per side-minute {len(sp) / nw:5.1f}; length ms p50 {np.median(sp):6.0f} p75 {np.quantile(sp, .75):6.0f} p90 {np.quantile(sp, .9):6.0f}; '
              f'on-time in spells >= 500 ms {100 * sp[sp >= 500].sum() / sp.sum():5.1f}%, >= 2 s {100 * sp[sp >= 2000].sum() / sp.sum():5.1f}%, >= 5 s {100 * sp[sp >= 5000].sum() / sp.sum():5.1f}%')
    sp = np.array(spells['stay -0.10']); s2 = float(sp[sp >= 2000].sum() / sp.sum())
    print(f"R5: stay-line on-time in spells >= 2 s {100 * s2:.1f}% -> reading = {'PERSISTENT' if s2 >= 0.60 else 'NOT'}")
    json.dump(dict(zone_mo1_by_lead={int(k): x for k, x in res.items()}, R2p=r2, R4=r4, share_other=share_other, stay_s2=s2, gate_on_250=g250), open(sys.argv[1], 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
