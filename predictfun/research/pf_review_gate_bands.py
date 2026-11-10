"""Review diagnostic D-E (2026-11-10, strategy review). Descriptive only; same consumed cohort as D-B..D-D
(220 windows of 2026-11-08, cut 23:00Z; 0-60 s touch trades; guard out of sample, ONE partial day; no interval, no inference).
Derived cohort outputs only (touch_trade_reasons.parquet, preds_t60k.parquet, minute_lines_t60k.json, scope.json).

Question (user 2026-11-10: "enter at the current line and pull only when the guard is clearly negative, we already have this
right?"): the studied pull buffer (B3 / LC50) keeps an accepted touch order while line - 0.01 c < v <= line, for at most 100 ms
(50 ms in the last pass; bands 0.005 / 0.02 / 0.04 were its sensitivity arms). D-D's two-line arms use a much lower exit line and
no time limit. Which of the two differences matters: the width of the band, or the time limit?
A frontier, NOT a selection: the forward-test candidates stay the two declared in D-D (H1, H2). Every arm enters at the PM90
minute-0 line. This values FLOW (every touch trade, share-weighted, pro rata), not our fills.

Arms (fixed before this table was seen; the D-D tables HAD been seen):
  uncapped   exit at line - d, d in 0 (= v3), 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30, and "never" (the line never ends the
             state; only a missing value or a grid gap does - the flow analogue of the rejected rung hold LH)
  reference  exit at -0.10 c (= D-D H1) and at the PM50 line (= D-D H2)
  capped     d = 0.01 with 50 ms (LC50) and 100 ms (B3 primary); d = 0.04 with 100 ms (B3 widest); d = 0.15 with 100 ms and 1 s
             (a wide band with a clock). The clock starts at the first grid point with v <= line and is reset by v > line.

Readings DECLARED BEFORE THE TABLE WAS SEEN (state read at W - 250 ms):
  R8 what limits the buffer. On the uncapped arm at the widest band the buffer study tested (d = 0.04):
     CAP-BOUND  if spell p50 >= 500 ms AND >= 60% of state-on volume trades with the state on >= 1 s (dropping the time limit
                from the existing buffer would be enough);
     BAND-BOUND if spell p50 < 500 ms AND that share < 60% (the band has to widen as well);  else MIXED.
  R9 the smallest uncapped d on the list with >= 60% of state-on volume at state age >= 1 s (printed; "none" if none).
  R10 the tail the exit line is for: flow while the "never" state is on and v <= line - 0.15.
     TAIL-ADVERSE if mo1 <= 0 AND mo5 <= -0.20 ; else TAIL-NOT-ADVERSE.
Self-test (falsifier): d = 0 and the -0.10 arm must reproduce D-D's saved H0 / H1 rows; a capped arm with cap 0 must equal
d = 0 and with an infinite cap must equal its uncapped arm on every grid point; the vectorised capped state must equal an
independent step-by-step state machine on 40 window-sides; with trade times shuffled within the window the on-minus-off
contrast of the -0.10 arm must collapse (|D| < 0.15).
Usage: predictfun/research/run_light.sh review_gate_bands -- python3 predictfun/research/pf_review_gate_bands.py <out.json> <D-D out.json>
"""
import json
import sys

import numpy as np
import pandas as pd

from pf_review_gate_hysteresis import OUT, RP, hyst, read, wm


def capped(tau, v, lin, lout, cap):
    """Two-line state whose stay below the enter line lasts < cap ms (cap = inf: the uncapped state)."""
    s, _ = hyst(tau, v, lin, lout)
    above = np.isfinite(v) & (v > lin)
    hold = s & ~above
    first = hold & ~np.r_[False, hold[:-1]]
    th = pd.Series(np.where(first, tau, np.nan)).ffill().values
    with np.errstate(invalid='ignore'):
        sc = above | (hold & (tau - th < cap))
    st = np.r_[sc[0], sc[1:] & ~sc[:-1]]
    start = np.where(sc, pd.Series(np.where(st, tau, np.nan)).ffill().values, np.nan)
    return sc, start


def brute(tau, v, lin, lout, cap):
    """Independent step-by-step state machine for the capped rule (self-test only)."""
    on = False; th = None; out = np.zeros(len(v), bool)
    for k in range(len(v)):
        fin = np.isfinite(v[k]); gap = k == 0 or tau[k] - tau[k - 1] > 60.0
        if fin and v[k] > lin:
            on = True; th = None
        elif (not fin) or v[k] <= lout or gap:
            on = False; th = None
        elif on:
            th = tau[k] if th is None else th
            if tau[k] - th >= cap:
                on = False
        out[k] = on
    return out


def main():
    sc = json.load(open(f'{RP}/scope.json')); coh = {int(w['market']): w['ws'] * 1000.0 for w in sc['windows']}
    LN = json.load(open(f'{RP}/research/step4/minute_lines_t60k.json'))['lines']
    P90, P50 = LN['90'][0], LN['50'][0]
    INF = float('inf')
    ARMS = {f'd={d:g}': (P90 - d, INF) for d in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30)}
    ARMS.update({'never': (-INF, INF), 'exit -0.10 (H1)': (-0.10, INF), 'exit PM50 (H2)': (P50, INF),
                 'd=0.01 cap 50 ms (LC50)': (P90 - 0.01, 50.0), 'd=0.01 cap 100 ms (B3)': (P90 - 0.01, 100.0), 'd=0.04 cap 100 ms': (P90 - 0.04, 100.0),
                 'd=0.15 cap 100 ms': (P90 - 0.15, 100.0), 'd=0.15 cap 1 s': (P90 - 0.15, 1000.0)})
    R = pd.read_parquet(f'{OUT}/touch_trade_reasons.parquet')
    T = R[(R.status != 'untimed') & (R.t_in < 60)].copy(); T['mid'] = T.mid.astype('int64'); T['W'] = T.mid.map(coh) + T.t_in * 1000.0
    P = pd.read_parquet(f'{RP}/research/step5/20261108/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
    P = P[P.mid.isin(set(T.mid))].sort_values(['mid', 'tau'])
    G = {m: (g.tau.values.astype(float), g.v_bid.values.astype(float), g.v_ask.values.astype(float)) for m, g in P.groupby('mid')}
    del P
    tot = T.shares.sum(); days = len(sc['windows']) / 288.0
    ST = {a: {m: (capped(tau, vb, P90, lo, cap), capped(tau, va, P90, lo, cap)) for m, (tau, vb, va) in G.items()} for a, (lo, cap) in ARMS.items()}
    # ---- self-test
    ref = json.load(open(sys.argv[2]))['arms']                           # D-D's saved output
    bad = []
    for a, r in (('d=0', 'H0'), ('exit -0.10 (H1)', 'H1'), ('exit PM50 (H2)', 'H2')):
        v, on, _ = read(G, ST[a], T, 'W', 250.0); son = T[on]
        got = (float(son.shares.sum() / tot), wm(son, 'mo1'))
        print(f"selftest: {a} state-on share {got[0]:.6f}, mo1 {got[1]:+.6f} vs D-D {r} {ref[r]['on_share']:.6f}, {ref[r]['on_mo1']:+.6f}")
        if abs(got[0] - ref[r]['on_share']) > 1e-9 or abs(got[1] - ref[r]['on_mo1']) > 1e-9:
            bad.append(f'{a} does not reproduce D-D {r}')
    lo = P90 - 0.01; n0 = ninf = nb = 0
    rng = np.random.default_rng(11); pick = set(rng.choice(sorted(G), 20, replace=False).tolist())
    for m, (tau, vb, va) in G.items():
        for vs, k in ((vb, 0), (va, 1)):
            n0 += int((capped(tau, vs, P90, lo, 0.0)[0] != ST['d=0'][m][k][0]).sum())
            ninf += int((capped(tau, vs, P90, lo, INF)[0] != hyst(tau, vs, P90, lo)[0]).sum())
            if m in pick:
                nb += int((brute(tau, vs, P90, lo, 100.0) != ST['d=0.01 cap 100 ms (B3)'][m][k][0]).sum())
                nb += int((brute(tau, vs, P90, P90 - 0.15, 1000.0) != ST['d=0.15 cap 1 s'][m][k][0]).sum())
    print(f'selftest: grid points where cap 0 differs from d = 0: {n0}; infinite cap differs from uncapped: {ninf}; vectorised differs from the step machine on 40 window-sides: {nb}')
    if n0 or ninf or nb:
        bad.append('capped state machine mismatch')
    T['Wsh'] = T.groupby('mid').W.transform(lambda s: np.random.default_rng(7).permutation(s.values))
    _, ons, _ = read(G, ST['exit -0.10 (H1)'], T, 'Wsh', 250.0); Dsh = wm(T[ons], 'mo1') - wm(T[~ons], 'mo1')
    print(f'selftest: -0.10 arm on-minus-off mo1 with trade times shuffled within the window {Dsh:+.3f}')
    if abs(Dsh) >= 0.15:
        bad.append('shuffled contrast did not collapse')
    if bad:
        print('SELFTEST FAILED -> no table:', bad); return 1
    out = {}
    for a, (lo, cap) in ARMS.items():
        v, on, age = read(G, ST[a], T, 'W', 250.0); v = pd.Series(v, index=T.index); on = pd.Series(on, index=T.index); age = pd.Series(age, index=T.index)
        son = T[on]; ts = son.shares.sum(); hold = T[on & (v <= P90)]; tail = T[on & (v <= P90 - 0.15)]
        sp = []; onb = ona = both = n = 0
        for m, (tau, vb, va) in G.items():
            t0 = coh[m]; k = (tau >= t0) & (tau < t0 + 60_000)
            sb, sa = ST[a][m][0][0][k], ST[a][m][1][0][k]
            onb += int(sb.sum()); ona += int(sa.sum()); both += int((sb & sa).sum()); n += int(k.sum())
            for s in (sb, sa):
                e = np.flatnonzero(np.diff(np.r_[0, s.astype(int), 0])); sp += list((e[1::2] - e[0::2]) * 50.0)
        sp = np.array(sp)
        out[a] = dict(exit_line=lo, cap_ms=cap, on_time_pct=100 * (onb + ona) / (2 * n), spells_per_side_min=len(sp) / (2 * len(G)), spell_p50_ms=float(np.median(sp)),
                      spell_p75_ms=float(np.quantile(sp, .75)), spell_p90_ms=float(np.quantile(sp, .9)), on_time_in_spells_ge_1s_pct=100 * float(sp[sp >= 1000].sum() / sp.sum()),
                      vol_pct=100 * float(ts / tot), mo1=wm(son, 'mo1'), mo5=wm(son, 'mo5'), mo60=wm(son, 'mo60'), usd_day_1s=float((son.shares * son.mo1).sum()) / 100 / days,
                      usd_day_5s=float((son.shares * son.mo5).sum()) / 100 / days, hold_vol_pct=100 * float(hold.shares.sum() / tot), hold_trades=len(hold),
                      hold_mo1=wm(hold, 'mo1'), hold_mo5=wm(hold, 'mo5'), vol_state_age_ge_1s_pct=100 * float(son[age[son.index] >= 1000].shares.sum() / ts) if ts else float('nan'),
                      both_pct_of_on=100 * both / (0.5 * (onb + ona)), tail_vol_pct=100 * float(tail.shares.sum() / tot), tail_trades=len(tail), tail_mo1=wm(tail, 'mo1'), tail_mo5=wm(tail, 'mo5'),
                      tail_mo60=wm(tail, 'mo60'))
    D = pd.DataFrame(out).T
    pd.set_option('display.width', 320); pd.set_option('display.max_columns', 40)
    print(f'\n== enter at the PM90 line ({P90:+.4f} c), state read at W - 250 ms; touch volume {tot:.0f} sh, {len(T)} trades, {T.mid.nunique()} windows, {days:.2f} window-days ==')
    print(D[['exit_line', 'cap_ms', 'on_time_pct', 'spells_per_side_min', 'spell_p50_ms', 'spell_p75_ms', 'spell_p90_ms', 'on_time_in_spells_ge_1s_pct', 'vol_state_age_ge_1s_pct', 'both_pct_of_on']].astype(float).round(3).to_string())
    print(D[['vol_pct', 'mo1', 'mo5', 'mo60', 'usd_day_1s', 'usd_day_5s', 'hold_vol_pct', 'hold_trades', 'hold_mo1', 'hold_mo5']].astype(float).round(3).to_string())
    # ---- declared readings, evaluated
    w = out['d=0.04']
    r8 = 'CAP-BOUND' if (w['spell_p50_ms'] >= 500 and w['vol_state_age_ge_1s_pct'] >= 60) else ('BAND-BOUND' if (w['spell_p50_ms'] < 500 and w['vol_state_age_ge_1s_pct'] < 60) else 'MIXED')
    ok9 = [d for d in (0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30) if out[f'd={d:g}']['vol_state_age_ge_1s_pct'] >= 60]
    r9 = min(ok9) if ok9 else None
    nv = out['never']
    r10 = 'TAIL-ADVERSE' if (nv['tail_mo1'] <= 0 and nv['tail_mo5'] <= -0.20) else 'TAIL-NOT-ADVERSE'
    print(f"\nR8 (uncapped d = 0.04): spell p50 {w['spell_p50_ms']:.0f} ms, state-on volume at state age >= 1 s {w['vol_state_age_ge_1s_pct']:.1f}% -> {r8}")
    print(f'R9 smallest uncapped d with >= 60% of state-on volume at state age >= 1 s: {r9}')
    print(f"R10 ('never' state on and v <= line - 0.15): {nv['tail_vol_pct']:.2f}% of volume, {nv['tail_trades']} trades, mo1 {nv['tail_mo1']:+.3f}, mo5 {nv['tail_mo5']:+.3f}, mo60 {nv['tail_mo60']:+.3f} -> {r10}")
    b3, h1 = out['d=0.01 cap 100 ms (B3)'], out['exit -0.10 (H1)']
    print(f"buffer (d = 0.01, 100 ms) vs v3: on-time {b3['on_time_pct']:.2f}% vs {out['d=0']['on_time_pct']:.2f}%; spells per side-minute {b3['spells_per_side_min']:.1f} vs {out['d=0']['spells_per_side_min']:.1f}; "
          f"spell p50 {b3['spell_p50_ms']:.0f} vs {out['d=0']['spell_p50_ms']:.0f} ms; H1: {h1['on_time_pct']:.2f}%, {h1['spells_per_side_min']:.1f}, {h1['spell_p50_ms']:.0f} ms")
    json.dump(dict(arms=out, R8=r8, R9=r9, R10=r10, n_trades=len(T), shares=float(tot), windows=int(T.mid.nunique())), open(sys.argv[1], 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
