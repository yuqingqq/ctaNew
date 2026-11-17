"""SOFT_JOINT CHECK (2026-11-17, session aea5bfb9; USER: "Can you load my codex session's latest progress, it has been using a new model to
proceed the mm strategy development, check that"). POST HOC, descriptive, read-only: no replay, no fit, no verdict.

WHAT: the other session's joint flow/price controller pilots - KEEP_ALL_CAP, EMPIRICAL, JOINT, JOINT_H2 (flow_control_20261116_v3/pilot) and
SOFT_JOINT, its retained baseline (value_bridge_20261116_v1/pilot) - on the 12 preselected Oct27 markets, full five-minute contracts. From
their own ledgers: where the marked P&L comes from - by contract minute (the inventory ledger's incremental marked P&L), by market, and by
fill class (order age < 100 ms, cancel pending, WS / E2 source, through / at price, depth, minute) with 1 s / 5 s markouts; the final
minute's price range per market; minutes 1-4 against minute 5.
CONTROLS (any failure -> no output): per case the fills, markets and inventory ledgers reconcile (shares, terminal and settlement P&L;
the minute increments sum to each market's terminal P&L) and equal the recorded report rows (shares, orders, pulls, terminal, settlement);
planted errors (one fill's value, one minute increment, moved by 1 c) must each fail them.
LIMITS: their simulation (rule M, queue M, unknown depth 0, 58 / 37 ms, M50 only); 12 markets of one consumed day; descriptive.
Usage (run_light, PF_SLOT=claude): pf_soft_joint_check.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

IO = '/home/yuqing/ctaNew/data/predict_fun/research'
SRC = {'flow_control_20261116_v3/pilot': ('KEEP_ALL_CAP', 'EMPIRICAL', 'JOINT', 'JOINT_H2'), 'value_bridge_20261116_v1/pilot': ('SOFT_JOINT',)}
OUTD = f'{IO}/soft_joint_check_20261117'
TOL = 1e-6


def cs(y, col):
    ok = y[col].notna()
    q = float(y.loc[ok, 'shares'].sum())
    return 100.0 * float(y.loc[ok, col].sum()) / q if q > 0 else float('nan')


def load():
    F, MK, IV, ROWS = [], [], [], {}
    for d, cases in SRC.items():
        f = pd.read_parquet(f'{IO}/{d}/fills.parquet'); mk = pd.read_parquet(f'{IO}/{d}/markets.parquet')
        iv = pd.read_parquet(f'{IO}/{d}/inventory.parquet'); rep = json.load(open(f'{IO}/{d}/report.json'))
        F.append(f[f.case.isin(cases)]); MK.append(mk[mk.case.isin(cases)]); IV.append(iv[iv.case.isin(cases)])
        for r in rep['rows']:
            if r['case'] in cases:
                ROWS[r['case']] = r
    return pd.concat(F, ignore_index=True), pd.concat(MK, ignore_index=True), pd.concat(IV, ignore_index=True), ROWS


def controls(F, MK, IV, ROWS):
    bad = []
    for c, r in ROWS.items():
        f, mk, iv = F[F.case == c], MK[MK.case == c], IV[IV.case == c]
        for nm, a, b in (('shares fills/report', f.shares.sum(), r['shares']), ('shares markets/report', mk.shares.sum(), r['shares']),
                         ('terminal fills/report', f.terminal_pnl_usd.sum(), r['terminal_pnl_usd']),
                         ('terminal markets/report', mk.terminal_pnl_usd.sum(), r['terminal_pnl_usd']),
                         ('settlement fills/report', f.settlement_pnl_usd.sum(), r['settlement_pnl_usd']),
                         ('orders markets/report', mk.orders.sum(), r['orders']), ('pulls markets/report', mk.pulls.sum(), r['pulls'])):
            if abs(float(a) - float(b)) > TOL * max(1.0, abs(float(b))):
                bad.append(f'{c}: {nm} {a} vs {b}')
        inc = iv.groupby('m').incremental_marked_pnl_usd.sum(); term = mk.set_index('m').terminal_pnl_usd
        if len(r) and (inc.reindex(term.index) - term).abs().max() > 1e-6:
            bad.append(f'{c}: minute increments do not sum to the market terminal P&L')
        if mk.m.nunique() != 12:
            bad.append(f'{c}: {mk.m.nunique()} markets')
    if set(ROWS) != {c for cs_ in SRC.values() for c in cs_}:
        bad.append(f'report rows {sorted(ROWS)}')
    return bad


def main():
    F, MK, IV, ROWS = load()
    bad = controls(F, MK, IV, ROWS)
    F2 = F.copy(); F2.loc[F2.index[0], 'terminal_pnl_usd'] += 0.01                     # planted error: one fill's value moved by 1 c
    IV2 = IV.copy(); IV2.loc[IV2.index[0], 'incremental_marked_pnl_usd'] += 0.01         # planted error: one minute increment moved
    if not controls(F2, MK, IV, ROWS) or not controls(F, MK, IV2, ROWS):
        bad.append('planted errors not caught')
    if bad:
        sys.exit(f'SOFT-JOINT-CHECK CONTROL FAILED -> no output: {bad}')
    L = [f'SOFT_JOINT CHECK (POST HOC, descriptive; the other session\'s pilot ledgers, 12 preselected Oct27 markets, full 5-minute contracts); '
         f'as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}',
         'controls: fills / markets / inventory ledgers reconcile with each other and with the recorded report rows for every case', '',
         'PART 1: marked P&L by contract minute ($, 12 markets) and markouts (c/sh) - minutes 1-4 vs minute 5']
    for c in ('KEEP_ALL_CAP', 'EMPIRICAL', 'JOINT', 'JOINT_H2', 'SOFT_JOINT'):
        f, iv = F[F.case == c], IV[IV.case == c]
        a, b = f[f.minute < 5], f[f.minute == 5]
        L.append(f'  {c:<13} total {f.terminal_pnl_usd.sum():+8.1f} | by minute ' + ' '.join(
            f'{iv[iv.minute == mm].incremental_marked_pnl_usd.sum():+7.1f}' for mm in range(1, 6)) +
            f' | minutes 1-4: {a.shares.sum():7,.0f} sh, +1s {cs(a, "markout_1000_usd"):+.2f}, +5s {cs(a, "markout_5000_usd"):+.2f}'
            f' | minute 5: {b.shares.sum():7,.0f} sh, +1s {cs(b, "markout_1000_usd"):+.2f}, +5s {cs(b, "markout_5000_usd"):+.2f}')
    L += ['', 'PART 2: by market - whole contract | minute 5 ($), sorted by SOFT_JOINT; final-minute pre-fill mid range seen by any case, outcome']
    tot = IV.pivot_table(index='m', columns='case', values='incremental_marked_pnl_usd', aggfunc='sum')
    m5 = IV[IV.minute == 5].pivot_table(index='m', columns='case', values='incremental_marked_pnl_usd')
    rng = F[F.minute == 5].groupby('m').agg(lo=('pre_mid_cents', 'min'), hi=('pre_mid_cents', 'max'), y=('settlement_yes', 'first'))
    for m in tot.sort_values('SOFT_JOINT', ascending=False).index:
        rr = rng.loc[m] if m in rng.index else None
        L.append(f'  {m}: ' + ' | '.join(f'{c} {tot.loc[m, c]:+7.1f} / {m5.loc[m, c]:+7.1f}' for c in ('KEEP_ALL_CAP', 'EMPIRICAL', 'JOINT', 'SOFT_JOINT')) +
                 (f' | minute-5 mid {rr.lo:.1f}-{rr.hi:.1f}, outcome {"Up" if rr.y > 0.5 else "Down"}' if rr is not None else ''))
    s = tot.SOFT_JOINT.sort_values(ascending=False)
    L.append(f'  SOFT_JOINT: best two markets {s.iloc[:2].sum():+.1f} of {s.sum():+.1f} ({s.iloc[:2].sum() / s.sum():.0%}); the other ten {s.iloc[2:].sum():+.1f}; '
             f'median market {s.median():+.1f}; positive {int((s > 0).sum())} / 12')
    top2 = list(s.index[:2])
    L.append('  without the best two SOFT_JOINT markets: ' + ' | '.join(f'{c} {tot[c].drop(top2).sum():+.1f}' for c in ('KEEP_ALL_CAP', 'EMPIRICAL', 'JOINT', 'SOFT_JOINT')))
    L += ['', 'PART 3: SOFT_JOINT by fill class: shares (share) | spread | +1s | +5s c/sh | terminal $ (c/sh) | settlement $']
    x = F[F.case == 'SOFT_JOINT']
    for nm, msk in (('all', x.shares > -1), ('age < 100 ms', x.age_ms < 100), ('age >= 100 ms', x.age_ms >= 100), ('cancel pending', x.cancel_pending),
                    ('not cancel pending', ~x.cancel_pending), ('WS-timed', x.ws_timed), ('E2 / imputed', ~x.ws_timed), ('through', x.through),
                    ('at price', ~x.through), ('WS & age >= 100 & live', x.ws_timed & (x.age_ms >= 100) & ~x.cancel_pending)):
        y = x[msk]
        L.append(f'  {nm:<24} {y.shares.sum():7,.0f} ({y.shares.sum() / x.shares.sum():5.1%}) | {cs(y, "spread_capture_usd"):+.2f} | {cs(y, "markout_1000_usd"):+.2f} | '
                 f'{cs(y, "markout_5000_usd"):+.2f} | {y.terminal_pnl_usd.sum():+7.1f} ({cs(y, "terminal_pnl_usd"):+.2f}) | {y.settlement_pnl_usd.sum():+7.1f}')
    for g in sorted(x.depth_group.astype(str).unique()):
        y = x[x.depth_group.astype(str) == g]
        L.append(f'  depth {g:<18} {y.shares.sum():7,.0f} ({y.shares.sum() / x.shares.sum():5.1%}) | {cs(y, "spread_capture_usd"):+.2f} | '
                 f'{cs(y, "markout_1000_usd"):+.2f} | {cs(y, "markout_5000_usd"):+.2f} | {y.terminal_pnl_usd.sum():+7.1f} ({cs(y, "terminal_pnl_usd"):+.2f})')
    mk = MK[MK.case == 'SOFT_JOINT']
    L += ['', f'PART 4: SOFT_JOINT activity: orders {mk.orders.sum():,} and guard pulls {mk.pulls.sum():,} over 60 market-minutes '
          f'({mk.orders.sum() / 60:.0f} / {mk.pulls.sum() / 60:.0f} a market-minute)']
    os.makedirs(OUTD, exist_ok=True)
    open(f'{OUTD}/soft_joint_check.txt', 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L), flush=True)


if __name__ == '__main__':
    main()
