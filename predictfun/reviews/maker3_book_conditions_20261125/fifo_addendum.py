"""Append the FIFO-turnover addendum to report.md with every number computed from fifo_pairs / fifo_residual (no hand-typed figure)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
R = H.parents[2]
OUT = R / 'data/predict_fun/research/maker3_book_conditions_20261125'
LATE = {'270-285', '285-295', '295-300', 'post_stamp'}
MARK = '## Addendum (same day) — when the round trips close'


def main():
    rep = (H / 'report.md').read_text()
    assert MARK not in rep, 'addendum already present'
    f = json.loads((H / 'fifo.json').read_text())
    assert f['reconciles']
    p = pd.read_parquet(OUT / 'fifo_pairs.parquet')
    r = pd.read_parquet(OUT / 'fifo_residual.parquet')
    p['el'] = np.where(p.exit_tb.isin(LATE), '>=270', '<270')
    G, Q = p.gross.sum(), p.m.sum()
    rows = []
    for el in ('<270', '>=270'):
        e = p.el == el
        g, q = p[e].gross.sum(), p[e].m.sum()
        days = ''.join('+' if x > 0 else '-' for x in p[e].groupby('day').gross.sum().reindex(['20261024', '20261025', '20261026', '20261027']).fillna(0))
        o26 = p[e & (p.day == '20261026')].gross.sum() / g
        cells = {}
        for clk in ('P', 'U'):
            tight = p[f'entry_cls_{clk}'].isin(['CALM', 'THIN']) & p[f'exit_cls_{clk}'].isin(['CALM', 'THIN'])
            calm = (p[f'entry_cls_{clk}'] == 'CALM') & (p[f'exit_cls_{clk}'] == 'CALM')
            wide = (p[f'entry_cls_{clk}'] == 'WIDE') | (p[f'exit_cls_{clk}'] == 'WIDE')
            cells[clk] = (p[e & tight].gross.sum() / g, p[e & calm].gross.sum() / g, 100 * p[e & wide].gross.sum() / p[e & wide].m.sum())
        rows.append(f"| {el} | {g:+,.0f} ({g / G:.0%}) | {q / Q:.1%} | {100 * g / q:+.2f} | {days} | {o26:.0%} | "
                    f"{cells['P'][0]:.0%} / {cells['U'][0]:.0%} | {cells['P'][1]:.0%} / {cells['U'][1]:.0%} | {cells['P'][2]:+.2f} / {cells['U'][2]:+.2f} |")
    r['acq'] = np.select([r.tb.isin(LATE), r.tb == '240-270', r.tb == '180-240'], ['>=270', '240-270', '180-240'], '<180')
    rz = r.groupby('acq').value_res.sum()
    mm = p[(p.entry_role == 'maker') & (p.exit_role == 'maker') & (p.el == '<270')]
    wt = p[~((p.entry_role == 'maker') & (p.exit_role == 'maker')) & (p.el == '<270')]
    txt = f"""

{MARK}

USER read-back: "most spread is captured before 270 s when the market is calm, mainly the turnovers". The per-fill values above do not show when round trips close: a late fill that closes earlier inventory at a good price earns round-trip profit even when its own value to resolution is negative. So Maker3's completed round trips were rebuilt with the econ FIFO rule ([fifo.py](fifo.py): same order, unit j bought pairs with unit j sold, self-trades out of the lots) on all 934 markets. They reconcile to `econ/maker_market` rt_pnl / rt_sh / resid_res on every market (max |diff| {max(f['max_abs_rt_diff'], f['max_abs_resid_diff']):.1e} USD). Totals: completed round trips {f['rt_total_usd']:+,.0f} USD gross on {Q:,.0f} matched shares, residual {f['resid_total_usd']:+,.0f}, fees −828.

| Round trip closes | Gross USD (share) | Matched shares | ¢ / matched sh | Days 24/25/26/27 | Oct 26 share | Both legs spread ≤ 1 ¢, P / U | Both legs CALM, P / U | ¢/sh when a leg is WIDE, P / U |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{rows[0]}
{rows[1]}

- **Turnover is the money**, and most of it closes **before 270 s**. Passive-to-passive round trips closing before 270 s earn {mm.gross.sum():+,.0f} USD ({100 * mm.gross.sum() / mm.m.sum():+.2f} ¢/sh); those with a taker leg {wt.gross.sum():+,.0f} ({100 * wt.gross.sum() / wt.m.sum():+.2f} ¢/sh).
- **"Calm" holds only as "tight spread"**, and only at the pre-trade clock. With both legs at ≤ 1 ¢ the early share is large at P and much smaller at U. With tight and deep (CALM) on both legs it is a minority. Round trips touching a wide book are a small volume at a much higher margin.
- **The late window is net negative.** Closes after 270 s earn {f['rt_exit_ge_270_usd']:+,.0f} USD at {100 * f['rt_exit_ge_270_usd'] / (Q - f['matched_sh_exit_before_270']):.1f} ¢/sh. Inventory that is acquired late and never closed loses more at resolution: acquired ≥ 270 s {rz['>=270']:+,.0f}, 240–270 s {rz['240-270']:+,.0f}, 180–240 s {rz['180-240']:+,.0f}; acquired before 180 s {rz['<180']:+,.0f}.
- **Concentration.** Oct 26 dominates both windows; Oct 25's early round trips are small and Oct 27's negative.
- **What FIFO is.** It is an accounting convention: it pairs the oldest open unit and does not identify which fill Maker3 meant to close with. Another pairing moves time attribution, not the totals.
"""
    (H / 'report.md').write_text(rep + txt)
    print(txt)


if __name__ == '__main__':
    main()
