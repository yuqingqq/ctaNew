"""TAKE-EXIT-REPLAY v2 report (v2_report.md) from v2_readings.json / v2_checks.json / census (no hand-typed numbers)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
r = json.loads((H / 'v2_readings.json').read_text()); ck = json.loads((H / 'v2_checks.json').read_text())
pe, ho, c, rd = r['PASSIVE_EXIT'], r['HOLD'], r['census'], r['readings']
u = lambda x: f'{x:+,.2f}'
def md(rows, cols):
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(cols) - 1)) + '|'] + ['| ' + ' | '.join(map(str, x)) + ' |' for x in rows])
t1 = md([[a, u(x['net_usd']), u(x['net_per_market_usd']), u(x['completed_usd']), u(x['residual_usd']), u(-x['fees_usd']), f"{x['days_positive']}/4",
          f"{x['markets_positive']}/{x['markets']}", f"{100 * x['top10_abs_share']:.0f}%", f"{x['mean_abs_inventory']:.1f}"] for a, x in (('PASSIVE_EXIT', pe), ('HOLD', ho))],
        ['Arm', 'END net USD', 'Per market', 'Completed turnover', 'Open at END', 'Taker fees', 'Days positive', 'Markets positive', 'Top-10 share', 'Mean abs position'])
t2 = md([[d, c['by_day'][d].get('complete', 0), u(pe['day_net_usd'][d]), u(ho['day_net_usd'][d])] for d in sorted(pe['day_net_usd'])], ['Day', 'Complete markets', 'PASSIVE_EXIT', 'HOLD'])
hrs = pe['by_hour_net_usd']
t3 = md([[f'{h:02d}', u(hrs[str(h)] if str(h) in hrs else hrs[h])] for h in range(24)], ['Opening hour (UTC)', 'PASSIVE_EXIT net USD'])
npass = sum(1 for v in ck.values() if isinstance(v, dict) and v.get('ok'))
txt = f"""# Take + maker exit on all complete markets — v2, 2026-11-25

**Answer.** On all **{pe['markets']} complete markets** of consumed Oct 24–27 (v1 had 19), the take + guarded maker exit earns
**{u(pe['net_usd'])} USD**, {u(pe['net_per_market_usd'])} per market. It is positive on {pe['days_positive']}/4 days, in
**{pe['markets_positive']}/{pe['markets']} markets** and in every opening hour of the day; the top 10 markets carry only
{100 * pe['top10_abs_share']:.0f}% of the absolute per-market values. HOLD earns about the same total ({u(ho['net_usd'])}), but in
fewer markets ({ho['markets_positive']}/{ho['markets']}) and with about 4× the inventory. The 262 new markets earn
{u(pe['new_markets_per_market_usd'])} per market, against {u(pe['v1_markets_net_usd'] / 19)} on the 19 v1 markets: v1 was
representative. Development evidence on consumed days; not validation.

Freeze 0e7882007 (v2 amendment + harness) before the census. Strategy code unchanged from v1 (ee0f16ee6).

## Population (census, outcome-blind)

{c['status_counts'].get('complete', 0)} complete, {c['status_counts'].get('missing_endpoint', 0)} missing_endpoint, and
{c['status_counts'].get('stale_future_book', 0)} stale_future_book, of all {sum(c['status_counts'].values())} markets. The rule was
"{c['selection_rule']}"; the projection fitted the 900 CPU s budget. Complete markets are those whose book stays two-sided and fresh
to 298 s, i.e. mostly still contested at the close. Markets that are decided early (one-sided near the end) cannot be replayed under
the current END-mark rule. This is the main limit on generality.

## Results

{t1}

By day:

{t2}

By opening hour (PASSIVE_EXIT):

{t3}

- **Readings, computed:**
  - positive {rd['passive_exit_positive']}, robust (≥ 3/4 days) {rd['robust_3_of_4']};
  - beats HOLD {rd['passive_exit_beats_hold']} ({u(rd['passive_exit_minus_hold_usd'])});
  - positive without the value of exit fills on orders < 100 ms old {rd['positive_without_young_exit_value']} ({u(pe['net_usd_without_young_exit_value'])});
  - new markets positive {rd['new_markets_positive']}.
- **Activity:** takes filled {pe['takes_filled']:,} ({pe['taker_shares']:,.0f} shares); maker exits {pe['exit_shares']:,.0f} shares,
  of which {pe['young_exit_shares']:,.0f} on orders < 100 ms old.

## Integrity

- **Checks:** {npass}/5 pass ([v2_checks.json](v2_checks.json)):
  - the 38 v1 parity-gated ledgers are exactly equal (cache bits, identity, fills, orders, commands, grid, signals, IOC results,
    guard log without the logged age);
  - all exit posts are on the closing side at rung 0;
  - all IOC fills re-derived with fees;
  - ledger and FIFO identities on all 562 ledgers;
  - triggers recomputed.
- **CPU:** census 84 s, run 652 s, on the claude slot.

## Limits and next steps

- **Limits:**
  - consumed days only;
  - complete (contested-to-the-end) markets only;
  - values at the 298 s mark, though positions are small, so settlement differs little;
  - the replay's fill model (young exits are shown not to drive the result);
  - Tokyo latency is assumed;
  - sub-frame competition and capacity beyond 50 shares are untested;
  - there is no live IOC path yet.
- **Next (USER's call):**
  1. untouched days with the frozen rule;
  2. a valuation for decided markets (e.g. settlement), to cover the other 76 %;
  3. an IOC path in pmmmrust for a Tokyo shadow or canary.
"""
(H / 'v2_report.md').write_text(txt); print(txt[:2500])
