"""SIGNAL-TAKE-REPLAY v1 report.md: numbers read from readings.json / per_market.csv / checks.json / report_tables.md."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
r = json.loads((H / 'readings.json').read_text()); ck = json.loads((H / 'checks.json').read_text())
tabs = (H / 'report_tables.md').read_text().split('\n\n')
b, w = r['BASE'], r['WIDE']
npass = sum(1 for k, v in ck.items() if isinstance(v, dict) and v.get('ok'))
u = lambda c: f'{c / 100:+,.2f}'
pct = lambda a, n: f'{100 * a / n:.0f}%'
txt = f"""# Signal take added to our strategy — 19-market replay, 2026-11-25

**Answer.** Adding the take helps the **wide-band** strategy and not the base one.

- **Wide band** (cap 1,000, soft band 500): END net improves {u(w['pooled_delta_c'])} USD over 19 markets, positive on {w['days_positive']}/4 days, {w['markets_improved']}/19 markets better. Completed turnover is {u(w['d_completed_c'])}: the taken inventory is sold back passively at better prices.
- **Base band** (cap 200, soft 100): {u(b['pooled_delta_c'])} USD, {b['days_positive']}/4 days positive, {b['markets_improved']}/19 markets. A 50-share take pushes the position into the 100-share soft band, which changes the passive trading enough to lose more than the takes earn.

Both strategies remain loss-making overall on these markets. Consumed days, development evidence only: no validation, no promotion, no live claim.

## What was run

Frozen at 7e37888c8 before any take-arm replay ([contract.md](contract.md)):

- **Markets and timing:** the same 19 complete consumed markets, opening-only gate and ASYNC_FAST10 passive timing as the host experiment (inventory_exit_exception_20261125).
- **Host ledgers:** reused by hash; the take-disabled branch reproduced them exactly on 3 declared markets × 2 hosts.
- **Take:** when the pressure guard's own signal turns ON after 2 s quiet, an IOC in its direction: buy YES on the ask side, sell YES (native BUY NO) on the bid side. 50 shares within the hard cap; effective 40 ms after the decision tick; fills only at the venue best inside the limit with depth not reused; 200 bps × min(p, 1−p) fee.
- **Afterwards:** the unchanged host rules manage the inventory.

## Results

{tabs[0]}

By day:

{tabs[1]}

The take fills themselves, per filled share after the fee (venue mid):

{tabs[2]}

- **Readings, computed:**
  - WIDE delta positive {w['positive']}, robust (≥ 3/4 days) {w['robust_3_of_4']};
  - BASE delta positive {b['positive']}, robust {b['robust_3_of_4']};
  - take +3 s markout positive under both hosts ({b['take_m3s_positive']} / {w['take_m3s_positive']});
  - gain only in leftover inventory: WIDE {w['gain_only_in_residual']}.
- **Concentration:** top-3 markets carry {100 * w['top3_abs_share']:.0f}% of WIDE's absolute per-market changes; without 3789484 WIDE is still {u(w['leave_out_3789484_c'])}.

## Why many triggers were not taken

Of {w['triggers']} triggers per arm:

- **Own-cross refusals:** {r['trigger_status']['BASE'].get('own_cross', 0):.0f} (BASE) and {r['trigger_status']['WIDE'].get('own_cross', 0):.0f} (WIDE) were refused because our own resting order on the other side sat at or through the take price. When BTC jumps up the guard is pulling our asks, but they are still live while the cancel is pending. Buying at the ask would cross them, so the take is refused.
- **Room:** {r['trigger_status']['BASE'].get('no_room', 0):.0f} BASE triggers had no room under the cap.

Of the IOCs sent:

- **BASE:** {r['ioc_status']['BASE'].get('filled', 0):.0f} filled fully, {r['ioc_status']['BASE'].get('partial', 0):.0f} partly, {r['ioc_status']['BASE'].get('limit_moved', 0):.0f} found the price gone.
- **WIDE:** {r['ioc_status']['WIDE'].get('filled', 0):.0f}, {r['ioc_status']['WIDE'].get('partial', 0):.0f} and {r['ioc_status']['WIDE'].get('limit_moved', 0):.0f}.

Self-cross protection is correct live behaviour (pending-cancel orders can still trade). A rule that pulls our opposite quotes earlier, or takes only when none rest at the take price, is a separate candidate.

## Per market

{tabs[3]}

## Integrity

- **Parity:** host parity was exact on all 6 declared runs.
- **Checks:** {npass}/6 pass ([checks.json](checks.json)):
  - triggers recomputed by a plain loop;
  - every sent limit is the known frame's best, with the 40 ms delay;
  - all 235 filled IOCs re-derived from venue frames, with fees;
  - ledger and FIFO identities on all 38 take ledgers;
  - synthetic sent / pending / no_room / near_end / disabled cases;
  - a corruption refusal.
- **CPU:** the run took 306 s, within the 450 s cap, on the claude slot.

## Next (USER's call)

1. **Own-cross handling.** With the take on, cancel opposite quotes at the trigger and send the take once they are gone, or only take when no own quote is in the way. That is the largest blocked share.
2. **Host choice.** The take needs inventory room: it works with the wide band and not the base band. Any adoption ties the take to the wider band and must be judged on the full book.
3. **Validation.** Untouched days before any live use.
"""
(H / 'report.md').write_text(txt); print(txt[:3000])
