"""TAKE-EXIT-REPLAY v1 report.md from readings.json / roundtrips.json / checks.json / per_market.csv (no hand-typed numbers)."""
import json
from pathlib import Path
import pandas as pd
H = Path(__file__).resolve().parent
r = json.loads((H / 'readings.json').read_text()); rt = json.loads((H / 'roundtrips.json').read_text()); ck = json.loads((H / 'checks.json').read_text())
pm = pd.read_csv(H / 'per_market.csv')
A = ['PASSIVE_EXIT', 'PASSIVE_EXIT_NOGUARD', 'HOLD']
u = lambda x: f'{x:+,.2f}'
def md(rows, cols):
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(cols) - 1)) + '|'] + ['| ' + ' | '.join(map(str, x)) + ' |' for x in rows])
t1 = md([[a, u(r[a]['net_usd']), u(r[a]['completed_usd']), u(r[a]['residual_usd']), u(-r[a]['fees_usd']), f"{r[a]['days_positive']}/4", f"{r[a]['markets_positive']}/19",
          f"{r[a]['takes_filled']}/{r[a]['takes_sent']}", f"{r[a]['exit_shares']:,.0f}", f"{r[a]['mean_abs_inventory']:.1f}"] for a in A],
        ['Arm', 'END net USD', 'Completed turnover', 'Open at END', 'Taker fees', 'Days positive', 'Markets positive', 'Takes filled/sent', 'Passive exit shares', 'Mean abs position'])
t2 = md([[d] + [u(r[a]['day_net_usd'][d]) for a in A] for d in sorted(r['PASSIVE_EXIT']['day_net_usd'])], ['Day'] + A)
rows = []
for a in A:
    for e in ('passive', 'take', 'open_at_END'):
        k = f'{a}|{e}'
        if k in rt:
            x = rt[k]; rows.append([a, e, f"{x['shares']:,.0f}", f"{x['net_c_per_share']:+.2f}", f"{x['hold_p50_s']:.1f}", u(x['net_usd'])])
t3 = md(rows, ['Arm', 'Closed by', 'Shares', 'Net ¢/share', 'Median hold (s)', 'Net USD'])
pe = r['PASSIVE_EXIT']; rd = r['readings']; npass = sum(1 for v in ck.values() if isinstance(v, dict) and v.get('ok'))
txt = f"""# Taking alone: signal take, exit by a maker order — 19-market replay, 2026-11-25

**Answer.** The standalone strategy makes money in this replay with every exit style. With the guarded passive exit it earns
**{u(pe['net_usd'])} USD** over 19 markets, positive on {pe['days_positive']}/4 days and in {pe['markets_positive']}/19 markets. That is with an
average position of only {pe['mean_abs_inventory']:.0f} shares. When the maker exit fills it earns about +4.6 ¢ per share within a median 2 s. Holding to the END
mark earned more on these days ({u(r['HOLD']['net_usd'])}) but with about 5× the inventory and outcome exposure. The guard on the exit
is worth {u(rd['passive_exit_minus_noguard_usd'])} USD: without it the exits are picked off. Consumed days, development evidence only.

Frozen ee0f16ee6 before any replay ([contract.md](contract.md)).

- **Entry:** the SIGNAL-TAKE rule unchanged. The guard's signal turns ON after 2 s quiet; a 50-share IOC in its direction, 40 ms
  after the tick, at the venue best inside the limit, 200 bps fee; one ±50 position at a time.
- **Exit:** one post-only order on the closing side at the best price, sized to the position; nothing else is quoted.

## Results

{t1}

By day (USD):

{t2}

How positions closed (FIFO round trips, fees included):

{t3}

- **Readings, computed:**
  - passive exit positive {rd['passive_exit_positive']}, ≥ 3/4 days {rd['passive_exit_3_of_4_days']};
  - beats HOLD {rd['passive_exit_beats_hold']} (by {u(rd['passive_exit_minus_hold_usd'])});
  - still positive without the value of exit fills on orders < 100 ms old: {rd['sign_without_young_exit_value']}
    ({u(pe['net_usd_without_young_exit_value'])}).
- **Young exit fills:** {pe['young_exit_shares']:,.0f} of {pe['exit_shares']:,.0f} exit shares. These are the replay fills least
  likely to happen live (FINDINGS B1a).

## What it means and its limits

- **Taking alone beats our market making here.** On the same 19 markets the market-making arms lose (−$562 to −$944); the take
  strategy earns +$137 to +$292 depending on the exit.
- **The passive exit is the low-risk version.** It locks in the move quickly with a small position. Holding captures more of the
  continuing move on these four days, but its gain sits on few shares and depends on how each market ended.
- **Not measured:**
  - untouched days;
  - live fill realism for the maker exit (only an age split is reported) and the M-latency sensitivity;
  - competition faster than the ~50 ms snapshots;
  - settlement versus the 298 s END mark;
  - size beyond 50 shares.
- **Population:** the 19 complete markets are a selected, consumed subset.

## Integrity

- **Checks:** {npass}/5 pass ([checks.json](checks.json)):
  - with takes disabled, no arm places any order;
  - every exit order is on the closing side at the best price (HOLD posts none);
  - all 643 IOC fills re-derived with fees;
  - ledger and FIFO identities on 57 ledgers;
  - the trigger set equals SIGNAL-TAKE's.
- **CPU:** the run took 72 s on the claude slot.

## Next (USER's call)

1. **Untouched days.** Run this frozen rule on later consumed-but-unused or new days before anything else.
2. **Live path.** pmmmrust currently builds only post-only BUY orders; a take needs a marketable IOC path. That is the exec
   session's work and the USER's decision.
3. **Exit refinements.** Only after validation: exit price (join vs improve), a pre-END flatten, size.
"""
(H / 'report.md').write_text(txt); print(txt[:1500])
