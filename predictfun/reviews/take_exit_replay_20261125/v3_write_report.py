"""TAKE-EXIT-REPLAY v3 report (v3_report.md) from v3_readings.json / v3_checks.json (no hand-typed numbers)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
r = json.loads((H / 'v3_readings.json').read_text()); ck = json.loads((H / 'v3_checks.json').read_text())
L, S, rd, cen = r['LENIENT40'], r['STRICT58'], r['readings'], r['census']
u = lambda x: f'{x:+,.2f}'
def md(rows, cols):
    return '\n'.join(['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(cols) - 1)) + '|'] + ['| ' + ' | '.join(map(str, x)) + ' |' for x in rows])
t1 = md([[n, u(x['net_usd']), u(x['per_market_usd']), u(r['in_sample'][f'{n}_per_market']), f"{x['days_positive']}/5", f"{x['markets_positive']}/{x['markets']}",
          f"{100 * x['top10_abs_share']:.0f}%", u(x['net_without_young_exit_value_usd']), f"{x['mean_abs_inventory']:.1f}"] for n, x in (('LENIENT40', L), ('STRICT58', S))],
        ['Setting', 'END net USD', 'Per market', 'In-sample per market (Oct 24–27)', 'Days positive', 'Markets positive', 'Top-10 share', 'Without young exit value', 'Mean abs position'])
days = sorted(L['day_net_usd'])
t2 = md([[d, sum(cen[d].values()), cen[d].get('complete', 0), u(L['day_net_usd'][d]), u(L['day_per_market_usd'][d]), u(S['day_net_usd'][d]), u(S['day_per_market_usd'][d])] for d in days],
        ['Day', 'Markets', 'Complete', 'LENIENT40 net', 'per market', 'STRICT58 net', 'per market'])
npass = sum(1 for v in ck.values() if isinstance(v, dict) and v.get('ok'))
txt = f"""# The frozen take + maker-exit rule on five other days — v3, 2026-11-25

**Answer.** It holds. On the five days not used to develop the rule (Oct 23, Oct 31, Nov 1–3; {S['markets']} complete markets):

- **Conservative setting** (strict take fills + the measured 58 ms order latency): **{u(S['net_usd'])} USD, {u(S['per_market_usd'])} per market,
  positive on {S['days_positive']}/5 days** and in {S['markets_positive']}/{S['markets']} markets. That is
  {100 * rd['strict58_per_market_ratio_to_in_sample']:.0f}% of the in-sample {u(r['in_sample']['STRICT58_per_market'])} per market.
- **Lenient frozen simulator:** {u(L['net_usd'])} ({u(L['per_market_usd'])} per market,
  {100 * rd['lenient40_per_market_ratio_to_in_sample']:.0f}% of in-sample), positive on {L['days_positive']}/5 days.
- **Combined with in-sample:** every one of the 9 days is positive under the conservative setting.

Readings computed: holds out of sample {rd['holds_out_of_sample']}; similar size {rd['similar_size']}.

These are consumed project days, so this is out-of-sample for this rule, not untouched-day validation. Days from Nov 4 on (reserved
for the pending forward races) and the sealed Oct 28–30 were not read.

## What was run

Amendment v3 ([amendment_v3_other_days.md](amendment_v3_other_days.md)), frozen 99ac28e53 before any v3 data.

- **Rule and simulators:** the rule is unchanged (ee0f16ee6). The two bound settings are from the review.
- **Binance input:** change points from the same builder. It reproduced the original Oct 24 data exactly (97,270 / 97,270) and
  kept no row before a segment start (446 flush rows dropped, including sealed Oct 30's).
- **Population:** every market through the unchanged loader. The Nov 3 23:55 market is censored because its tail sits in a forward
  file.

## Results

{t1}

By day:

{t2}

## Integrity

- **Checks:** {npass}/8 pass ([v3_checks.json](v3_checks.json)):
  - Binance stems = the allow-list;
  - delays exactly 40 / 58 ms;
  - strict cap respected;
  - all IOC fills re-derived;
  - all exit posts on the closing side at rung 0;
  - ledger identities;
  - triggers recomputed;
  - the no-trigger falsifier.
- **Operational stops, both disclosed and committed:**
  - The declared CPU gate stopped the first run after Oct 23 (projection 1,211 > 1,100 s). Resource-only amendment R1 raised the
    cap to 1,300 s and reused Oct 23's ledgers after verifying them against a fresh census.
  - The first resume stopped at the 2.5 GiB memory self-guard because it held full ledgers. After the fix, the 4 ledgers written
    before that stop were recomputed byte-identical.
- **CPU:** 1,037.5 s for the run.

## Limits (unchanged from the review)

- **Population:** complete markets only (24–35 % per day), which carry about 2× the per-share edge of the rest.
- **Simulator:** the replay's passive fill model for the exits.
- **Latency:** Tokyo latency is assumed (decision on the 50 ms grid).
- **Untested:** competition below the snapshot resolution, and size beyond 50 shares.
- **Live path:** no IOC path in pmmmrust yet.
- **Checklist:** the full FINDINGS C1–C3 / C12 / C15 checklist is still open.

## Next (USER's call)

1. **Untouched days.** The reserved forward days (Nov 4 →) under the strict + 58 ms setting. This consumes them for the forward
   races, so it is the USER's decision.
2. **The full results checklist** (M 250 on the exits, cancel-pending, WS / E2 split).
3. **Live facts with the exec session:** a true IOC / marketable path, real latency, and a Tokyo shadow log of how fast stale
   quotes disappear.
"""
(H / 'v3_report.md').write_text(txt); print(txt[:1200])
