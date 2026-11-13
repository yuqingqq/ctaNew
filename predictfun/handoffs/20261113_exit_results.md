# EXIT: do the top makers leave a price just before a trade takes the whole level? (results, 2026-11-13)

Session ce68ebe0. USER: "how do we bridge the gap between our current strategy and to maker's" / "Yes" to the tape check.

Scripts and runs, claude slot:

| Step | Script / driver | Commit | Run |
|---|---|---|---|
| EXIT | `research/pf_exit_check.py` + `run_exit.sh` | 11fb2adc, committed before the first run | 04:31-04:34Z |
| Post hoc robustness | `pf_exit_check.py posthoc` | 5abc1d96, after R392's result, before its run | 04:35Z |

Design (the script header has it in full):
- **Data:** public tape only (real maker legs, real book frames). 9 consumed replay days, first 60 s, WS-timed taker orders.
- **Level-taking trade:** a taker order that filled price p and a worse price, so it took every order resting at p.
- **Presence:** maker x was the only maker filled at p by a taker order that stopped at p, and the next book frame still shows p.
  - So x's order was first in the queue and had size left.
  - x's last fill at p before the level-taking trade must be such a record, at most 1 s earlier.
- **Outcome:** STAYED if x is in the level-taking trade at p, GONE if not.
- **Groups:** top 5 / ranks 6-20 / rest by daily volume.
- **P:** FRONT-PULL's per-day model of "the level goes in (t + 24, t + 274] ms", fitted without the day, at t0 - 24 ms.

Controls passed:
- Synthetic cases.
- Three planted errors, each caught.
- No record at or after t0 is used.
- The null's mean difference is -0.0023.

Statuses:
- 571,941 WS-timed taker orders.
- 38,825 level-taking events.
- 387,892 presence records.
- 3,057 observations ambiguous by E2-timed legs.
- 10,180 observations; 558 with P undefined.

## Answer

| Reading | Result |
|---|---|
| R390 EXIT | NOT-SHOWN. GONE share: top 5 51.3%, ranks 6-20 61.8%, rest 46.4%. Gap-matched top 5 minus field -0.8 points (null p1 / p99 -2.6 / +2.3; higher on 5, lower on 4 days) |
| R392 SIGNAL | SIGNAL-USED. GONE share from the bottom to the top quintile of P: top 5 28.2% -> 79.6% (+51 points); field 42.1% -> 63.1% (+21) |

R391, GONE share by the time between the proof and the trade:

| Group | 0-100 ms | 100-250 ms | 250-1000 ms |
|---|---|---|---|
| Top 5 | 31.4% (242) | 38.7% (796) | 58.6% (2,059) |
| Ranks 6-20 | 43.1% (197) | 47.6% (540) | 68.1% (1,802) |
| Rest | 35.7% (392) | 34.1% (1,321) | 53.6% (2,831) |

Post hoc robustness of R392:
- The top 5's P effect exceeds the field's on 9 of 9 days.
- It holds in every gap bin: +19 / +45 / +60 points against +15 / +13 / +30.
- It rises steadily across P quintiles for the top 5: 28 / 39 / 54 / 63 / 80%.
- For the field it is flatter: 42 / 52 / 54 / 51 / 63%.

R393 (descriptive), volume share and shares per maker leg:

| Trade kind | Top 5 | Ranks 6-20 | Rest |
|---|---|---|---|
| Level-taking (at the taken price) | 33.9%, 51.6 sh/leg | 36.1%, 27.6 | 30.0%, 14.0 |
| Sole-maker stop (first in queue) | 42.4%, 12.7 sh/leg | 23.9%, 9.3 | 33.7%, 8.6 |
| Other stops | 36.8%, 35.9 sh/leg | 36.1%, 21.0 | 27.0%, 11.3 |

## What it means

1. **The top makers do not leave more often; they leave at the right moments.**
   - Overall, about half of front orders are gone before the level is taken, for every group.
   - The top 5 stay when our P is low (72% stay in its bottom quintile, against 58% for the field) and leave when it is high (80% gone,
     against 63%).
   - This is the keep / pull pattern we were trying to build, and it follows the public signal we already compute. It agrees with
     MOVE: no missing information.
2. **The choice is made 100-1000 ms before the trade, not in the last 100 ms.** Within 100 ms of the trade their P effect is about
   the field's (+19 against +15 points). Their edge here is the decision, not last-instant speed.
3. **They hold the front, with size.**
   - They are 42% of first-in-queue volume against 34% of the volume taken when a level goes.
   - When a level is taken while they are in it, their orders there are large (52 sh per leg against 28 and 14).
4. **For us:** FRONT-PULL's signal points the same way as the top makers' exits. What we lack is their position. In the replay (rule
   M) we reach the front only when others leave, which is right before the level goes. The next information is how they get and keep
   that position.

## Limits

- Consumed days, first minute. Public tape: theirs real, nothing of ours.
- Presence needs a sole-maker stop, so the population leans to front orders and small trades.
- A maker can hold several orders at one price.
- GONE also counts cancels for other reasons and moves to another price.
- R392 is not gap-matched; the post hoc split covers that.
- P is our model (fitted on these days, without the scored day).

Outputs: `data/predict_fun/research/exit_20261113/` (exit.json / .txt, exit_posthoc.json / .txt, obs_*.parquet, vol_*.parquet, day_*.json,
logs); copies (json / txt) in `reviews/exit_20261113/`.
