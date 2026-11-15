# Information available to the wait/post decision

This catalog accompanies `research/pf_entry_information.py` and the frozen plan
`20261115_entry_information_plan.md`. The complete machine-readable column list
is emitted as `inventory.csv` and `inventory.json` in the study report. There are
325 named columns, including aliases and derived quantities; this is not 325
independent sources of information. No new live feed has been collected.

## Existing market inputs

| Names | Information and units |
|---|---|
| `bn_r25/50/100/250/500/1000/3000` | Trailing Binance mid log return, basis points; suffix is milliseconds |
| `bn_imb` | Current Binance best-size imbalance, `(bid size - ask size)/(sum)` |
| `bn_flow250/1000/5000` | Signed Binance aggregate-trade quantity in trailing milliseconds |
| `bn_ntr1000` | Binance aggregate-trade count over the last second |
| `bn_age` | Milliseconds since the latest usable Binance book observation under the recorded clock |
| `pf_spread`, `pf_mid` | Predict spread and mid, cents/ticks |
| `pf_micro_dev` | Size-weighted microprice minus mid, cents |
| `pf_imb1`, `pf_imb3` | Displayed size imbalance at the best or first three levels |
| `pf_lbs`, `pf_las` | `log1p` of displayed best bid/ask shares |
| `pf_age` | Milliseconds since the latest known Predict frame |
| `pf_nfr1000` | Number of received Predict frames over the last second |
| `pf_dm250/1000/5000` | Trailing known Predict mid change, cents |
| `pf_pend_bid1000`, `pf_pend_ask1000`, `pf_pend_tot5000` | Received pending-size increment sums; a flow proxy, not an exact execution tape |
| `t_rel` | Seconds since contract start |
| `sigma` | Pre-window estimate of one-second Binance log-return volatility |
| `basis_bps` | Trailing reference/Binance basis estimate, basis points |
| `gap` | Model-implied fair probability in cents minus known venue mid; missing while the start reference is unknown |

Seventeen `side_` aliases multiply returns, flows, imbalance, microprice and
midprice changes by +1 for bid or -1 for ask. They contain the same raw market
information. Bid and ask fits are separate.

## Existing candidate, order and portfolio state

The original fields are `side_sign`, `rung`, `price`, `depth`, `spread`, `age`,
`quantity`, `known_filled`, `ahead_known`, `initial_ahead_known`, `shown_known`,
`size_better`, `queue_fraction`, `known_position`, `own_exposure`,
`other_exposure`, `capacity_left`, `time_remaining`, and `frame_age`.
Prices/distances are cents or ticks, quantities are shares, and times are ms.
At an empty entry slot, some are structural zeros or duplicates: there is no
resting candidate age yet, and initial/current queue estimates equal displayed
size. A column's presence therefore does not mean it adds independent signal.

`side_position` expresses known position relative to the entry side.
`idle_age_ms` and `side_price_change_while_empty` record waiting history.
`time_remaining` is time to the research valuation cutoff; it is not time to
the contract's actual expiry.

Each side and each of six ladder rungs has the following 16 fields, for 192
columns (`own/opp` is relative to the candidate):

| Field suffix | Information |
|---|---|
| `orders` | Number of client-live orders at the slot |
| `remaining_lots`, `pending_place_lots`, `pending_cancel_lots`, `filled_lots` | Client-known quantities divided by the nominal 30-share order size; fills only after their notice |
| `price_gap_mean/min/max` | Order-price distance from its side's known best |
| `oldest_age_log`, `newest_age_log`, `active_age_log` | `log1p(age_ms/50)` for order/submission/expected-activation state |
| `shown_lots`, `queue_lots` | Displayed size and public queue proxy at held prices, divided by 30; unknown depth stays missing |
| `queue_unknown` | Number of held orders whose queue proxy is unavailable |
| `empty_age_log` | `log1p(empty_ms/50)` while the slot is empty |
| `generation_log` | `log1p(generation)` |

Each side's book adds nine fields: trailing liquidity scale, best-price age,
frame age, best imbalance, and relative sizes at levels 1-5. The scale is the
known trailing 30-second median best size; it is not a whole-day statistic.
Three flags finish the original 284 columns: `start_known`, `own_empty_count`,
and `opp_empty_count`.

## Additional information measured in this study

| Family | Columns | Meaning |
|---|---:|---|
| Recovered upstream dynamics | 4 | `own_d250`, `opp_d250`: changes in log best size; `d_bnimb250`: Binance imbalance change; `act_rel`: log trade count relative to its trailing 30-second baseline |
| Book geometry | 18 | Each side's price gaps to ranked levels 2-5, and displayed lots within 1-5 ticks of its best |
| Level dynamics | 12 | Each side's best-price movement; net change, cumulative shrink and cumulative growth at the candidate's exact price; own imbalance change. All use received histories over 250/1000 ms |
| Implied fair-value dynamics | 7 | Side-aligned fair-value and gap changes over 250/1000 ms, sensitivity per Binance basis point, distance from 50, and true seconds to contract expiry |

Candidate shrink/growth comes from successive visible snapshots; it cannot
identify whether a reduction was trading, cancellation or another book update.
Missing coverage anywhere in the candidate-price segment stays missing.
Fair-value history remains masked until its reference is known. Sensitivity
and distance are transforms of current inputs, not independent market feeds.

## Available records excluded from decision features

- Venue trade legs and inferred execution anchors: used by replay/labels.
  This study has not reconstructed a causal received-trade feature stream.
- Legacy guard and other model scores: excluded because their training lineage
  is not verified for the entry folds. Raw market fields remain usable.
- Future frames, settlement outcome and future fills: outcomes only.
- Simulator's actual queue and unreceived fills/cancel acknowledgements: hidden
  state, never substitutes for client-known information.

The study has no exact public queue, order identities for other makers, deeper
Binance L2, or causal venue-trade tape as additional inputs. The historical
Binance clock assumes event-to-known latency. Signal usefulness must ultimately
survive realistic clock/queue assumptions.

## Reading the measurements

The per-feature screen measures marginal association, including nonlinear
train-defined quintile profiles. Shared information and structural aliases can
make several columns appear useful for the same reason. The family comparison
asks whether histories/geometry improve a fixed learner beyond current book,
contract and order state. Family improvements are not additive causal effects.
All dates are consumed development data; a favorable entry here is a research
lead, not a validated trading signal.
