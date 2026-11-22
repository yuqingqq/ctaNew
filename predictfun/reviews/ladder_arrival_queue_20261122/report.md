# ARRIVAL-QUEUE-DECOMPOSITION-v1 — receipt

2026-11-22, session aea5bfb9.
- **Frozen:** `29869269`, before the run.
- **Specified by:** the USER's review-only commits:
  - `aad19914`;
  - `ed7c3db5`: the horizon report's ONE next priority;
  - `13a70491`: the queue amendment.
- **Run:** selftest → run → checks → readings, all through `launch_arrival_queue.sh`.
- **Coverage:** consumed Oct 24–27, 19 complete markets, saved data only.
- **Scope:** a descriptive diagnostic of the neutral BASELINE live states.
  - It is not predictive skill, a guard, an avoided-loss estimate or a validation.
  - It measures no model skill.
  - No FULL-policy state enters.
  - 4 UTC days < 5, so all figures are point estimates with no intervals.

## Integrity

| Item | Result |
|---|---|
| Membership | 4,840 live states, each exactly once; 19,360 interval rows (I0..I3 per state) |
| Reconciliation | Interval union = the audit window on every state: fills, shares, END, +50 ms / +100 ms / +1 s, pre-effect count. 6,992 window fills, 74,887.9 shares |
| Independent checks | `aq_checks.py`: A / B / F / C / R / S / I all pass. C covers 8,760 cell rows, 0 bad. R reproduces the USER's `conditions.csv` (1,431 / 1,431 rows), so the five earlier views are identical. Falsifier 12 / 12 corruptions refused |
| Notice latency | Known remaining = size − fills notified by d on 4,840 / 4,840. At the first review, the known-minus-venue remaining equals the fills inside the 100 ms notice on 4,840 / 4,840 |
| Mutation and selftest | Mutation check 37 / 37 caught; selftest groups 1–7 pass |
| Resources | This round 0.077 core-hours (cap 0.10; the frozen pipeline used 0.012). The ladder line is at 1.942 of 2.0. Peak 699 MiB |

Exclusions are statuses, not drops. Per interval [I0 / I1 / I2 / I3]:

| Status | I0 | I1 | I2 | I3 |
|---|---:|---:|---:|---:|
| at_risk_full | 3,340 | 1,774 | 309 | 0 |
| removed_in_interval | 1,213 | 1,163 | 1,136 | 254 |
| filled_out_in_interval | 165 | 214 | 271 | 49 |
| end_censored | 10 | 18 | 6 | 2 |
| not_at_risk | 112 | 1,671 | 3,118 | 4,535 |

Union by generation end: removal 3,993, filled out 699, END 36, and ended before the effect 112. Outside the window: 315 pre-effect fills (in [d, e)) and 1,884 fills before the decision.

## R1 — when fills arrive and what they are worth

Intervals are measured from the earliest effective cancel e (d + 37 ms). Per-share values are in cents, on each row's fixed common cohort, where every horizon is valid for every fill.

| Arrival after e | At risk | Filled states | Any-fill rate | Shares | Fill-time mid | 50 ms | 100 ms | 1 s | 5 s | 30 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0 < 250 ms | 4,728 | 545 | 11.5 % | 17,468.6 | +0.263 | −1.007 | −1.130 | −2.192 | −2.366 | −1.527 |
| I1 250 ms – 1 s | 3,169 | 623 | 19.7 % | 23,435.7 | +0.277 | −0.986 | −1.127 | −2.502 | −2.928 | −2.256 |
| I2 1 – 5 s | 1,722 | 614 | 35.7 % | 28,667.8 | +0.486 | −0.801 | −0.965 | −2.118 | −2.106 | −3.788 |
| I3 ≥ 5 s | 305 | 124 | 40.7 % | 5,315.7 | +0.584 | −0.569 | −0.687 | −1.799 | −2.476 | −4.883 |
| Union | 4,728 | 1,702 | 36.0 % | 74,887.9 | +0.379 | −0.886 | −1.030 | −2.227 | −2.446 | −2.893 |

On the union-common cohort (4,591 states), the value splits by arrival interval [I0 / I1 / I2 / I3] as follows:

| Quantity | I0 | I1 | I2 | I3 |
|---|---:|---:|---:|---:|
| Shares | 22.8 % | 30.5 % | 38.7 % | 8.0 % |
| 1 s net | 22.5 % | 34.3 % | 36.8 % | 6.4 % |
| 30 s net | 12.0 % | 23.8 % | 50.7 % | 13.5 % |

Computed predicates:
- **Net < 0 at 1 s:** in every interval and in the union.
- **30 s below 1 s:** I0 false, I1 false, I2 true, I3 true, union true.

Reading:
- Loss is already present 50 ms after the fill in every arrival interval.
- Fills arriving within 1 s of the effect partially recover by 30 s.
- Fills arriving later keep losing to 30 s. They carry 64 % of the 30 s net.
- An action at the decision can only change I0 fills if the condition is already visible at d. That is 22.5 % of the 1 s loss.

## R2 — the queue axis

Union, common cohort. The fill parts are the fill-level positive / negative per share at 1 s.

| Band at the decision | States | Median Q (sh) | Median Q / R | Filled states | Any-fill rate | Shares | 1 s | 5 s | 30 s | Fill parts at 1 s | END (c) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| front (Q = 0) | 789 | 0 | 0 | 238 | 31.9 % | 8,475.7 | −0.930 | +0.272 | +0.319 | +1.054 / −1.984 | −4,070.8 |
| within_R | 1,929 | 30 | 0.31 | 705 | 37.4 % | 33,206.7 | −2.015 | −2.257 | −1.914 | +0.452 / −2.467 | −172,824.0 |
| beyond_R | 2,122 | 239 | 3.16 | 759 | 36.2 % | 33,205.5 | −2.613 | −3.019 | −4.268 | +0.376 / −2.990 | −96,070.4 |

No state is unknown or invalid_R.

By arrival interval (1 s per share, with filled states in brackets):

| Band | I0 | I1 | I2 | I3 |
|---|---:|---:|---:|---:|
| front | −2.804 (128) | +1.105 (86) | −0.234 (50) | sparse (4) |
| within_R | −1.878 (248) | −2.379 (276) | −1.732 (253) | −2.534 (42) |
| beyond_R | −2.377 (169) | −3.143 (261) | −2.604 (311) | −1.559 (78) |

Any-fill rate in I0: front 17.2 %, within_R 13.1 %, beyond_R 8.1 %.

Computed order (R2):
- **Union:** beyond_R < within_R < front, at 1 s and at 30 s.
- **I0 at 1 s:** front < beyond_R < within_R; the front is worst.
- **I1 and I2:** beyond_R < within_R < front.
- **I3:** insufficient_support.

Read from the frozen per-day cells after the run (not a declared predicate), the union 1 s order holds on each day:

| Day | beyond_R | within_R | front |
|---|---:|---:|---:|
| Oct 24 | −1.73 | −1.49 | −0.60 |
| Oct 25 | −2.65 | −2.63 | −1.70 |
| Oct 26 | −2.78 | −2.32 | −0.06 |
| Oct 27 | −3.70 | −2.00 | −1.58 |

At 30 s the order holds on 3 of 4 days. Front has only 11 filled states on Oct 27. The order also holds on both sides at 1 s.

**Material confound: "front" is mostly "alone at the price", not "at the head of a queue".** From the state counts:
- Of the 789 front states:
  - 313 price inside the received best (o_depth < 0, with nothing displayed at that level);
  - 343 sit at an empty level behind the best (o_depth 1–4);
  - only 133 are at the touch.
- within_R and beyond_R are all at or behind the touch (o_depth ≥ 0).

The declared views therefore do not separate queue position from price-level occupancy and distance. The front result is a statement about these bands, not about priority at a shared level. The fill parts show that front fills carry both larger positive and smaller negative parts per share. Fill rates alone are not the difference.

## R3 — queue × Binance pressure

All **0 of 60** declared contrasts are evaluable; every one is insufficient_support.
- Positive Binance pressure is rare in neutral states: 206 of 4,840 (positive_below_q5 95, q5 111).
- Within a queue band there are at most 17 filled states (union) against the rule's 20.

The q5 band as a whole (R4b, q5|PF_adverse) has:
- 40 filled states, −4.779 c/share at 1 s, against −2.130 / −2.324 for the two non-adverse bands;
- 36 of those 40 fill in I0, so its exposure is short-lived: at risk 94 in I0, 8 in I1.

The joint question the plan asked, whether the signal separates value within comparable queue states, cannot be answered on this population.

## R4 — progress

The ok first reviews split into:
- advanced (dQ < 0): 2,344;
- unchanged: 1,970.

No state is increased, because the running-minimum estimate never rises. The median dQ is −6.5 sh.

Union per share:

| Band | 1 s | 30 s |
|---|---:|---:|
| advanced | −2.328 | −2.630 |
| unchanged | −2.139 | −3.859 |
| first_review_is_decision (526 states) | −1.938 | +1.645 |

The order flips with the horizon, so there is no consistent separation. Recent 250 ms depletion is available for only 166 states:
- not_yet_born_at_t_minus_250: 1,583;
- unavailable_no_saved_state: 3,091.

## R5 — client queue estimate vs the simulator's venue queue

The venue record is available at the exact first-review tick for 4,840 / 4,840 states. This is explanatory only: the venue queue is the replay's own model.

Client minus venue (declared quantiles), in shares:

| p10 | p50 | p90 |
|---:|---:|---:|
| −100 | 0.0 | +0.00001 |

Read after the run (not declared):
- within 1 share: 2,903 (60.0 %);
- the client below by ≥ 1 share, i.e. fewer shares ahead than the model: 1,535 (31.7 %);
- above by ≥ 1 share: 402 (8.3 %).

The running-minimum proxy is optimistic in about a third of first reviews. The declared exact-agreement split (1,484 / 1,111 / 2,245 at 1e-9) is dominated by float-level differences.

## Limits and what this does not show

1. **Population.** The population is BASELINE neutral live states, one sampled review per generation. Window fills are fills under the baseline continuation, not an always-kept fill rate. FULL's reached states are not covered.
2. **Queue vs price level.** Queue bands are confounded with price-level occupancy and distance (R2). Nothing here is a queue-dependent rule; consumed-day descriptive cells promote nothing.
3. **Pressure support.** The pressure question within queue states is unsupported: the signal is rarely positive in neutral states.
4. **END** is the variable-horizon pre-expiry mid mark, not settlement.

**Largest evidenced limitation:** the queue axis as declared mixes shares-ahead with being alone at a price, inside the spread or at an empty level behind the best. Only 133 of the 789 front states are at the touch. The contrast that matters for keep-vs-cancel, priority at a shared level, is not isolated.

Files:
- `outputs/` holds the receipt copies (`states`, `intervals` and `cells` .csv.gz, `audit.json`), `readings.json` and the reading tables r1–r4b;
- `checks.json` and `checks_falsifier.json`;
- `logs/`.

The rehearsal outputs under `data/predict_fun/research/ladder_arrival_queue_v1_20261122/rehearsal{,2}` are superseded and unreported.
