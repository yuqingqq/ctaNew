# ARRIVAL-QUEUE-DECOMPOSITION-v1 — declaration

2026-11-22, session aea5bfb9. This round implements the ONE next priority in the USER's review-only commits:
- `aad19914`: the horizon protocol;
- `ed7c3db5`: `reviews/automatic/20261122_horizon_conditions/report.md`;
- `13a70491`: `reviews/automatic/20261122_queue_conditions/plan.md`, which adds queue position as a mandatory axis.

**The question:** on the neutral BASELINE live snapshots, when do the fills of a still-existing generation arrive after a cancellation could take effect? How much value do they carry at each post-fill horizon? How does that differ by queue position, jointly with the declared Binance-pressure bands?

The diagnostic is saved data only:
- no replay, tape load, fit, threshold, rule or new day;
- the days are the consumed Oct 24–27 development days, with the 19 complete markets;
- it is exploratory description, with no skill, guard or promotion claim;
- the stopped POSITIVE-PRESSURE-GUARD-v1 stays stopped, and its resource decision stays with the USER.

The specification is the header of `predictfun/research/pf_ladder_arrival_queue.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before the run)

| Item | File |
|---|---|
| Builder (population, arrival intervals, horizons, queue fields, progress, venue join, views, tables, reconciliation) | `pf_ladder_arrival_queue.py` |
| Selftest (groups 1–7) | `pf_ladder_arrival_queue_selftest.py` |
| Mutation harness | `pf_ladder_arrival_queue_mutants.py`. The final run is in `logs/mutants_final`. |
| Input identities, edges, horizons, intervals, bands, views, cap | `manifest.json` (write-once) |
| Independent checks | `aq_checks.py`: families A, B, F, C, R, S, I and a falsifier |
| Reading rules | `tools/read_arrival_queue.py` (the READINGS below) |
| Launcher | `launch_arrival_queue.sh` |

## Population and joins

- **States.** Every `kind == live` row of the committed audit states (`ladder_audit_20261121/outputs/states.csv.gz`, sha256 `ec2eaf7c…`):
  - 4,840 rows, one sampled live review per generation;
  - no-fill rows are kept;
  - label `BASELINE_neutral_live`.
- **FULL-policy states are not present** and none enter.
- **Fills.** The generation's own fills come from the pressure receipt's disabled arm (= the original ladder). The parent day records are checked against the committed copies and against the residual manifest's sha256s.
- **Identity.** A snapshot's gid, side, rung and size must equal its ledger order's, or the run refuses. The states carry no price, so the same price is the same gid: an order's price is fixed for its life.

## Arrival × horizon

- **Effect.** e = max(d + 37, submit + 95) before activation, else d + 37 (the audit's clock).
- **Window.** The generation's fills with t_f ≥ e, split into four intervals:

  | Interval | Range |
  |---|---|
  | I0 | [e, e + 250 ms) |
  | I1 | [e + 250 ms, e + 1 s) |
  | I2 | [e + 1 s, e + 5 s) |
  | I3 | [e + 5 s, the generation's end) |

- **Fills outside the window** are counted, not dropped: pre_effect [d, e) and before_decision (< d).
- **Generation end.** G = min(removal, full fill, END), with its cause.
- **Follow-up status per interval:**
  - not_at_risk;
  - at_risk_full;
  - removed_in_interval;
  - filled_out_in_interval;
  - end_censored.

  Each carries its at-risk ms. The union's status is the end cause, or not_at_risk.
- **Horizons.** For h = 0 (the fill-time mid, i.e. the execution edge), 50 ms, 100 ms, 1 s, 5 s and 30 s, the value is the side-signed q × (mid(t_f + h) − p).
  - The value is missing when the saved mid is null or t_f + h > END. Missing never becomes 0.
  - END = q × (mark − p).
  - Fill-level positive and negative parts are kept apart from the state-level sums.
- **Cohorts.** The common cohort of a row has every horizon valid for every fill. The native cohort is per horizon.
- **Reconciliation.** The union of the intervals must reproduce the audit window per state, or the run refuses:
  - the fill count;
  - the quantity;
  - END;
  - the +50 ms, +100 ms and +1 s values;
  - the pre-effect count.

## Queue axis (USER plan item 1)

- **Saved fields:**
  - Q = fp_queue, the client's running-minimum estimate of shares ahead;
  - R = fp_remaining, the known remaining after the +100 ms notice;
  - Q/R when R > 0, and o_q_rel;
  - size, and known filled = size − R;
  - the age from submission and the age since activation;
  - o_depth, the birth rung, the price and pf_age (saved; no staleness threshold is chosen).
- **Bands** are analysis bins, not thresholds:
  - invalid_R (R not finite or ≤ 0);
  - unknown (Q not finite or < 0; never front);
  - front (Q = 0);
  - within_R (0 < Q ≤ R);
  - beyond_R (Q > R).
- **Views** (nine, exploratory multiplicity):
  - the USER's five: pressure_flow, depletion, depth, age, inventory;
  - queue;
  - queue_pressure (queue × Binance band);
  - queue_pressure_flow (the declared Predict-flow secondary split);
  - progress (below).

  Their band counts go in `audit.json`. Sparse cells stay sparse, and nothing is merged.

## Progress and queue quality (plan items 4–5)

- **Progress.** Only from a saved receipt-time state of the same gid; never from a lifetime summary or the venue.
  - A decision before activation is not_yet_born.
  - From the first-review snapshot:
    - ok, with dQ = Q − Q_first;
    - first_review_is_decision;
    - future_state_refused;
    - no_first_review_saved;
    - order_mismatch;
    - unknown_queue.
  - From the entry snapshot (the pre-activation POST state, i.e. the displayed size the order joined): the same statuses, giving dQ_from_entry.
  - Recent 250 ms depletion is available only when the first review sits exactly at d − 250. Otherwise it is not_yet_born_at_t_minus_250 or unavailable_no_saved_state.
- **The progress view.** The first-review status; its ok rows are split into advanced (dQ < 0), unchanged (= 0) and increased (> 0).
- **Venue queue.** The disabled arm's record [t, ahead, rem] at the gid's first live review, read after that tick, is set beside the first-review snapshot's client Q and R at exactly that tick.
  - It is explanatory only: never a band, predicate or view, and never a nearest-time substitute.
  - The known-minus-venue remaining must equal the fills in (t − 100, t] (notice_gap_agrees).
- **Known remaining.** Recomputed as size minus the fills with t_f + 100 ≤ d, and checked against fp_remaining.

## READINGS (computed by `tools/read_arrival_queue.py`; nothing is selected)

Shared rules:
- **Sparse:** a cell with fewer than 20 states with a fill.
- **Days:** 4 UTC days < 5, so no intervals and no significance claims; point estimates only.

| Reading | Content |
|---|---|
| R1 | Per interval and the union over all 4,840 states: opportunities, at-risk, any-fill rate, fills, shares, END, and common and native net / per share at every horizon.<br>Computed twice (intervals.csv directly; cells.csv's queue view summed over its bands); they must agree, or the readings refuse.<br>Predicates: net < 0 at 1 s; net at 30 s below net at 1 s; the fraction of the union-common net and shares by arrival interval. |
| R2 | Queue band × interval (common): opportunities, at-risk, any-fill rate, shares, partial / filled-out, per share and per opportunity at 0 / 1 / 5 / 30 s, fill-level positive / negative per share at 1 s, END, sparse.<br>Predicate: the per-share order of front / within_R / beyond_R at 1 s and 30 s (insufficient_support if any is sparse). |
| R3 | Queue × Binance band × interval (common), with the same columns.<br>Contrasts within each queue band and interval: q5 − non_adverse and positive_below_q5 − non_adverse per share, at 1 s and 30 s, only between non-sparse cells (else insufficient_support).<br>Per day the same contrast where both day cells are non-sparse, plus the count of days sharing the pooled sign.<br>R3b: the Predict-flow secondary split, as a table only. |
| R4 | Progress band × interval (common), with the advanced / unchanged order at 1 s and 30 s.<br>R4b: the USER's pressure_flow view by interval. |
| R5 | Queue quality: venue availability; client − venue at the first review (exact agreement, above / below, p10 / p50 / p90, median absolute difference, by side); notice-gap and known-remaining agreement. |

Interpretation rules (USER plan):
- A difference only in fill rate or filled quantity is not adverse-movement discrimination.
- A better queue gives access to both good and bad fills, so fill-level positive and negative parts are reported.
- Window fills are fills under the baseline continuation, not an always-kept fill rate.
- Consumed-day descriptive cells promote no queue rule.

## Checks (`aq_checks.py`, this checker's own code; the module is not imported)

| Family | Checks |
|---|---|
| A | The frozen files committed; the input identities; the USER's table committed; the output hashes. |
| B | Membership: the 4,840 states once each, I0..I3 once per state. |
| F | Per state and interval, everything recomputed from the ledgers, and the union = the audit window. |
| C | Every cells.csv row recomputed by separate groupbys, with the same key set. |
| R | The USER's committed `20261122_horizon_conditions/conditions.csv` (1,431 rows) reproduced from this checker's union rows. |
| S | `audit.json`'s status counts. |
| I | Resources. |

The falsifier has twelve corruption cases. Each must be refused while the clean evidence passes.

## Pre-freeze rehearsal (disclosed)

Before this commit, the unfrozen builder ran twice on the same inputs:
- `data/predict_fun/research/ladder_arrival_queue_v1_20261122/rehearsal` and `…/rehearsal2`;
- the checker ran on both;
- the quiet readings ran on the second.

What was read: only the reconciliation, the population / status counts, the check verdicts and the readings' completion line. **No value column, cell value or reading value was printed or read.**

What it found: the first rehearsal showed that 227 generations end exactly on an interval boundary. The union status then named the earlier interval's at_risk_full instead of the end cause.
- The union status now takes the end cause (selftest group 6, two mutants).
- The progress view was added before this freeze.

The frozen run's outputs supersede both rehearsals and are the only ones reported.

## Resources

- **Round cap: 0.10 core-hours**, including the selftests, both mutation runs and both rehearsals. The ledger is `logs/`, checked by family I.
- **Line cap: 2.0.** As of the second rehearsal, the line is at 1.904, including this round.
- Peak RSS < 1,536 MiB; the rehearsal measured 697 MiB.
- One job at a time via `run_light.sh` with PF_SLOT=claude and absolute paths.
