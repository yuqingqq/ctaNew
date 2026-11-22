# QUEUE-PRICE-STATE-AUDIT-v1 — declaration

2026-11-22, session aea5bfb9. This is review `72a36de9`'s ONE next priority; that review's target was `75e9531d`.

**The question:** does the apparent queue ordering of ARRIVAL-QUEUE-DECOMPOSITION-v1 persist at a comparable, occupied price level? Or is it mostly composition, of price state and coverage?

Scope:
- the same 4,840 neutral BASELINE live states;
- saved data only: no new threshold, model, day, venue-queue decision input or replay;
- consumed Oct 24–27; descriptive, with no skill or promotion claim;
- POSITIVE-PRESSURE-GUARD-v1 stays stopped.

If the comparable cells are thin, the answer is that this saved population cannot answer the question. A request for more data, or a counterfactual queue-value experiment, is a separate USER decision.

The specification is the header of `predictfun/research/pf_ladder_queue_price.py`. Where this text and the header differ, the header governs.

## Frozen in this commit (before the run)

| Item | File |
|---|---|
| Builder (inputs, exact-key join, groups, views, union, cells) | `pf_ladder_queue_price.py` |
| Selftest (groups 1–5, synthetic rows only) | `pf_ladder_queue_price_selftest.py` |
| Mutation harness | `pf_ladder_queue_price_mutants.py`: 23 / 23 caught (`logs/mutants_try2`). The `try1` run was a harness failure, a missing `run` hook, and judged nothing. |
| Input hashes, horizons, support minimum, views, cap | `manifest.json` (write-once) |
| Independent checks | `qp_checks.py`: A / B / U / G / C / P / I and a 10-case falsifier |
| Reading rules | `tools/read_queue_price.py` (the READINGS below) |
| Projection gate | `tools/projection_qp.py`, write-once. The run stage refuses unless it fits. |
| Launcher | `launch_queue_price.sh` |
| Review verification (counts only, before any economics) | `tools/verify_review_findings.py` and its log |

## Inputs, join and groups (declared before any economics)

- **Inputs**, each hash-bound:
  - ARRIVAL-QUEUE's committed states and intervals copies (raw sha256 = its committed `audit.json`);
  - the four original LADDER_REST feature parquets (= the committed day records' `sha256['feat']`).
- **Join.** Each state matches exactly one live feature row at the same (day, m, gid, t, side); there is no nearest match. fp_queue / fp_remaining / o_depth must equal the state's Q / R / o_depth exactly, or the run refuses.
- **price_state** (o_depth = ticks behind the received best):
  - inside_best (< 0);
  - at_touch (= 0);
  - behind_best (> 0);
  - unknown.
- **display** (fp_shown = the external size displayed at the order's price in the last received frame):
  - shown_zero;
  - shown_positive;
  - shown_invalid (< 0);
  - shown_unknown.
- **Queue band:** ARRIVAL-QUEUE's rule, re-derived, and required to equal its saved band.
- **Q ≤ shown is not imposed.** Exact depth, rung and price are exported.
- **Views:**
  - price_display;
  - price_display_queue. **The PRINCIPAL comparison is at_touch|shown_positive by queue band**, by side and day;
  - behind_depth_queue: behind_best & shown_positive by exact depth and band, a separate diagnostic that is never pooled with the principal cells. Other states are labelled `outside` and kept;
  - principal_pressure: the principal states by queue band and ARRIVAL-QUEUE's frozen Binance band. This is a support census.

## Cells and support (review finding 2)

**Granularity.** There is one row per view × band × interval (I0..I3 and the union) × day (and all) × side (and all) × cohort (native / common).

**Counts.** All-state counts are named `all_*`:
- opportunities, at-risk, filled states, fills, shares;
- partial, filled-out, every follow-up status;
- the median time to END at the decision, and the states with < 30 s to END.

`censored_from_common` = all-state opportunities − common-valid states.

**Per horizon** h = 0, 1, 5, 30 s:
- valid, valid at-risk and valid filled states;
- valid shares;
- the valid any-fill rate;
- net;
- state and fill positive / negative parts;
- per opportunity, per share;
- `sparse`.

**Support rule:** a value is evaluable only when its own cohort has ≥ 20 filled states valid for it. Common c/share is never multiplied by all-state shares.

## READINGS (`tools/read_queue_price.py`; computed, nothing selected)

| Reading | Content |
|---|---|
| Q1 | The occupancy census: states by price_state × display × queue band. The front by price_state × display supersedes the prior receipt's occupancy count (finding 1). |
| Q2 | The principal comparison: the common-cohort per share at 1 s and 30 s per interval and the union.<br>The order of front / within_R / beyond_R when all three are evaluable, else insufficient_support.<br>days_evaluable, days_same_order, and `same_order_all_4_days`, true only when all four declared days are evaluable and match (finding 2's predicate correction). Also sides_same_order. |
| Q3 | The all-state queue order, pooled over price / display at the same support rule.<br>`q3_persists_at_touch_occupied`: the principal union order equals the pooled order, or insufficient_support. |
| Q5 | The behind-best occupied exact-depth diagnostic: the order per depth. |
| Q6 | The pressure support census: states and valued filled states per principal band × Binance band, and the evaluable contrasts (q5 / positive_below_q5 vs non_adverse, both evaluable at 1 s). |
| Q7 | Principal censoring: censored_from_common, the median time to END and the states with < 30 s to END. |
| Q8 | Client − venue queue at the first review by price_display group: n, within 1 share, below / above by ≥ 1 share. Explanatory only: simulator first-review diagnostics, never a band or predicate. |

Point estimates only (4 UTC days < 5).

## Corrections bundled (review 72a36de9)

1. **Occupancy.** Q = 0 is not occupancy. The current displayed size `fp_shown` sits beside Q and depth.
   - Checked before any economics (`tools/verify_review_findings.py`, counts only): **of the 789 front states, 458 show zero displayed size and 331 positive; of the 343 behind-best front states, 198 are occupied; all 133 at-touch front states are occupied.**
   - The prior receipt's "656 alone at their price" is superseded by these counts.
2. **Denominators.** Every table names its all-state counts and its cohort counts, and support is screened on the value's own filled states. The misnamed `r3_evaluable_negative_all_days` of the prior reader (`read_arrival_queue.py:142`) is superseded by the explicit day counts and `same_order_all_4_days`. Its value was 0, and is unaffected.
3. **Checker coverage.** `qp_checks.py` re-derives every reused field, including client_minus_venue from the audit first rows and the ledgers' venue records at the exact tick. It verifies the ledger gzip bytes against their day records. It recomputes every cell column (rates, per share, sparse) and every reading. Its falsifier includes the review's two demonstrated escapes.
4. **Timing wording.** An arrival interval is not the latest actionable signal time.
   - A fill in I0 = [d + 37, d + 287) can still be excluded by a later 50 ms review: e.g. a request at d + 100 takes effect at d + 137.
   - The prior "22.5 %" is a share of net 1 s loss on the union-common cohort, not a timing-unavoidable fraction.
   - The prior receipt's sentence implying otherwise (`report.md:68`) is superseded.

## Pre-freeze rehearsal (disclosed)

Before this commit, the unfrozen builder ran once on the same inputs (`…/ladder_queue_price_v1_20261122/rehearsal`). The quiet readings and the checker ran on it.

Only these were read:
- the state / principal / cell-row counts;
- the population counts in the audit summary: Q > shown in 0 states; the principal pressure census of state counts;
- the check verdicts.

**No value column, cell value or reading value was printed or read.**

The first checker rehearsal crashed on a checker bug. `A.lt`, a pandas method, was read instead of the column. It was fixed and vectorized; the second checker rehearsal passed every check except A_committed (pre-commit), with 10 / 10 falsifier cases refused. The frozen run supersedes the rehearsal.

## Resources

- **Cap.** The whole batch is capped at **≤ 0.020 core-hours** (review 72a36de9). That includes the review verification, the selftests, the mutation runs, the rehearsal and the frozen stages. The ledger is `logs/`; family I checks it.
  - At the rehearsal checks, this round had used 0.0103.
  - The line stood at 1.9725 of 2.0.
- **Projection gate.** `tools/projection_qp.py` projects the frozen stages at 1.25 × their measured rehearsal cost and is written once. The run refuses if the round would exceed 0.020 or the line 2.0.
- One job at a time via `run_light.sh` with PF_SLOT=claude and absolute paths.
