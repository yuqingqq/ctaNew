# USER amendment — reconstruct the bulk of behavior with explicit estimates

2026-11-22. USER: "if we don't have enough information we can only estimate
in some cases, but we need to figure out most of it".

This governs the depth/coverage objective in `plan.md`. Exact hidden command
timestamps are not a prerequisite for a useful reconstruction. The eight
examples are an initial schema/control check, not the final research result
or a separate mandatory review gate. Bundle that check, the feasible broader
measurement and material issues into one round under a declared resource cap.
The current frozen discrimination experiment is unchanged.

## Objective

Explain the bulk of the makers' observable trading through coherent queue
and order-lifecycle histories: how they accumulate priority, retain it through
partial fills, change prices or withdraw, refill, and complete two-sided
turnover. Infer plausible behavior where direct observations are missing;
quantify where assumptions can change the answer. Do not stop merely because
an exact submission/cancellation millisecond cannot be recovered.

## Reconstruction method

1. Combine every relevant source already on disk: full order identity,
   matches, displayed-depth changes, intervening trades, subsequent same-hash
   fills, other makers at the level and both sides' executions. Audit raw
   fields before accepting gaps inherited from old extraction tables.
2. Enforce hard consistency constraints first: quantities, order price/side,
   execution sequence, supported timestamps, visible-depth limits and the
   declared queue/matching assumptions. Missing original sizes, orders never
   filled and depth outside the feed remain explicit unknowns.
3. For hidden quantities/times, reconstruct a range of compatible histories
   under a small predeclared set of plausible assumptions. A scored best-fit
   trace may illustrate the result, but retain alternative traces and their
   consequences. Prior assumptions must not be selected to maximize maker
   profit or to make their behavior agree with our preferred hypothesis.
4. Classify each inference as **observed**, **constrained estimate**,
   **assumption-sensitive**, or **unresolved**. Apply this per field/event,
   not as one confidence label for an entire maker. Report time/queue/size
   ranges. A consistency score is not a calibrated probability; numerical
   confidence probabilities require separate calibration evidence.
5. Use later evidence to reconstruct historical presence when justified,
   while keeping it out of any later causal policy's inputs. For a hidden
   decision window, examine conditions across the window; a feature measured
   just before a later trade may already be after the decision. Action
   associations that reverse across plausible times are assumption-sensitive.

## Coverage and economic usefulness

The first broader receipt must quantify, separately for each maker/day/side:

- Exact identity/fill reconciliation coverage, with all exclusion statuses.
- Fractions of observed filled orders, executions and executed shares whose
  posting, persistence/queue state and removal explanation are respectively
  observed, constrained, assumption-sensitive or unresolved. These are
  different denominators: a mapped order is not an identified cancellation.
- Distributions of inferred posting/removal-window widths and queue ranges;
  do not hide wide windows behind a count of reconstructed traces.
- Repeated-fill volume, completed turnover and residual exposure attributable
  to behavior patterns that persist across the declared compatible histories.
  Include uncertainty in important losing as well as winning episodes.
- A list of economically important unresolved cases and the missing evidence
  that would narrow them. High coverage of small harmless fills must not mask
  uncertainty around the events carrying most losses or gains.

"Most" is a research objective, not a result to print in advance. Report
the actual recovered fractions and which behavioral conclusions survive the
assumptions. No coverage claim may include orders that never appear in the
public records as though their histories were observed. Use the full supported
market interval where available; first-minute evidence must remain labelled.

Keep broad facts useful even when precise actions remain unknown. For example,
several compatible traces might agree that repeated same-price participation
earns most turnover while disagreeing about exact submission time. Conversely,
if exhaustion versus cancellation changes the proposed pull rule, that rule
is not identified even when the execution ledger matches perfectly.

## Delivery and stopping rule

The final output is a representative set of whole-market timelines, aggregate
behavior/coverage tables and a short set of robust economic mechanisms worth
testing in our simulator. It is not just a list of missing data or eight
hand-selected examples. A subsequent imitation must earn its own priority and
fills and be evaluated under our constraints.

The earlier "stop" instruction applies to manufacturing exact action labels
or training a policy from unsupported labels. It does not prohibit estimated
traces, assumption sensitivity or economically useful partial conclusions.
Stop only an inference whose alternatives cannot be narrowed meaningfully,
record it as unresolved, and continue the supported parts of the reconstruction.

No new run or resource extension is claimed here. Development should freeze a
bounded population, assumptions, controls and resource projection for this
follow-up; the existing discrimination allocation is not silently repurposed.
