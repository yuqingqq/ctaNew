# Quote absence and closed recovery gates

As of **2026-11-13T16:00:57.453496+00:00**. On the six consumed test dates
(Oct 26/27/31 and Nov 1/2/3), much of the recovery candidate's entry-phase
absence overlaps both recovery gates being closed. This is a reconstruction
of recorded state and actual accepted-order lifetimes. It does not estimate
the profit or extra presence obtainable by reopening earlier.

## Entry-phase result

Every row includes **1,728 windows**, including windows without orders. Entry
is the first **60 seconds** of each window. Percentages in the final three
columns use **no-quote time**, not all entry time, as their denominator.

| Scenario | Policy | Any quote, % of entry | No quote, % of entry | Both gates closed during no-quote time | One closed | Neither closed |
|---|---|---:|---:|---:|---:|---:|
| X | DEFAULT | 89.01% | 10.99% | 0.00% | 0.00% | 100.00% |
| X | DIRECT10 | 12.85% | 87.15% | 79.19% | 10.61% | 10.20% |
| X | COMPONENTS10 | 13.01% | 86.99% | 79.39% | 10.38% | 10.23% |
| X | SELECTIVE10 | 19.57% | 80.43% | 76.66% | 12.25% | 11.09% |
| stress | DEFAULT | 88.60% | 11.40% | 0.00% | 0.00% | 100.00% |
| stress | DIRECT10 | 16.93% | 83.07% | 71.63% | 17.12% | 11.25% |
| stress | COMPONENTS10 | 16.91% | 83.09% | 71.48% | 17.30% | 11.22% |
| stress | SELECTIVE10 | 22.34% | 77.66% | 69.69% | 18.28% | 12.03% |

For SELECTIVE10, no quotes together with both gates closed occupy **61.65% of
all entry time in X** and **54.12% in stress**. These are the same observations
with a different denominator. On a side-time basis, **83.47% / 79.43%** of
absent side-time overlaps that side's closed gate. Side-time and wall-clock
time are distinct and are not added together.

All generation statuses remain in the receipt. For SELECTIVE10, X has
100,474 generations (99,244 with positive accepted lifetime): 90,479 pulled,
5,974 scored-keep, 2,800 accepted-no-cancel and 1,221 post-only refusals.
Stress has 91,920 generations (90,342 positive lifetime): 82,353 pulled,
4,784 scored-keep, 3,216 accepted-no-cancel and 1,567 post-only refusals.
These statuses describe generations, not windows or independent decisions.
Zero-lifetime records contribute zero time and remain in population accounting.

## Interpretation for larger residuals and prediction

Larger residuals can retain an existing order through several partial fills.
They cannot retain its queue position after it has been cancelled and its
replacement is withheld. The two mechanisms therefore need separate evidence:
value in keeping the residual, and value in re-entering after withdrawal.

The recovery gate is a substantial concurrent restriction on entry in these
runs. It opens only after the existing recovery estimate remains clear for
250 ms with no observation gap over 50 ms. The diagnostic does not establish
that the closed periods were unnecessarily cautious: inventory, prices, data
and other constraints can also prevent orders, and entering during those
periods could lose money. The recovery proxy's previously recorded forecast
weakness and the [negative primary/stress portfolio result](20261113_recovery_primary_stress_review.md)
remain relevant; higher presence alone is not the objective.

The queued five/50-share experiment includes notified residual quantity,
partial-fill state and inventory under a shared cap, with native KEEP/RESET/
PULL alternatives and a consistent pressure-clear re-entry rule. Its
historical size/model advantage is still unknown. The separately frozen depth
forecast tests further adverse movement; it does not yet supply a new entry
or cancellation policy. No frozen design was changed using this diagnostic.

## Reconstruction and controls

Source **0d100fef4972d176ec9330fd613637795a8ff801** binds four direct source
identities, the inherited scientific/accounting declarations and **313 named
inputs** before calculation. Twelve completed scenario-days contain four
policies each: **13,824 policy-windows** in total. All accepted intervals are
rebuilt from native activation clocks and lifetimes and reconcile exactly
with saved presence artifacts. Every recovery trace is checked by the native
event verifier; saved cycle/observation/placement counters must match.

Gate closure runs from the recorded pull until the recorded clear
observation, or window end if unresolved. Actual quote intervals continue
through any cancellation delay. Synthetic and native controls cover overlap,
duplicate intervals, phase conservation, invalid clocks/lifetimes, unresolved
gates, zero-generation windows and live orders while cancellation is pending.
Recorded model estimates are used to reconstruct the gate; they are not
independently recomputed by this diagnostic.

The first-60-second and subsequent reduce-only partitions sum to the full
**240.1-second** replay interval. This is not a 300-second live uptime claim.
The receipt carries all three phases, daily totals, generation statuses and
trace counts. No null, interval, model selection or new economic gate is
introduced for this descriptive time decomposition.

Execution completed with exit zero, **335 MiB** peak RSS, **365.40 seconds**
wall and **361.95 seconds** user CPU on the free auxiliary slot. All named
source/input hashes and all three receipt copies matched after execution;
the hash checker also refused a deliberately incorrect digest.

- [Result](../reviews/model_recovery_20261113/recovery_absence_overlap_v1.json),
  SHA `1b8d148a7483f9948510ff5e7535b98b44ea5a803209dadc1cd58b1d3587fdc9`.
- [Declaration](../reviews/model_recovery_20261113/recovery_absence_overlap_v1_declaration.json),
  SHA `57f830ac42b4d2bdc49ac6abb5f6529d23cb4ff96b91da7871a20843990be704`.
- Preserved execution log SHA
  `4ada2cd002a7dc4e362ba3ebde50e06f5d35d3c3a2a7e7f878ecf79145bec6e1`.

Recovery remains 14/18 full runs verified; Oct 31/age500 is active. Full
recovery reporting and verification, followed by the frozen size/depth work,
remain pending. No forward validation, live improvement or goal completion
is established.
