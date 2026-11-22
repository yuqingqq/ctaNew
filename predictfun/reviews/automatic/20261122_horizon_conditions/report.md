# Longer-horizon and condition diagnostic — 2026-11-22 UTC

USER asked to inspect conditions more carefully and whether one second is too
short. Source receipt target: **2063d31a91ab7e6124d7b8b7b7ec5df01ea66711**.
Repository state before this task: **26d50e3ee2451cef7944d8f48b9c64c80176b539**.
Protocol and analysis frozen before outputs in
**aad19914b7ad969e73010e592ffbb098f147fea7**. This is a saved-evidence diagnostic,
not a replay, candidate fit, validation or review of the stopped positive-pressure
pilot. No development source was changed.

## Result

**One second alone does not describe the longer economic exposure.** The saved
data already contains 50/100 ms, 1/5/30 s and END marks; the recent residual
74.2% partition used the 1 s endpoint. On exactly the same FULL-policy fills,
losses are larger at 5 and 30 s in aggregate. Some individual fills recover, and
the day and condition ordering changes with horizon. Extending the measurement
does not establish a longer mandatory resting time or an effective new guard.

FULL means the accepted q5 Binance-pressure policy, with both cancellation and
withholding. Same common cohort at every horizon: **6,336 of 7,395 fills**,
**69,035.026771 shares**, 19 complete markets on consumed Oct 24–27. Other fills
are excluded from this common comparison when any required horizon is unavailable;
the native per-horizon populations are also retained in `horizons.csv`.

| Time after fill | Net marked value, cents | Cents per filled share |
|---|---:|---:|
| Fill-time mid | +40,947.44 | +0.5931 |
| 50 ms | −23,056.42 | −0.3340 |
| 100 ms | −25,786.76 | −0.3735 |
| 1 s | −76,398.88 | −1.1067 |
| 5 s | −83,398.44 | −1.2081 |
| 30 s | −101,413.25 | −1.4690 |
| Episode END, variable horizon | −74,490.87 | −1.0790 |

Each day is negative at 1/5/30 s. Oct 26 improves after 1 s; the other days do
not support a uniform recovery story. END is the supported pre-expiry mid mark,
not settlement, a 30 s endpoint, or realized liquidation. These are signed
execution-to-mid markouts including execution edge; calling them pure price
movement would omit that distinction. Long-horizon losses also include exposure
to later market moves and do not prove that every move was predictable at entry.

Of the 3,843 common-cohort fills negative at 1 s, 1,068 (27.8%) become
nonnegative at 5 s and 1,616 (42.1%) at 30 s. Their recovered shares are 28.4%
and 42.5% of the initially negative shares, respectively. The original 1 s
loss carried by those recovered subsets is only 19.0% and 37.9% of the initial
gross loss. These are retrospective recovery descriptions, not actionable masks.
The entire common cohort still loses more at 30 s, including fills that were
not initially negative. See `transitions.csv` and `readings.json`.

## Conditions actually checked

The separately labelled condition population is **neutral BASELINE live
snapshots**, not FULL's reached states: 4,840 original rows, 4,591 complete
1/5/30 s windows, 249 censored, and 3,138 complete rows with no future fill
retained as zero. Each row is one snapshot of a generation. Its outcome contains
that generation's actual fills at/after a cancellation could take effect,
through its remaining lifetime under the baseline continuation. All features
are from the decision snapshot. Five views were declared before output:

1. Binance adverse-signed 250 ms movement × Predict adverse-signed 250 ms flow.
2. Own displayed-depth depletion.
3. Order distance from best.
4. Generation age.
5. Inventory direction relative to the order.

`conditions.csv` contains every view, day and side, with opportunity/no-fill/
censor counts, shares and positive/negative/net values. Selected illustrations
below explain interpretation; they are not a ranked feature search or promotion.

| Snapshot condition | Common opportunities | 1 s c/share | 5 s c/share | 30 s c/share |
|---|---:|---:|---:|---:|
| Binance non-adverse; Predict adverse flow | 2,084 | −2.130 | −2.507 | −2.931 |
| Binance non-adverse; Predict non-adverse flow | 2,310 | −2.324 | −2.223 | −2.404 |
| Own depth depleting | 2,023 | −1.980 | −2.326 | −3.387 |
| Own depth not depleting | 2,550 | −2.397 | −2.491 | −2.528 |
| Order adds to known inventory | 1,491 | −2.564 | −3.124 | −4.177 |
| Order reduces known inventory | 3,046 | −2.068 | −2.158 | −2.423 |

The flow and depletion ordering reverses between 1 and 30 s. Per-opportunity
and per-filled-share comparisons can also differ because quantity/fill
probability differs. The inventory rows omit the portfolio benefit of reducing
an existing position, so they must not become an instruction to withhold risk
reduction. Marginal comparisons mix market, side, age and remaining exposure;
none demonstrates a conditional forecast or incremental policy gain.

Only 95 snapshots have positive Binance pressure below the old q5 threshold;
91 have common windows and 33 of those have future fills. The positive-pressure
× non-adverse-Predict-flow subgroup has only three filled windows. Its extreme
30 s value is too thin to justify a rule. Most neutral snapshot states have
Binance pressure <=0. That means this particular signal is non-adverse at that
instant; it does not mean no market event occurred, or that no later signal
arrived. This table does not replace the FULL residual fill-time classification.

## Consolidated interpretation limits and next priority

1. **Material interpretation limit: two time axes are currently mixed.** A
   250 ms trailing signal is evaluated against all later fills of a generation,
   some much later than its decision. The fill-arrival interval and the markout
   horizon after a fill are different. Code evidence: `pf_ladder_rest.py:90`
   defines saved markout horizons; `:368–380` implements venue-mid lookup;
   `pf_ladder_audit.py:16–18` and `:86` define the remainder-of-generation window
   and cancel-effect clock. `analyze.py` reproduces that accepted window. A
   weak relationship here can reflect loss of signal relevance before the fill,
   as well as an uninformative signal. This is not evidence that all raw market
   information is useless.
2. **Material scope limit: descriptive, selected coverage.** Four consumed
   days, 19 of the original 96 markets complete; horizon-complete subsets
   further omit late/unmarked exposure. No intervals or untouched-day skill
   claim. The neutral snapshots are not FULL's policy-specific state distribution.
   The replay's unchanged external flow assumption is inherited, not tested here.
3. **No new blocking implementation finding.** Saved identities, accepted 1 s
   values and END accounting reconcile. Existing residual placeholder counters
   and post-fill T1 limitations were not reused as decision inputs. The prior
   deferred checker limitations stay deferred. No repair-only cycle is required.

**ONE next priority for the deeper diagnosis:** a single declared, saved-data
**decision-to-fill × post-fill horizon decomposition**, bundled into the next
development receipt. Keep every neutral decision opportunity and its no-fill
status. Separately measure fills arriving in [effect,effect+250 ms),
[effect+250 ms,effect+1 s), [effect+1 s,effect+5 s), and later while the generation
still exists, with explicit follow-up/cancellation/end censoring. Within each
arrival interval show the already saved 50/100 ms and 1/5/30 s markouts, fixed
common cohorts for horizon contrasts, counts, quantities, fill probability,
value per opportunity and value per filled share. Reconcile the interval union
to the current windows. Reuse the five declared condition views, separately by
side/day, without selecting new thresholds or dropping no-fill opportunities.
Label baseline states and FULL states separately if FULL feature coverage is
added; do not silently use one as the other. Controls must include exact
effect-boundary placement, missing marks and a deliberately changed fill.

This is a test of whether the condition matters **soon enough** and whether
harm **persists**, before another predictor/guard is selected. It does not
authorize relabelling a frozen candidate's primary endpoint after seeing its
outcomes. Reviews can use 1 s as the existing short-term endpoint and longer
horizons as declared secondary diagnostics, with END/inventory accounting apart.
Fast reassessment remains compatible with a longer-value target; a 30 s markout
does not require a fixed 30 s hold.

No new USER decision is needed for a bounded saved-data follow-up within the
existing resource authorization. The development session's
POSITIVE-PRESSURE-GUARD-v1 remains **stopped** at its write-once projection:
0.153 h exceeds its 0.14 h round cap. This task neither resumes it nor changes
that cap; that specific resource amendment still needs the USER. Do not combine
the present diagnostic with the unfinished experiment as a passed replay.

## Reproduction, controls and cost

Run with PF_SLOT=codex through the absolute `predictfun/research/run_light.sh`
wrapper, using the repository's Python environment and absolute script paths.
`analyze.py` binds part records to committed receipt bytes, verifies compressed
ledger hashes, loads the committed neutral states and reconciles 19-market
membership, unique keys, fill/mark counts, accepted +1 s values and END sums.
Hand bid/ask marks, missing-value propagation and an incorrect-hash refusal
run before the analysis. No tapes, fits or new labels were built.

`table_inspection.py` checks the q5 float32/scalar boundary against the current
rows: zero disagreements. This bounds the present table, not a general
authorization to reuse comparison code in a new guard. Mid ages are retained:
FULL common 1/5/30 s marks have median ages 14/17/17 ms and maxima 295/291/249 ms.
No post-result age filter was selected.

Analysis log: 2.36 CPU s, peak 223 MiB. Table inspection: 0.52 CPU s. The first
summary helper failed because the inspection helper was named `inspect.py`,
shadowing Python's standard library; it was renamed `table_inspection.py` without
changing the analysis. A retry omitted the Python executable and refused to
execute; the final invocation succeeded. Both failed logs are retained. No
scientific output or frozen script was changed. The final summary added 0.26
CPU s. Recorded use as of that job was about **1.86535 cumulative core-hours**;
this diagnostic's numerical checks used about **0.00090 core-hours**, below
0.01 and the existing 2.0 cumulative cap. Staging/commit verification is separate
small administrative work. `SHA256SUMS` binds all report artifacts.

Largest evidenced limitation: **the available condition labels combine when a
fill arrives with what happens after it, on a small consumed neutral-path
population**. The longer marks confirm economically persistent harm in aggregate;
they do not yet identify a reliable decision-time discriminator.
