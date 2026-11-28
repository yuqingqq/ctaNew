# One-day test of the whole pipeline (prediction model + value mapping + keep / pull rule), Oct 27 — 2026-11-19

**USER:** "We run the whole pipeline for one day test first". The pipeline:
1. [the export](../pipeline_export_20261119/report.md): causal inputs and labels on the simulator's JOIN orders;
2. `pf_pipeline_bench.py`: the prediction model and the contract's value mapping;
3. `pf_pipeline_policy.py`: the declared keep / pull rule through the simulator's 19-scenario bank.

**Scope.**
- Fitted on Oct 23-25 and tested on Oct 27, one consumed day; Oct 26 was not read.
- Readings, the gate and the rule were declared in the modules before any label or prediction was read.
- Values are local marks at END (+5 s), in cents, not P&L.

## Gates

| Gate | Result |
|---|---|
| Benchmark: selftest; mutation suite | pass; **17 / 17** |
| Benchmark controls B1-B5 (chronology and export hashes, no label among the inputs, the mapping re-derived from the raw model heads, saved models reload, bounds) | pass |
| Scenario replay v6.2 (`outcome()` takes an external rule's pull steps): selftest; mutation suite | pass; **49 / 49** |
| Trading test: selftest; mutation suite | pass; **11 / 11** |
| Trading test controls P0a-P0f (A+0 = the committed report and keep / pull run; the learned rule at B = the clock audit; policy-free values agree; predictions = the export's eligible rows; the rule's actions = the contract's `decide_one` on 1,998 sampled rows, 2 atoms refused) | pass |
| Memory | benchmark peak 1.6 GiB, trading test 1.5 GiB |

**Four first-run failures**, each kept in [logs/](logs/):
1. The benchmark refused a changed export part but left its partial folder. Exports are now verified before anything is written.
2. The benchmark's first real run stopped on its own 2 GiB guard at 3.4 GiB, with no output: it loaded every column of every row. It now reads only the columns used, filters as it reads, and works one side at a time.
3. One scenario mutant (malformed triggers) crashed instead of being caught. The selftest now records a malformed trigger that reaches the valuation.
4. One trading-test mutant (P0d disabled) survived: the KEEP-drift falsifier trips both parts of P0d and the selftest required only the label.
   It now requires both specific messages.

## The prediction model ([bench.txt](bench.txt))

**Candidates**, per kind (POST, RETAIN) and side, with the pipeline's pinned LightGBM settings:
- **J** (primary): a joint two-part model of P(fill) × E[q | fill] and E[q · move | fill], mapped by the contract's formula (the E[q · move] term is never replaced by E[q] · E[move]);
- **D**: direct value;
- **M**: mechanical inputs only;
- **F**: the fair edge only (A+0's information).

**Mapping gate (J calibrated in amount and ranking beyond the fair edge, both sides): PASS for POST and RETAIN.**

| RETAIN, Oct 27 (per decision; rows of one order overlap) | Bid | Ask |
|---|---:|---:|
| Decisions | 703,520 | 701,020 |
| Predicted / realized quantity | 0.953 | 0.986 |
| Predicted / realized E[q · move] | 0.890 | 1.011 |
| Fill AUC | 0.776 | 0.780 |
| Value rank correlation: J / D / M / F | +0.120 / +0.113 / +0.107 / +0.047 | +0.120 / +0.110 / +0.103 / +0.054 |
| Lowest predicted decile: predicted / realized mean | -26.6 / -26.7 c | -30.1 / -24.9 c |
| Highest predicted decile: predicted / realized mean | +1.15 / -0.08 c | +0.97 / -0.71 c |

Most of the ranking comes from the order and queue inputs (M). The market features add about +0.015.

**POST:**
- J calibrated (0.970 / 0.974); value rank +0.040 / +0.053 against the fair edge's +0.023 / -0.015.
- Every predicted decile's realized mean is negative; the best is -5.5 c (bid) and -2.6 c (ask).

## The keep / pull rule through the scenario bank ([policy_20261027.txt](policy_20261027.txt))

**Rule (declared): pull at the first live step whose predicted RETAIN value is below 0.**

Counts computed by [counts.py](counts.py) ([counts.txt](counts.txt)):
- The model predicts a negative keep value for 93.2% of the 1,404,540 decisions.
- It pulls 27,630 of 27,881 orders (99.1%), 97.1% of them at the first live step (+100 ms).

| Value at B (c) | |
|---|---:|
| KEEP (never pull) | -443,038 |
| A+0 (the fair-edge rule; 13,861 pulls) | -181,823 |
| MODEL-J (the learned rule; 27,630 pulls) | **-23,790** |
| PULL-ALL (every order at its first live decision) | -23,346 |

**Across the 19 scenarios, MODEL-J's value is:**
- above A+0 in 17 / 19 (not at S2hi or EVhi, the two scenarios where KEEP is positive);
- above PULL-ALL in only 3 / 19 (S2hi, EVhi, L100);
- above KEEP in 17 / 19.

Its E (improvement over its own matched null) is positive in 18 / 19.

**Readings (declared):**
- P1 / P3 / P1-NY / P3-NY, P4 (V - KEEP), P5 (V - PULL-ALL), D1 / D2 / D1-NY (against A+0): all **NOT-STABLE-IN-TESTED-SCENARIOS**.
- C2 at M250: HOLDS. C1 B1a: HOLDS-SIGN. The benchmark's RETAIN gate: PASS.

**POST, descriptive at B** (the review's never-post control; not run through the bank): 1,892 orders (6.4%) have a positive predicted POST value. Their realized mean is -5.97 c, against -15.10 c for all orders. Not posting (0) beats posting even the best-predicted orders on this day.

## What this supports (protocol section 4)

- **The pipeline runs end to end with every control and suite passing.** Its prediction model is calibrated in amount and ranks the value of
  keeping a resting order better than the fair edge, on a day after its training days.
- **Its keep / pull rule rediscovers "cancel almost at once".** For these 30-sh JOIN orders at the best, the predicted value of keeping is
  negative almost everywhere. The rule beats A+0, which keeps many losing orders, but not PULL-ALL: the few orders it keeps cost 444 c more
  than pulling them at B.
- **The loss is in entering, not in when to cancel.** Every POST decile loses on realized value. The order design (a 30-sh JOIN at the best)
  is what loses in this simulator, under most timing scenarios.

None of this is P&L, validated on live orders (C10), robust to the queue rule (C5), or more than one consumed day (G = 1). The model is one of
the benchmark's 8 candidates.

**Limit.** The model's inputs that depend on the order's own fills (known remaining) use the baseline timing's fills in every scenario.

**Provenance.** The two selftest-only fixes (3 and 4) came after the first trading-test run, so the test was rerun on the final sources. The
rerun reproduces run 1 exactly in every value, reading and difference ([compare_run1.py](compare_run1.py) → [compare_run1.txt](compare_run1.txt)).
Run 1 is kept as `pipeline_policy_20261119_run1`. Every recorded source and output hash matches the committed files
([sources.sha256](sources.sha256)); the 44 model files are hashed in [bench.json](bench.json).
