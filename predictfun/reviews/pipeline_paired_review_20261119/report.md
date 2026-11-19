# The four-part review of the paired run (R0): validity, prediction, controller, risk — 2026-11-19

**USER:** "After the run, Launch a loop: diagnosis first, if there are issues, clear the issues. After the run, we review: Validity: Are labels,
timing, populations and accounting correct? Prediction: Do predicted action advantages match realized outcomes? Controller: Does acting on those
predictions improve value and useful quoting? Risk: Where do severe losses and inventory costs arise?"

**Run reviewed.** R0 ([receipt](../pipeline_paired_v2_20261119/report.md), commit dbe77f9c): the paired v2 rerun from v1's models, and
evaluation v2.

**Days.** Oct 26 and Oct 27, both consumed.

**Module.** [`pf_pipeline_paired_review.py`](../../research/pf_pipeline_paired_review.py) was declared and committed before its data run
([DECLARATION.md](DECLARATION.md), commit 6775eabd, sha256 beada765...). Both days' outputs record that hash.

**Scope.**
- The restricted experiment: POST versus NO_ENTRY to END.
- Values in cents at END marks, not P&L.
- Single-order episodes.

**Tables.** [tables.md](tables.md), generated from the outputs by [tables.py](tables.py). Every claim below is one of its computed statements,
or a number in its tables.

## Verdicts

| Part | Verdict | Computed evidence (both days unless named) |
|---|---|---|
| **Validity** | **PASS, no open defect** | V1 labels match a per-row loop over the engine fills (0 of 51,642 / 56,366 sampled rows differ). V2 value identity holds on all 1,299,570 / 1,433,878 label rows. V3 timing: 0 of 1,300,498 / 1,434,692 export rows off; pi0 steps and kept fills in order. V4: two independent sources agree on the original (30,242 / 29,820) and with-path (29,832 / 29,338) counts; every row has a status. V5: the ledger equals the evaluation at B to 1e-6 c (0 of 53 checks differ), including the matched null's mean, the first-review shares and the evaluation's input hashes |
| **Prediction** | **FAIL**: A's predicted advantages do not match outcomes | A's old gate fails 23 of 24 groups (it passes only Oct 27 ask PULL on reached rows). POST rows predict +1.69 to +2.91 c per row and realize at most +0.02 c. KEEP on A's own reached rows predicts +0.40 to +1.37 c and realizes −0.24 to +0.17 c |
| **Controller** | **FAIL**: acting on A lowers value | A is below N0 (0), PI0, NEU, E1 and A-PA at B. It is above 0 in 4 and 3 of the 19 scenarios. Matched control with its spread: **PASS on Oct 26, FAIL on Oct 27**. Quoting: A rests longer and is pulled at the first review less often than PI0, NEU and E1, but cancels more per filled share |
| **Risk** | **Located** | A's severe-episode loss is largest after the earliest effect (the continuation window). Loss per episode is highest in the market window's last minute: 2.32 / 2.67 c, against 0.65 / 0.90 c in the other minutes. 88.1% / 86.2% of A's loss arises at a spread of at most 1 c. The spread component is positive and the movement component negative (+5,679 / −7,712 c; +5,817 / −17,682 c). A's time-integrated inventory (45,835 / 44,933 share-s) is above PI0's (20,243 / 16,447), NEU's and E1's. Volatility is mixed: the tercile with the largest loss is high on Oct 26 and mid on Oct 27 |

### FINDINGS checklist (C1 / C3 / C12, both sides)

- **C1, fill age.** Fills on orders under 100 ms old carry 91.0% / 55.5% of A's net value (−1,850 of −2,032 c; −6,586 of −11,865 c).
  - B1a found such replay fills only 8–12% real, so the replay overstates the level of A's loss.
  - A without its young fills is still below 0 (−183 / −5,279 c).
  - Under each B1a age weighting, A stays below PI0, NEU and E1.
  - Policies whose B1a sign differs: PI0 and E1 on Oct 27. Their losses are young fills.
- **C3, cancel-pending.**
  - A: 135 / 121 fills worth +1,480 / −2,307 c, 27.6% / 24.5% of its gross.
  - A-PA: about 54% of its gross is cancel-pending, from the first review's landing window.
- **C12, trade source.**
  - Oct 27: A's E2-timed trades (−8,305 c) lose more than its WS-timed (−4,485 c).
  - Oct 26: the WS-timed trades lose (−1,591 c) and the E2-timed gain (+651 c).

## Corrections in-band (this receipt supersedes these statements; the old receipts stay as provenance)

1. **"A's admissions beat a random selection of the same size on both days."**
   - Source: the paired v1 report; the R0 receipt carried the evaluation's MATCHED CONTROL: PASS.
   - That reading compares A only with the matched null's mean.
   - With the null's exact spread and 1,000 seeded draws (both sides):
     - **Oct 26: z +3.34, draws p 0.001, PASS.**
     - **Oct 27: z +0.51, p 0.30, FAIL.**
   - Per side, only Oct 26 ask passes (p 0.001); Oct 26 bid has p 0.057.
   - The evaluation's reading stands as what it states: above the null's mean.
2. **The practical predicates against A-PA are not evidence.**
   - They are at-best time on both sides, first-review reversals, and cancellations per filled share: True on both days.
   - A and A-PA share admissions, and A's pulls never land before A-PA's, so A keeps a superset of A-PA's fills and live time on every admitted
     order. The predicates hold by construction.
   - The nesting predicate holds on both days: IMPLIED-BY-NESTING.
   - The practical gate's verdict, FAIL, stands through its economic part.
3. **"pi0 still pulls almost every order at its first review"** (paired v1 report). Computed: pi0 pulls at the first review on
   - 91.4% / 91.7% of every eligible opportunity;
   - 74.3% / 72.9% of its own admissions.

## The independent review (a read-only agent, USER: "launch a separate agent to check if there are remaining issues"), each item at the code

| Item | Holds? | Handled |
|---|---|---|
| P2: the practical gate against the same-admission PULL-ALL is nested by construction | Holds | Labelled IMPLIED-BY-NESTING when the computed nesting predicate holds, plus a quoting table against PI0 / NEU / E1, which is not nested. Plan v2's ENTRY-PA quoting comparator has the same property: a decision below |
| P2: the matched control is compared with the null's mean only | Holds | Exact finite-population variance plus 1,000 seeded draws; correction 1 above. The plan's gate definition is mean-only: a decision below |
| P2: the paired evaluation omits C1 / C3 / C12 | Holds | Per-policy tables (above; [tables.md](tables.md)) |
| P3: K7 is a tautology and lets NaN through (`> 1e-6`) | Holds as code | No NaN exists on these days: the v2 identity's numpy maximum over every row is finite (NaN would propagate), and this review's V2 (NaN fails) finds 0 rows. The committed pipeline is unchanged; for its next revision |
| P3: the review's V1 / V2 let NaN pass | Holds | Fixed before the run: NaN fails; falsifiers and mutants |
| P3: K6 cannot fail in reuse mode | Holds | K8 covers the reused models (44 byte-identical files) |
| P3: V4 held a tautology and counted no fallback decisions | Holds | Fixed: two independent sources. Steps without a forecast: 518 / 332, of which 27 / 0 are on A's reached path and 0 at its first review |
| P3: the review never checked where the evaluation JSON came from | Holds | Fixed: V5 checks the evaluation's input hashes and source |
| P3: the evaluation's `loss_p95/p99` are quantiles of −V | Holds, cosmetic | This review's tails use max(−V, 0) |
| P3: compare_v1_v2.py enforces less than its docstring | Holds | The R0 receipt's row was corrected before its commit |
| P3: at-best time on the 50 ms grid; lifetimes include END survivors uncensored; `bins_of` puts NaN in the top bin; relative 1e-6 in R0b / R0c; base LightGBM parameters recorded; the "one refused POST" falsifier tests a non-admitted order | Hold as limits | None moves a number here: there are 0 nonfinite predictions on both days |

## Limits and watch-outs

- **Peak RSS 1,787 / 1,898 MiB against the 2,048 MiB guard.** The buffer is thin; chunk the export read before any larger run.
- **Two metrics are uninformative here by construction.**
  - The worst-1% share of loss is 100%, because fewer than 1% of A's episodes lose.
  - The median lifetime is 79 ms for every policy, because each pulls more than half of its live orders at the first review. The p90 lifetimes
    are in the quoting table (A: 679 / 779 ms).
- **Fill-notice timing is not a computed predicate here.** The independent reviewer verified the convention at the code (a + 100 <= x).
- **The ledger is at B only.** The 19 scenarios are the evaluation's; its R0b–R0f controls passed in every scenario.
- **Single-order episodes:** no joint bid / ask inventory and no re-post.
- **Consumed days; G = 1 per day:** no interval.

## Decisions for the USER

1. **R1 (plan v2):** build the genuine 50 ms WAIT / re-entry branch engine, or not. Nothing past R0 is built without the go.
2. **The gate design for any next candidate:**
   - a matched control with its spread (draws, a declared level);
   - a quoting comparator that is not nested, such as PI0, NEU or never posting, instead of ENTRY-PA / the same-admission PULL-ALL.
3. **A new candidate declaration (plan v2 R3), and untouched days** for its validation (at least 5 complete UTC days).
4. **Deleting the void / pre-fix outputs** stays the USER's call.

## Files

- **Module:** [`pf_pipeline_paired_review.py`](../../research/pf_pipeline_paired_review.py) and its
  [mutation harness](../../research/pf_pipeline_paired_review_mutants.py), selftest plus 38 / 38.
- **Output copies:**
  - review_<day>.json / .txt;
  - CSVs: review_<day>.csv (the tails), review_null_draws_<day>.csv, review_findings_<day>.csv.
- **Logs:** [logs/](logs/).
- **Hashes:** [sources.sha256](sources.sha256).
- **Data:** `data/predict_fun/research/pipeline_paired_review_20261119`, with the per-opportunity ledgers.
