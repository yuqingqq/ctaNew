# R0 (plan v2): the corrected paired-diagnostic receipt — 2026-11-19

**USER:** "Check my commit, fix the issues, do the pipeline run and evaluation" (plan v2 17850030, review 0026ab85).
This receipt supersedes, in-band, three things in [the paired receipt](../pipeline_paired_20261119/report.md):
- its reached-row diagnostics;
- its KEEP / PULL prediction groups;
- its `pair_qm` column.

The models, `pair_v`, the thresholds and every evaluation value stand. The rerun reproduces them (below). No model was refitted (plan v2: "Do
not refit to repair reporting").

**Name.** This is the restricted experiment: **POST versus NO_ENTRY to END**. Under pi0 a WAIT can never post before END, so its zero is not
evidence about an adaptive WAIT or a later entry. That experiment is plan v2's R1.

## Review 0026ab85, each finding at the code

| Finding | Holds? | Fix |
|---|---|---|
| P2: "own-policy reached" rows include orders the policy never posted | Holds: `pf_pipeline_paired.reached()` and `diag()` (and the evaluation's reached-state shift) never intersected the head's own POST admissions | A head's reached rows now require its own usable positive POST admission (same day / order) and stop at its first pull; the all-posted counterfactual is kept under its own name. Falsifier: one refused POST of two gives one reached row; positive control: both admitted give two |
| P2: `pair_qm` subtracts moves referenced to different mids | Holds: `label_retain` / `label_post` subtracted `qm_m50` at the later pi0 pull, referenced to that step's mid | `pair_qm = pair_v - sign * (mid_now - price) * pair_q`. **K7**: the full value identity on every saved row. **K4** now also reconciles `pair_q` with the fills. Falsifier: the review's case (mids 50 then 60, removable 10 then 5 sh) gives 5, not 55; positive control: identical mids |
| Gate: WAIT never posts later; PULL never re-enters | Holds as disclosed | The alternative is named NO_ENTRY to END throughout; the genuine 50 ms WAIT engine is plan v2's R1 |

## Checks (all before the rerun)

| Check | Result |
|---|---|
| `pf_pipeline_paired.py` v2: selftest; mutation suite | pass; **28 / 28**. The first run found 2 selftest gaps: K4's value checks had no value-only falsifier once a quantity check was added. Fixed with value-only falsifiers; log kept |
| `pf_pipeline_paired_eval.py` v2: selftest; mutation suite | pass; **18 / 18** (one new mutant: the reached-state shift ignoring admissions) |
| [G0 boundary cases](../pipeline_one_day_v3_20261119/g0_coherence_cases.py), rerun against regression | **12 / 12** |
| `pf_pipeline_entry.py` (the per-order path) | unchanged since its 32 / 32 suite |

The rerun's summary header prints "CONTROLS: YES (K1-K6)", a stale label. K7 (the value identity) and K8 (the reused models' hashes) also ran:
a rerun cannot finish without them.

## The rerun (no refit) and the evaluation

Declared before v2 ran: [compare_v1_v2.py](compare_v1_v2.py) → [logs/r0_compare.log](logs/r0_compare.log). **Every check is True:**

| Check | Result |
|---|---|
| Model files | all 44 byte-identical to v1 (K8 refused a changed file in the selftest) |
| Label columns | identical on every row except `pair_qm`, Oct 24–27 (5.7 million rows) |
| v2 value identity `pair_v = sign*(mid - p)*pair_q + pair_qm` | holds on every row; max error 1.1e-13 c |
| v1 rows failing that identity (POST / RETAIN) | Oct 24: 1,338 / 22,626; Oct 25: 957 / 16,793; Oct 26: 942 / 13,808; Oct 27: 747 / 10,568. **Exactly the review's counts** |
| Predictions (pi0, NEU, A, AF, AM, `pair_v`, `pair_q`) | identical |
| POST diagnostics | unchanged |
| A's reached rows (both sides) | Oct 26: 126,063 → **67,776**; Oct 27: 144,060 → **76,731**. The review's numbers |
| Evaluation values, matched null, admissions, pulls | identical on both days (enforced) |
| Evaluation readings | none differ (printed, not enforced); the other metrics were not compared; the reached-state shift changes by design |

The memory fix brought the evaluation's peaks to 1,475 / 1,535 MiB (v1: 1,951 / 2,075).

### The corrected KEEP / PULL diagnostics (A; reached = A's own admitted orders up to its first pull)

| A | Oct 26 bid | Oct 26 ask | Oct 27 bid | Oct 27 ask |
|---|---:|---:|---:|---:|
| Admitted orders | 6,319 | 7,807 | 5,866 | 7,453 |
| Reached rows (v1, every order posted) | 25,831 (52,698) | 41,945 (73,365) | 26,138 (55,522) | 50,593 (88,538) |
| KEEP: n / predicted / realized (c) | 20,097 / +27,521 / −4,830 | 34,769 / +19,843 / +5,942 | 20,815 / +23,061 / −46 | 43,752 / +17,550 / −4,713 |
| PULL: n / predicted / realized (c) | 5,734 / −6,727 / −3,801 | 7,176 / −6,808 / −2,053 | 5,323 / −5,582 / −1,132 | 6,841 / −5,237 / −5,874 |
| Gate groups (POST / KEEP / PULL) | FAIL / FAIL / FAIL | FAIL / FAIL / FAIL | FAIL / FAIL / FAIL | FAIL / FAIL / **PASS** |

On the executed path A's reached decisions predict +0.499 c and realize −0.070 c on Oct 26, and predict +0.388 c and realize −0.153 c on Oct 27.
The v1 all-posted figures were +0.348 / −0.228 and +0.308 / −0.135. Combined KEEP label value is +1,112 c on Oct 26 (v1 −7,374) and −4,759 c on
Oct 27 (v1 +5,291).

**The candidate's verdicts stand.** Its prediction gate fails on both days; the corrected population changes which states look harmful,
not the verdict. Its economic and practical gates fail, and its matched control passes.

Unchanged values at B (c), reproduced:

| Policy | Oct 26 | Oct 27 |
|---|---:|---:|
| A | −2,032 | −11,865 |
| A-PA | −1,053 | −6,944 |
| PI0 | +478 | −2,608 |
| NEU | +3,540 | −2,827 |
| E1 | +277 | −824 |

## Files

- **Sources:** `pf_pipeline_paired.py` v2 and `pf_pipeline_paired_eval.py` v2, with their mutation harnesses.
- **Receipt scripts:** [compare_v1_v2.py](compare_v1_v2.py).
- **Output copies:** [paired.json](paired.json), [paired.txt](paired.txt), eval_<day>.json / .txt / .csv.
- **Logs:** [logs/](logs/); hashes: [sources.sha256](sources.sha256).
- **Data:** `data/predict_fun/research/pipeline_paired_v2_20261119` and `pipeline_paired_eval_v2_20261119`; v1 stays as provenance.

## Open items from an independent review (handled in the next step)

A read-only reviewer agent, started on the USER's request, found no defect that changes a published number or verdict. Its P2 items are:
- **The practical gate is nested by construction.** Against its own admissions pulled at the first review, any candidate that keeps some
  orders longer gains at-best time and lowers both first-review reversals and cancellations per filled share.
- **The matched control is compared with the null's mean only**, not its spread. "A beats random selection" therefore means "above its
  expected value".
- **The paired evaluation omits the FINDINGS checklist's age split, cancel-pending and trade-source tables (C1 / C3 / C12).**

Its P3 items include NaN rows passing the four-part review's V1 / V2, and uncomputed sentences in the v1 report, such as "pi0 still pulls
almost every order at its first review". These are verified and cleared in the four-part review's receipt.

**R0 gate (plan v2 section 3):** no unexplained component mismatch, no reached row without admission, corrected subgroups and honest names
in a new receipt, and the direct-value baselines reproduce. All met (above).
