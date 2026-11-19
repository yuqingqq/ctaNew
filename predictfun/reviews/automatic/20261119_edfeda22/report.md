# Latest paired-model results reviewed — 2026-11-19

Target commit: **edfeda22dd72aacaee0c758e96e6e29fe2667422**.
Reviewed from detached checkout `/tmp/pf-review-edfeda22`; development changes
were preserved. This is a manual review; the scheduler remains disabled.

**The headline losses reproduce. The previous coherence defect is fixed.
Two diagnostic defects remain, and the planned adaptive POST-versus-WAIT
experiment is still incomplete.** The current experiment is a useful, narrower
entry-versus-no-entry test with learned cancellation. Its failure does not
establish that the full planned timing model fails.

Evidence: [independent probe](verify.py), [receipt](receipt.json),
[synthetic counterexamples](synthetic.json), [prior-fix check](prior_fix.py),
[prior-fix receipt](prior_fix.json). All checks used `PF_SLOT=codex` through
`run_light.sh`, with absolute script paths and pinned source. No fitting,
raw-tape experiment or untouched validation day was used.

## Findings, ranked by impact on the next decision

### 1. P2 — “own-policy reached” diagnostics include orders the policy never posted

`pf_pipeline_paired.py:192` stops at each order's first negative RETAIN score,
but `diag()` at line 202 does not intersect that mask with the same head's POST
admissions. `pf_pipeline_paired_eval.py:261` repeats this omission for the
reached-state shift. It affects A, AF and AM's KEEP/PULL prediction diagnostics;
the production value aggregation does apply admissions.

| A's RETAIN population | Oct26 | Oct27 |
|---|---:|---:|
| Reported “reached” rows | 126,063 | 144,060 |
| Reached after requiring actual POST admission | 67,776 | 76,731 |
| Rows belonging to never-posted orders | **58,287** | **67,329** |
| Corrected mean prediction / paired label, cents | +0.499 / −0.070 | +0.388 / −0.153 |

These are overlapping decision diagnostics, not independent orders or returns.
Candidate admissions are 14,126 / 13,319 orders. About 46–47% of the published
reached population is not on the candidate's executed path.

The selection error materially changes attribution. Corrected A KEEP-selected
groups are:

| Day/side | Reached KEEP rows | Predicted / realized paired value, cents |
|---|---:|---:|
| Oct26 bid | 20,097 | +27,521 / −4,830 |
| Oct26 ask | 34,769 | +19,843 / +5,942 |
| Oct27 bid | 20,815 | +23,061 / −46 |
| Oct27 ask | 43,752 | +17,550 / −4,713 |

Combined Oct26 KEEP label value changes from approximately **−7,374 to +1,112**;
Oct27 changes from **+5,291 to −4,759**. This changes which states appear harmful.
All four corrected KEEP calibration groups still FAIL; POST diagnostics are
unchanged, so A's overall prediction gate still FAILS. Oct27 ask PULL's corrected
group passes its individual calibration gate. Do not preserve the old subgroup
interpretations merely because the overall verdict remains FAIL.

Correction: intersect each head's reached mask with its own usable positive
POST admissions, joining day and order identity. Keep the all-posted
counterfactual population separately named. Add a two-order falsifier where one
POST is rejected; our counterexample returns two reached rows instead of one,
while the all-admitted positive control works.

### 2. P2 — `pair_qm` subtracts movement terms referenced to different mids

`pf_pipeline_paired.py:103` and line 115 subtract each of `value_m50`, `q_m50`
and `qm_m50` at the later pi0 pull. However, `pf_pipeline_export.py:203` defines
`qm_m50 = value_m50 - side_sign * (current_mid - price) * q_m50` separately at
each decision. The two movement terms therefore use different origins.
Their raw difference is not the movement component of the paired value at the
current decision and fails the existing quantity-plus-movement mapping.

For this fixed-price, single-generation experiment the aligned component is:

```
pair_qm_at_current_mid = pair_v - side_sign * (current_mid - price) * pair_q
```

Our physical synthetic example uses bid price 50, END mark 51, current mid 50,
later mid 60, and removable quantities 10 then 5. Paired value and quantity are
both 5, but the saved arithmetic returns movement **55 instead of 5** for both
POST and RETAIN. The identical-mid positive control returns 5 correctly.

Saved artifacts also fail the mapping, using absolute tolerance 1e-6 cents:

| Day | POST mismatches / rows | RETAIN mismatches / rows |
|---|---:|---:|
| Oct24 | 1,338 / 30,358 | 22,626 / 1,439,399 |
| Oct25 | 957 / 29,558 | 16,793 / 1,491,072 |
| Oct26 | 942 / 29,832 | 13,808 / 1,269,738 |
| Oct27 | 747 / 29,338 | 10,568 / 1,404,540 |

Most have zero paired quantity yet nonzero reported movement. Example: Oct24
order 27561 POST has `pair_q=0`, `pair_v=0`, `pair_qm=-165` cents. These are
reference-price artifacts, not evidence of severe adverse selection on the
incremental fills.

**This does not cause A's measured loss.** The direct heads train on `pair_v`
only (`pf_pipeline_paired.py:297`), and independent fill-window accounting
reproduces both `pair_v` and `pair_q` on all four days. K4 at line 264 verifies
value, not movement, so its passing result does not validate all three fields.
Rebase before using this column for economic decomposition or a movement head;
alternatively rename it explicitly as a difference of differently referenced
terms and do not treat it as paired movement. Require the full value identity
on every saved row. A genuine multi-generation branch target will need its own
cash/inventory decomposition, rather than assuming this single-price formula.

## Open target gate: WAIT still cannot lead to later entry

`pf_pipeline_paired.py:20` explicitly sets the WAIT branch to zero because pi0
only enters on the original five-second grid. Its `label_post()` at line 115
therefore implements POST-with-pi0-cancellation versus **NO_ENTRY to END**.
This is internally consistent for that restricted policy and clearly disclosed
in the development report. It does not satisfy the approved plan's G2 requirement
that WAIT one 50 ms review can later POST from its own state
(`handoffs/20261119_entry_retention_priority.md:129`). PULL likewise has no
re-entry continuation. G1 was explicitly skipped by the user; that is not a
finding or a reason to require the skipped baseline now.

Keep this run named as the restricted experiment. Full adaptive timing still
needs branch-local delayed entry/re-entry, queue and pending states, common
absolute END and causal forecasts on changed generations. The absence of this
engine does not invalidate these saved single-order values, but it prevents
claiming that the planned POST/WAIT pipeline has been tested. The one-order
common-END convention also makes nonzero paired and neutral labels share a
sign; the synthetic opposite-sign arithmetic case is not a production-engine
demonstration of the planned continuation behavior.

## What is verified, and what the model actually achieved

Baseline: 30 shares, JOIN, M50, activation 58 ms, cancellation 37 ms, notice
100 ms, common END at +5 seconds. Values are **cents marked at END**, not
settlement P&L or realized spread capture. Per-fill accounting independently
reproduces all six candidate/control policies and their same-admission
PULL-ALL counterparts, both sides, plus all-entry PULL-ALL at B on both days.

| Policy / metric | Oct26 | Oct27 |
|---|---:|---:|
| A, new paired-value candidate | **−2,032.12** | **−11,865.45** |
| A-PA, identical admissions, pull at first review | −1,053.02 | −6,943.81 |
| A's extra value from its cancellation timing | **−979.11** | **−4,921.64** |
| PI0, warm-up reference | +478.19 | −2,607.70 |
| NEU, neutral-target rule on the same training days | +3,540.18 | −2,826.52 |
| E1, prior rule of record | +276.93 | −823.53 |
| No entry | 0 | 0 |
| A first-review pulls, fraction of admitted orders live there | 57.05% | 57.36% |
| A median live lifetime | 79 ms | 79 ms |

First-review cancellation falls compared with the earlier all-entry rule, but
the new candidate still has negligible useful two-sided quoting and fails its
economic/practical gates. Different admission populations prevent interpreting
the reduction in absolute loss versus all-entry as a pure prediction improvement.
The same-admission comparison establishes that A's extra retention loses value
on both days. Failing calibration remains a real issue after correcting the
diagnostic population; these results do not establish that market information
has no predictive value or that a more complex model would solve the problem.

## Prior findings and verification limits

Review **1adf6a334306623bbd9d1af2332d21a9e0adcc55**:

- Zero-quantity/nonzero-movement coherence: **resolved** in benchmark v3 and
  policy v3.1. All 12 pinned boundary cases pass, including both signs,
  probability endpoints, nonfinite moments and numerical tolerances. Saved
  flags reject the previous 23 / 15 rows; first triggers change zero times.
- Fixed weights and historical values: 44 v2/v3 model files are byte-identical
  and their actual hashes verified; policy CSVs are unchanged on both days.
  The saved all-row parity receipts agree on all 1,269,738 / 1,404,540 RETAIN
  rows. This review inspected that implementation and receipt, rather than
  rerunning the full Python contract loop.
- Later-pull/never-pull attribution and overbroad entry conclusion: **resolved**
  by the v3 superseding report; E1's previously reviewed conditional-entry
  values reproduce here.

Chronology: pi0 Oct23; candidate training Oct24–25; evaluation Oct26–27.
All are consumed development dates. There are two evaluation days, hence no
confidence interval or untouched-validation claim. Original opportunities are
30,242 / 29,820, of which 29,832 / 29,338 have a path. POST-only refusals remain
in the labelled population with zero value; other statuses remain in the export
receipts. Full eligible/refused label counts and per-day export as-of are in
`receipt.json`. Paired artifacts as-of **2026-11-19 11:54:21 UTC**; evaluations
as-of **12:52:32 / 12:53:15 UTC**; review artifact check **13:26:50 UTC**.

Checks took about 35 seconds total including the provenance follow-up, peak RSS
688 MiB. Synthetic cases are
implementation evidence only. Source, model, prediction, label, receipt and
export-part hashes were checked; independent label accounting covers 5,723,835
rows. Baseline value is independently re-accounted; all 19 alternative scenarios,
model fits and the developer's full mutation suites were not rerun. The matched
null is an exact cell-weighted mean in source, not a sampled extreme or a
statistical significance result. The current replay remains single-order,
without shared inventory/re-entry; timing scenarios use frozen decisions.

## One concrete next priority

**Publish a corrected paired-diagnostic receipt before another candidate fit.**
The development session should rebase or retire the misleading `pair_qm` field,
intersect KEEP/PULL diagnostics with each head's actual admissions, and name the
current alternative NO_ENTRY. Acceptance: the two independent falsifiers and
their positive controls pass; every saved paired component reconciles to value;
reached counts are admission-aware; corrected groups/gates and the unchanged
direct-value baselines are recorded in a superseding receipt. No refit or
threshold change is needed for this deliverable. Then the substantive unmet
experiment is G2's genuine 50 ms WAIT/delayed-entry continuation, before treating
another head or calibration change as a test of the full planned model.

The other session owns implementation. This review changes only review evidence
and review-owned handoff/state entries; it does not modify models or the engine.
