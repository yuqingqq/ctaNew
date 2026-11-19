# R2-A: the shadow-probe / feature / support adapter, manifest `r2_shadow_sampling_v1`, and the Oct 23 census — 2026-11-19

**USER:** "Check my latest commit, next, can you ask codex research session for next priority. continue the implementing and next priority
loop".

**Review 7d59bde7's next priority:** "Deliver and review R2-A: the shadow-probe / feature / support adapter and `r2_shadow_sampling_v1`
manifest, synthetic controls, and the fixed first 128 original Oct23 opportunities per side, with no teacher fit." The contract is c68a072c's,
with efac0452's clarifications.

**Module:** [`pf_r2_shadow.py`](../../research/pf_r2_shadow.py), with its [mutation harness](../../research/pf_r2_shadow_mutants.py).
**No model is fitted or read.**

## What it builds

| Part | Implementation |
|---|---|
| Population (fixed before any outcome) | Oct 23, the first 128 original opportunities per side by oid, of **any status**: 256 selected, of which 240 eligible, 14 `refused_post_only` and 2 `missing_endpoint`. The 2 no-path rows stay as status rows. Selected-ID hash `4538b694c92c5b65...` |
| Manifest | phase = the first 8 bytes of SHA-256(`r2_shadow_sampling_v1\|day\|instrument\|episode\|side\|head`, UTF-8), mod 5 (never `hash()`). Entry offsets: 0, plus 50k ms where k mod 5 = phase(post). The k = 99 offset (4,950 ms) is kept as `near_end`. Retained decisions are sampled at k mod 5 = phase(retain), plus each probe's first live review. Joint absent / live states use the post / retain phase. The census is sorted in the manifest's order (stable original identity, source / head, offset, generation, decision) |
| Sources (labelled separately) | `entry`: one neutral shadow per offset, each its own alternative world. `retain`: those shadows' live decisions. `joint_post_keep` / `joint_chase`: R1's fixed stand-ins on both sides; at a sampled feasible state the engine forks the focal side's neutral hold from that exact state |
| Neutral J targets | The focal generation alone, held under the client structural rule to the **original** END, never replaced. POST: its fills from activation. RETAIN: its fills at or after decision + 37 ms (`pf_pipeline_export.affected`, unchanged). Outputs: q, value, qm = value − s(ref − p)q (ref = pf_mid at the decision), and severe excess. Zero, no-fill and refusal outcomes are kept |
| Inputs | The export's 91, in its order, from the unchanged builders: `real_features`; `known_steps` → `order_inputs` (with the generation's own clock and the original terminal); the `fp_` extras; `fair_edge` and `t_to_end`. They are stored as float32, the benchmark's dtype |
| Context | c68a072c's 14 `ctx_*` fields, taken from the branch engine's **client** state (acknowledged / reserved, never `Engine.rem`) |
| Identity | Every branch carries `ident` (efac0452 amendment 1): day, instrument, episode, side, absolute start / END, mark source. Rows add `decision_abs_ms`, `end_abs_ms`, `end_mark` (label only) and `mark_source` |
| Weights | Equal total weight per original episode × side × head across its sampled ok rows and source families (w = 1 / n). Zero elsewhere |
| Costs | Explicit zeros (fee 0, rebate 0; the gross research convention), recorded before any teacher fit |

## Checks

**Selftest** ([log](logs/pr2_selftest13.log)), synthetic:
- **M1 manifest:** 50 keys against an independent hex-digest computation, plus its range and offsets, and no `hash()`.
- **Later birth:** offset probes join the book known at their own decision.
- **Hidden-notice context:** a fill at 200 ms is not visible at 250 ms but is at 300 ms. The planted engine-truth context is caught.
- **Short horizon and `near_end`.**
- **Census completeness:** the retained rows match an independent count of open ticks.
- **Sampling rule:** includes the first live review; a 1,000 ms review excludes a fill at 1,010 ms, which comes before its 1,037 ms landing.
- **Labels:** by a per-fill loop, including a 60 c loss (severe excess 30 c) and the value identity.
- **Price:** a row's price equals its fills' price.
- **Identity on every branch,** and missing marks as statuses.
- **The synthetic-day census:** inputs, context, weights, provenance and canonical order.
- **The export's causal checker on the features,** with the look-ahead stub caught.
- **Order inputs:** unchanged by a later fill, changed by a notified one.
- **The real-row causal check:** fires on the look-ahead builder.
- **The truncated-tape context:** a known fill changes it; truncation exposes a planted future leak.
- **The census label reconciliation:** fires on a planted label error.

**Mutation suite:** **25 / 25** caught by a check ([log](logs/pr2_mutants8.log)).

**Earlier runs, kept as provenance:**
- **[3 unexpected outcomes](logs/pr2_mutants1_three.log):**
  - a phase-4 search crashed instead of failing as a check;
  - the severe threshold went untested because no fixture had a loss;
  - a little-endian phase survived.
- **[An equivalent mutant](logs/pr2_mutants2_equivalent_byte_order.log):** the byte order. Since 256 ≡ 1 (mod 5), an 8-byte integer's value mod 5 is its byte sum mod 5 in either order, so the order cannot change the phase. It was replaced by a first-4-bytes mutant.
- **[A truncation mutant](logs/pr2_mutants4_truncation_escape.log):** it was undetectable until a planted future leak made the truncated-tape check able to fire.
- **[A vacuous falsifier](logs/pr2_mutants5_vacuous_falsifier.log):** it had no trade after the decision.

## The Oct 23 census

| Run | Module | Result |
|---|---|---|
| [v1](census_20261023.json) | before 7d59bde7's acceptance items | 213,349 rows, 44,959 sampled ok. **Peak RSS 2,001 MiB**, too close to the 2,048 guard. Rows were held as dicts; the internal reading of 950 MiB missed the peak |
| [v2](census_20261023_v2.json) | the amendments (weights, provenance, causal checks, reconciliation), per-episode float32 frames | Same rows. Labels and causality PASS. Peak 1,143 MiB. **The [declared v1 / v2 comparison](compare_v1_v2.py) FAILED as declared** ([log](logs/pr2_compare_v1_v2_FAIL_row_order.log)). [Diagnosis](diag_order_v1_v2.py) ([log](logs/pr2_diag_order.log)): only the 2 no-path rows moved, from positions 45,553 / 95,540 to the end. Sorted by identity, statuses, sampling and labels are identical and all 107 input / context fields agree at float32. This exposed that the census was not in the manifest's order |
| **[v3](census_20261023_v3.json)** | the canonical order (`SORT_KEY`) | **[v3 = canonical(v2)](compare_v2_v3.py), every column exact** (213,349 rows × 137 columns) ([log](logs/pr2_compare_v2_v3.log)). Peak 1,466 MiB, including the sort's copy; 1,022 s |

**v3 readings** ([log](logs/pr2_census_20261023_v3.log)), each computed on the real rows:

| Reading | Result |
|---|---|
| Per-fill reconciliation of q, value, severe excess and the value identity, on every sampled ok row | **PASS: 44,959 checked, 0 failures** |
| Outcomes kept | 32,902 sampled rows with no fill; 312 post-only refusals at activation |
| Full-row causality on every 17th sampled row | **PASS: 2,765 checked, 0 failures.** It covers market features (the export's X5: future public inputs and receipt times moved, a same-time input fires), order inputs (every not-yet-notified fill moved or resized), and the client context recomputed on the tape truncated after the decision |
| [Receipt checks](receipt_checks.py) ([log](logs/pr2_receipt_checks.log)) | Weights sum to 1 in each of 508 episode × side × head groups and are 0 elsewhere. Absolute decision / END times and the mark source are on every episode row. The 91 inputs are in the export's order, and the 14 context fields are on every sampled ok row |

**Rows by source / head / status:**

| Source | ok | Other statuses |
|---|---|---|
| entry | post 5,235 | near_end 55 |
| retain | 33,103 | unsampled 113,164; post_pending 5,235; cancel_pending 5,755 |
| joint_post_keep | post 97, retain 3,351 | post: cap_blocked 6,826, near_end 108, unsampled 412. Retain: unsampled 13,388, post_pending 509, cancel_pending 709 |
| joint_chase | post 192, retain 2,981 | post: cap_blocked 7,013, near_end 113, unsampled 767. Retain: unsampled 11,919, post_pending 959, cancel_pending 1,456 |
| original | | no_path 2 |

**Support (diagnostic at this size;** [cells](support_cells_20261023_v3.csv), [categories](support_categories_20261023_v3.csv)):
- **All 28 side × head × horizon cells are INSUFFICIENT_TRAINING_SUPPORT.** With 128 originals per side, the largest cell has 127 distinct episodes, below the 200 floor.
- Per efac0452, the floor is not lowered and probes are not counted as episodes. Support is assessed on the full Oct 23 census.
- The census does reach the new states:

| Coverage | Range |
|---|---|
| Generations | up to 13 |
| Signed inventory | −30 to +30 shares |
| Other-side states | every one of them |
| Queue-ahead bins | every bin |
| Reservations | 0–30 shares |
| Generation age | 0–4,950 ms |
| t_to_end | 50–5,000 ms |

## The acceptance items of review 7d59bde7

| Item | Evidence |
|---|---|
| Stable selected original IDs and hashes; every status retained; original absolute END preserved; real-row identity and mark provenance | `selected_ids` and their SHA-256 are in the census JSON. Status rows are kept. `end_abs_ms`, and `t_to_end` / `o_horizon` measured to the original END, are checked. `ident` is on every branch; `end_mark` / `mark_source` are on every row |
| Absolute times, generation / side identity, acknowledged inventory, reservations, pending states; full-feature causal perturbations over public receipt times and private fill notices | `decision_abs_ms`; the client context. The causality PASS above, plus the synthetic hidden-notice and truncation falsifiers |
| Per-fill reconciliation of the neutral J moments, separate from paired labels; zero / no-fill / refusal kept | The reconciliation PASS above; the outcome counts |
| Unchanged market and known-state semantics; ordered names and units; deterministic sampling; episode-balanced weights | The builders are reused unchanged. The list hash is in the census JSON (`inputs_list_sha256`). Manifest phases are per row. The weights check passes |
| Support by the predeclared strata, distinct original episodes, explicit fallbacks; insufficient support reported without lowering the floor | The support tables above |

## Limits

- **Consumed data:** Oct 23 is the teacher day. No forward or validation claim.
- **Stand-in joint sources:** the R1 deterministic policies, not a teacher.
- **Single-order neutral targets:** never paired labels.
- **Gross values:** costs are explicit zeros.
- **Timing:** scenario B only.
- **Fixed slice:** the first 128 originals per side, which cannot carry teacher support.
- **The causal check is bounded:** every 17th sampled row.

**Next, per c68a072c / 7d59bde7:** after this review, the full Oct 23 teacher data with its support assessment. Then fit and freeze pi0-50
with its support contract, then candidate paired labels on Oct 24–25 and Oct 26–27 diagnostics. Each of these is consumed development data.

## Files

- **Module and harness:** hashes in [sources.sha256](sources.sha256).
- **Receipt scripts:** [compare_v1_v2.py](compare_v1_v2.py), [diag_order_v1_v2.py](diag_order_v1_v2.py), [compare_v2_v3.py](compare_v2_v3.py), [receipt_checks.py](receipt_checks.py).
- **Outputs:** the census / support JSONs (v1–v3) and the support CSVs. The census parquets (8–9 MB each) are hashed and kept in `data/predict_fun/research/r2_shadow_20261119`.
- **Logs:** [logs/](logs/).
