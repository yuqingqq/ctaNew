# R2-A review — 2026-11-19

Pinned target: **5432f636c63c11e7e04d59be35320589b2cfcf64**.
Range: **7d59bde798d5f1e0399bdad44819f2d324b939b0..5432f636c63c11e7e04d59be35320589b2cfcf64**.
Source: `/tmp/pf-review-5432f636`, detached checkout.
Receipt: [R2-A](../../r2a_shadow_census_20261119/report.md).
Contracts: [c68a072c](../20261119_fefa5aca/report.md),
[efac0452](../20261119_c70a5db4/report.md), [7d59bde7](../20261119_5a4242b4/report.md).

**The core target and saved numerical results reconcile. R2-A is not yet ready
to scale into the full teacher build: two acceptance instruments have gaps,
and the all-in-memory collector cannot be carried over to a full day. Close
those in one R2-A superseding receipt; do not fit or tune a model yet.**

Evidence: [artifact audit](artifacts.json), [independent counterexamples](synthetic.json),
[pinned selftest](selftest.log), [reproduction script](verify.py).

## What is verified

The phase/offset and source design follows the contract. SHA-256 first-eight-byte
phase sampling, separate entry worlds, naturally evolved retained states, joint
POST_KEEP/CHASE roll-ins, original END, and first-live retained reviews are present
(`pf_r2_shadow.py:83–98,162–230`). Keeping the phase-4 4,950 ms entry as a
`near_end` status is consistent with preserving exclusions.

`hold_from :137–148` copies the actual decision state, holds the focal generation
without replacement, and keeps the other side's existing exposure. `target
:151–159` values the focal generation only: POST after activation; RETAIN after
decision + 37 ms. The quantity, current-mid-rebased movement and severe-excess
identities are consistent with the neutral J target, not a paired label.

`add_inputs :238–264` calls the unchanged market and known-state builders with
the generation's own activation and original absolute terminal. The input
list is the export's **91 plus 16 context fields = 107**. The enumerated list
in the original contract has 16 fields; the subsequent shorthand “14” was a
counting error, not a requirement to omit two fields. No model fitting or
architecture change occurred here. Client context uses acknowledged fills and
reservations; the R1 fixes remain closed.

Independent artifact checks verify all 15 listed hashes, the pinned source and
engine, committed receipt copies, the actual keep/pull receipt and order-table
hashes, deterministic selection, phases, offsets, retained sampling, unique keys,
canonical order, weights, absolute clocks and saved target algebra. **v3 equals
canonical(v2) exactly across every column.** Coarse support counts independently
reproduce. Earlier positional v1/v2 failure remains disclosed, not relabelled a
passing test.

Saved as-of **2026-11-19T19:49:42.100912+00:00**, consumed Oct23:

- **256 selected originals**, all represented: 240 eligible, 14 original
  post-only refusals, two original missing-endpoint status rows.
- **213,349 census rows**, **44,959 sampled ok rows**, **508** episode/side/head
  weight groups, each totaling one; weights are zero elsewhere.
- Saved receipt: 44,959 per-fill reconciliations with zero failures; 2,765
  causality checks with zero failures **under the narrower checker below**.
- **0/28** coarse support cells supported, as expected with at most 127
  represented training episodes per side/head/cell and a floor of 200.

No current saved original was shown dropped, no actual saved feature leakage
was demonstrated, and no saved q/value/qm identity failed. These are preparation
results, not predictive skill, net profitability or untouched validation.

## Findings ranked by severity

### P2 — the “full-row” causality test does not verify the stored input vector

At **`pf_r2_shadow.py:301–309`**, `causal_rows` tests a fresh feature-builder call
and fresh `known_steps` outputs. It does not compare those with the row produced
by `add_inputs`, nor rerun the whole adapter under perturbation. Only the context
fields are compared with the actual row at **`:310`**. In particular, the stored
market/order/fp values and `fair_edge` / `t_to_end` can be wrong without failing
this claimed full-row check.

Independent synthetic examples replace one stored field at a time—`bn_age`,
`fp_remaining`, `fair_edge`, `t_to_end`—with a future-END-derived value. All four
still report **1 checked / 0 failures**. Positive and known-bad controls work:
the clean row passes; a changed context is caught; a look-ahead feature builder
is caught. This isolates the missing link to the stored adapter output.

Required closure: verify the actual ordered 107-input row against a rebuild at
the same identity, then rebuild the full adapter with future public inputs,
receipt times and unnotified fills perturbed. Include the OWN/fp fields and the
archived-fair input's provenance in that contract. Plant corruptions/leaks in
the adapter output itself and require refusal. Appropriate dtype tolerances
should be declared, not selected after failures. Keep the current builder-level
checks; they are useful components but do not establish the broader claim.

This is an acceptance-test gap, not a finding that today's committed feature
builder actually leaks. Supersede the receipt's full-row claim accordingly.

### P2 — a selected original can disappear if its source window is not yielded

`census` selects originals at **`:370–371`**, intersects the source's market list
at **`:372`**, and emits original/status rows only inside `src.pairs(data)` at
**`:378–387`**. There is no final selected-ID versus represented-ID reconciliation.
An absent market or a selected market without a yielded pair therefore produces
no status row for that original.

Independent synthetic census: two selected originals with known
`missing_endpoint` status, but only one source window yielded. The census returns
success with **selected=2, represented=1**, silently losing original 2. The
all-present control emits both status rows. This uses no real tape and does not
assert that the current 256-original artifact suffered the problem; its ID set
is complete.

Required closure: retain every selected original independently of loader yields,
emit a specific missing-source/no-path status for the unrepresented cases, and
fail an exact population/duplicate reconciliation. Test missing market, missing
pair and all-present cases. Training weights and support denominators must be
computed only after that reconciliation.

### P2 — full-day collection is not memory bounded

The per-episode frames are accumulated in **`parts` at `:373,408`**, concatenated
at **`:412`**, then copied and globally sorted at **`:416`**. Per-batch guards at
`:409` do not make this a streaming collector or cover the final sort's peak.

The saved frame alone uses **167,406,757 bytes** by pandas deep accounting.
Oct23 contains **30,466 originals**, 15,233 per side. A proportional extension
of this slice implies roughly **18.55 GiB for that one frame alone**, before
the retained parts, concatenation, sort, feature caches and support copies.
This is an extrapolation, not a full-day measurement, but it is already far
outside the existing 2 GiB RSS guard and 4 GiB address-space launch limit.
The slice's 1,466 MiB peak cannot be treated as a flat full-day cost.

Required before the full day: process deterministic bounded partitions, finish
weights within each original episode/side/head, write them, and release their
frames. Preserve a canonical logical order and full manifest; aggregate support
without retaining all 137 columns. Demonstrate identical outputs on the frozen
slice and bounded memory across multiple partitions. No population thinning or
changed sampling is authorized as a memory workaround.

### P3 — joint POST refusals lose their outcome flag

Entry rows preserve `refused` at **`:189`**, but joint POST rows at **`:223–226`**
omit `gen_f['refused']`. The census refusal counter at **`:401`** treats the absent
flag as false. Thus “312 refusals” is not established as the total across all
POST source families.

Independent crossing-at-activation fixture: the entry and both joint probes
are refused by the engine. The entry carries `refused=True`; both joint rows
remain correctly `ok, q=0` but carry no refusal flag. The zero labels are retained,
so this does not change the neutral target. Preserve the outcome flag for joint
POST rows and reconcile refusal counts by source. The saved census does not
contain enough information to reconstruct any missing joint refusal count
without additional execution evidence.

### P3 — two receipt descriptions overstate/miscount coverage

The receipt says 14 context fields, while **`:71–73`** defines 16 and the stored
107-input list contains all 16. Correct the wording rather than change inputs.
It also says every other-side state is covered; the saved sampled rows contain
absent, post_pending, live and cancel_pending, but **zero terminal** rows on
either side/head. Terminal is naturally outside this decision horizon; disclose
that exclusion. Neither point calls for manufacturing terminal training rows.

## ONE concrete next priority

**Publish an R2-A superseding acceptance receipt that closes the stored-row
causality and no-drop population checks, preserves the joint refusal outcome,
and demonstrates partitioned output on the same frozen 256 originals.**
Include unchanged valid labels/inputs/weights under canonical identities,
firing counterexamples, complete status reconciliation and measured peak memory.
Complete the support diagnostics below in that preparation step. No full-day
build or teacher fit should precede this closure. The target, model architecture,
feature calculations, manifest sampling and parameter choices remain unchanged.

## Full Oct23 stage after that gate: population, support, resources and cost

These decisions are recorded now so the next implementation can proceed without
another plan round. They do not claim that the full build or teacher has passed.

**Population:** all **30,466 original Oct23 opportunities** in the hashed committed
orders artifact, both sides (**15,233 each**), every original status retained.
Apply the same deterministic offsets, retained sampling, joint roll-ins, original
END, 30-share size/cap and episode/side/head weights. Freeze all original IDs,
partition membership, source hashes and selected probe identities; do not choose
a subset by realized fill, loss, or support outcome. Restart from committed
partition manifests without duplicating episodes or renormalizing partial groups.

**Support before pi0-50 fit:** preserve the 200-distinct-original / 20-nonzero-
original floor per side/head/remaining-horizon cell. Every cell the teacher
will use must satisfy the relevant floors; unsupported cells remain explicit
fallbacks. Do not require an artificial 28/28 pass, nor lower floors to obtain
it. An entirely unsupported side/head cannot supply the planned teacher.

The current `support :336–360` reproduces coarse counts and marginal row counts,
but does not finish the declared support contract. Add per-category distinct
original and nonzero-original counts, the predeclared finer cross-cell tables,
and numeric-context ranges **per side/head**, not only pooled ranges. Keep
state-category and out-of-range fallbacks explicit. For example, the saved bid
POST age range is only 0 ms, while the pooled range extends to 4,950 ms; a pooled
range must not validate an unseen POST condition. Near-zero inventory roundoff
(about 1e-15 shares here) should use the existing numerical tolerance for the
inventory-sign diagnostic, not create economic long/short categories.

Report support/coverage for delayed entry and later generations separately from
offset-zero entry. Sparse fine cells are diagnostics, not new outcome-selected
filters. If the predeclared support leaves inadequate later-entry coverage,
revise the probe manifest prospectively; do not tune it on later-day outcomes.
Unsupported absent states WAIT, unsupported live states KEEP, subject to the
separate structural/risk constraints. Pending states process notices.

**Resources:** the measured slice was 1,021.82 seconds and 1,466 MiB peak RSS.
Simple per-original scaling gives **about 33.7 hours** on one core for Oct23;
this is a capacity estimate, not a measured completion guarantee. Loading can
amortize and later-state density can differ. Start with partitions of at most
64 originals: roughly 4.3 minutes at that rate. Target peak **at most 1.5 GiB**,
retain the existing **below-2-GiB RSS / 4-GiB address-space** limits, and measure
the final write/sort peak as well. If a 64-original partition exceeds 15 minutes
or memory exceeds the target, stop at its checkpoint and profile before scaling.
Use a resumable full-day budget around 40 one-core hours only after a bounded
multi-partition run supports it; revise that estimate from measured throughput.
No full-day experiment was launched by this review.

**Cost decision:** accept the explicit zero fee / zero rebate convention for
this gross research teacher and target. Preserve it consistently through J,
paired labels and value mapping; do not call the resulting values net profit.
Latency's execution effects are already represented by the 58/37/100 ms engine
clocks. The manifest's “unmodelled latency costs” should mean no additional
exogenous penalty, not that those execution effects are absent. An economic-cost
change would require a versioned target/controller comparison, not a late silent
adjustment after fitting.

## Tests, prior findings and limitations

R1 preflight remains accepted; its source hash matches the reviewed engine.
No earlier engine finding is reopened. The R2-A positional-order correction is
verified. The open items above belong to the adapter's acceptance and scale path.

Bounded codex-slot checks: pinned R2 selftest passes (synthetic only); independent
artifact audit; three synthetic counterexamples with positive/firing controls.
Developer 25/25 mutation outcomes were inspected, not fully rerun. The saved
44,959 per-fill and 2,765 causal reading counts were verified as receipt evidence;
this review independently checked saved algebra and exposed the causal check's
scope, without replaying the real tape. The discarded per-row fill lists prevent
a fresh per-fill reconstruction from the census alone.

Six successful bounded invocations consumed **24.59 seconds**, peak **783 MiB**;
final artifact audit as-of **2026-11-19T20:00:05.498612+00:00**.

The artifact audit was repeated after extending the independent script so its
final hash matches all JSON evidence. One initial shell redirection typo failed
before launching any research process. No fit, full labels, untouched day or
new real-tape experiment was performed. Seen Oct23–28 remain consumed development
data. Only review evidence and review-owned state are committed; other-session
changes are preserved. No implementation edits, push or external messaging.
