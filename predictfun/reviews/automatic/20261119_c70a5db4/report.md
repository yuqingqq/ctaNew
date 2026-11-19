# R1 closure review — 2026-11-19

Pinned target: **c70a5db491075e4e65526ed23b41b65d0618d0e0**.
Range: **c68a072cd191c83a7a98919cacfa58e3b3365d6e..c70a5db491075e4e65526ed23b41b65d0618d0e0**.
Source reviewed in detached checkout `/tmp/pf-review-c70a5db4`.
Prior review: [c68a072c report](../20261119_fefa5aca/report.md).
Development receipt: [R1 closure](../../branch_engine_r1_closure_20261119/report.md).

**The named counterexamples are fixed, the timed structural-pull check is fixed,
and the declared valid-path ledger comparison is supported. Full saved-context
closure is not yet established: its fingerprint omits the engine's actual END,
which can disagree with the reported branch END and change a label undetected.
There is also a missing-mark status edge case. Neither invalidates the saved
slice. R2 adapter development can continue under the existing preparation plan;
close these small context-boundary gaps before accepting its label/cache gate.**

Evidence: [independent synthetic cases](synthetic.json), [artifact audit](artifacts.json),
[selected mutation rerun](mutants.json), [review script](verify.py).
This review changes no implementation and makes no predictive or profitability claim.

## Resolved versus open

| Previous request | Conclusion / pinned evidence |
|---|---|
| Supplied scenario, mark, branch END, tape, tick/episode and focal side must agree | Named cases fixed. `validate_ctx`, `pf_branch_engine.py:410–421`, runs before arm construction at `:431–440`. Independent counted-decision controls show zero arm decisions on nine refused cases. |
| Identity in fingerprint; metadata from executed context | Listed branch identity fields are included at `:358–364`; result metadata comes from the validated context at `:452–453`. The omitted execution fields below prevent claiming a complete-state fingerprint. |
| Pickle positive control | Fixed. Independent direct/context/round-trip comparisons have exactly equal complete arm dictionaries and Y. F11 also checks the round trip, `:884–891`. |
| Double-processing protection | Fixed for the declared phase path: `decide :340–343`, `acted` reset at a new tick, and expected already-acted sides checked at `:420–421`. |
| Structural pull timing (called P3 in the closure receipt) | Closed. `pull_effect_ok :502–511` tests request and effect time, including pending cancellation. Positive and wrong-time examples pass/fail as expected. New mutation 27 is caught. |
| Valid-path ledgers unchanged | Supported on the declared frozen Oct24 slice, as detailed below. No unobserved-population claim follows. |
| Prior hidden-fill P1 | Stays closed. The unchanged-module selftest passes, including F10. No new evidence reopens that causal issue. |

## Findings ranked by severity

### P2 — effective engine END is outside the saved-context fingerprint and validation

The executable horizon is **`Engine.end`**, stored at `pf_branch_engine.py:108`
and used to stop consuming trades at **`:153–155,176`**. The fingerprint at
**`:362–363`** includes `Branch.end` but omits `Branch.E.end`. The identity check
at **`:415–416`** likewise checks only `Branch.end` against the context and END.

Independent synthetic reproduction, with no recomputation of the fingerprint:

1. Save a valid tick-4 bid context under HOLD_ONCE. The tape has one 10-share
   bid fill at 4,500 ms, price 50 cents, END mark 51 cents.
2. The normal KEEP-minus-PULL value is **10 cents**.
3. Change only `ctx['state'].E.end` to 4,000 ms. The fingerprint is **unchanged**.
4. `pair(..., ctx=...)` accepts it, reports END **5,000 ms**, but skips the
   4,500 ms fill and returns **0 cents**.

This is a synthetic corrupted-context test, not evidence of spontaneous tape
corruption or incorrect normal slice results. It is directly within the guard
contract being closed: even a changed effective END must be refused before
either arm executes. The existing F11 altered `Branch.end`, leaving this second
execution clock untested.

The same fingerprint also omits **`Engine.client`** (selects engine/client
structural processing at `:180`) and **`Engine.nf`** (limits frame traversal at
`:128,162,178`). Their fingerprint omissions were independently confirmed;
the demonstrated value-changing case is END. Do not call these separate
economic defects without a valid-path reproduction. `Branch.fallbacks` and
`Branch.constraints` are also omitted despite being returned in ledgers at
`:384`; include them if claiming complete saved-ledger integrity.

Required bounded repair: fingerprint the semantic execution state, or make
derived/constants immutable and validate them explicitly. At minimum require
`E.end == B.end == ctx.end == END`, `E.client is True`, and `E.nf == len(E.f)`
for this client context; compare the context's settings with the supported
`SETTINGS`, not just its duplicate record. Retain the existing tape, identity,
phase and tick guards. Add per-field corrupt-context refusals before arms,
including a late-fill positive control and unchanged direct/pickle ledgers.
Inventory the fields once so this closure does not rely on testing only one
name for a duplicated clock. This is a small validation patch, not an engine
redesign or a reason to rerun the full market experiment.

### P3 — NaN END marks become context errors instead of `missing_mark` statuses

`finish` at **`:379–380`** explicitly treats both None and nonfinite marks as
missing. The new equality **`B.mark == ctx['mark']` at `:415`** is false for a
NaN mark even when the context was just created from that same input.

Independent otherwise-identical direct POST probes return `missing_mark` for
None, but raise `ValueError` for NaN before reaching the missing-mark status.
The finite-mark positive control succeeds. No NaN mark was demonstrated in
the saved eligible slice. The risk for R2 is that a censored opportunity aborts
a build or becomes a generic refusal rather than the declared missing-mark
population.

Canonicalize nonfinite marks to the declared missing representation at the
input boundary and retain `missing_mark` in the census; alternatively use a
missing-aware identity comparison. Include None/NaN/nonfinite fixtures and a
finite-mark positive control. Do not silently drop these opportunities or
substitute a zero price. This can be handled in the R2 adapter preflight, but
the generic pair API should document its accepted missing-value representation.

## Does the unchanged-ledger comparison support the receipt?

**Yes, within its exact scope.** The independent audit verifies the committed
engine, mutation harness and comparator hashes; the closure JSON's committed
copy; all output hashes and input receipt identity; and dependency source hashes.
The execution module hash changed from `13239a24...` to `98e3824b...`.

The old and closure **orders CSV, forks CSV and histories JSON are byte-for-byte
identical**, stronger than the receipt's DataFrame/parsed-JSON comparison. Their
settings, inputs and every reading are identical. Population at saved as-of
**2026-11-19T18:12:39.646391+00:00**:

- **256 orders**, 128 per side, on consumed **Oct24**;
- **1,380 table rows** = 276 base runs + 1,066 valid forks + 38 not-live anchors;
- **1,066 paired histories**, corresponding to the valid forks;
- 138 distinct tapes as previously verified in c68a072c.

The raw `ok` status count is 1,342 because it includes the 276 base runs; it is
not 1,342 independent paired forks. The comparator's positive and altered-value
controls fire. Prior c68a072c independently reconciled these identical ledgers
against fills, values and request clocks; this review does not repeat raw tape
replay. Thus the current findings do not overturn a saved reading. This is
deterministic-policy implementation evidence, not learned timing skill, net
P&L or untouched validation.

The committed comparator expresses an exact equality requirement. Its commit
contains both comparison and results; this review does not independently prove
the wall-clock order of the claimed pre-run declaration. No model-selection or
forward-validation credit is assigned to this implementation comparison.

## ONE next priority

**Complete the saved-context preflight with the effective engine-state guards
and missing-mark handling, then publish its narrow closure evidence.** Required
acceptance: the existing context/round-trip controls still pass; the altered
effective END is refused before either arm; derived engine fields cannot silently
disagree; missing marks remain declared statuses; valid ledgers remain unchanged.
Use synthetic cases and the already-saved comparison. No fit, full label build
or new tape run is needed to settle this issue.

R2-A's first deliverable remains the **shadow-probe / feature / support adapter
and `r2_shadow_sampling_v1` manifest**, synthetic fixtures and the fixed first
128 original Oct23 opportunities per side, **no teacher fit**. Development may
continue building it in parallel; acceptance of its labels/cache requires the
preflight above. The remaining issue is not a reason to revise the model or
retune the prediction architecture.

## R2 preparation amendments / clarifications

The c68a072c preparation contract remains authoritative for offsets, horizons,
phase sampling, episode weights, support strata/fallbacks, context feature names
and units, and unchanged `real_features`, `known_steps`, `order_inputs` math
and J architecture/parameters. **No numerical sampling or model change.**

1. **Context preflight:** validate the actual engine state as above. Populate
   `ident` at the adapter boundary with original instrument/episode identity,
   original absolute start and END, and mark-source provenance; do not leave it
   `{}` for real rows. These are label/join metadata, not forecast inputs.
   Source/content hashes belong in the manifest. Never use END marks as inputs.
2. **Missing data:** normalize None/NaN/nonfinite marks consistently and keep
   their `missing_mark` rows/status counts. Test that this cannot abort the
   census or become an outcome-dependent selection rule.
3. **The bounded population is original opportunities:** select the first
   128 per side by stable original ID before outcomes or path availability;
   retain ineligible/no-path rows with statuses. Do not copy the R1 slice's
   eligible-only population definition into this R2 census.
4. **Support counts are diagnostic at this size:** 128 original opportunities
   per side cannot establish the declared 200-distinct-episode support floor.
   Report `insufficient_training_support` where appropriate. The small receipt
   can pass implementation/causality checks without claiming teacher support;
   assess that support on the subsequent full Oct23 training census. Do not
   lower the floor or count descendants as new episodes to pass the slice.

After that preparation gate: build full Oct23 teacher data, assess support,
fit/freeze PI0-50, and only then generate candidate paired labels. Previously
seen Oct23–28 are consumed development data, not untouched validation. Resolve
the cost convention before targets/teacher fitting as already required.

## Tests, limits and ownership

- Independent synthetic check: nine context disagreements refused before any
  arm decision; exact valid direct/context/pickle arm identity; six pull-time
  controls; effective-engine-END counterexample; missing-mark boundary.
- Independent saved-artifact audit: hashes, byte equality, populations, settings
  and comparison falsifiers. Audit as-of **2026-11-19T18:16:19.207972+00:00**.
- Pinned harness rerun: unchanged module/selftest passes; all **10 selected
  closure mutants (18–27)** caught by selftest checks. Full developer log's
  unchanged control plus **49/49** mutant entries inspected; no full rerun.
- Three successful codex-slot invocations, **17.87 seconds**, peak **218 MiB**.
  One prior environment-only attempt failed before importing NumPy because
  `/usr/bin/python3` lacked the dependencies; retained in `synthetic.log`.
  The successful attempt used the environment's `python3`, `synthetic2.log`.
- No fit, full labels, raw tape experiment or untouched day accessed. This does
  not establish venue-level fill validity outside the bounded engine contract.

Only review evidence and review-owned state prefixes are committed. Other-session
edits and unrelated deletions are preserved; no implementation change, push,
external message or scheduler action.
