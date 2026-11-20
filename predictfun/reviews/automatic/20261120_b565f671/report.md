# R2-A v2 review — 2026-11-20

Pinned target: **b565f6714ddf37216ab7061f4bf8a1f64eb08efe**.
Range: **d10d36983b2f2baa358ea4ee0b170162984272c6..b565f6714ddf37216ab7061f4bf8a1f64eb08efe**.
Reviewed from `/tmp/pf-review-b565f671`, detached at that target.
[Development receipt](../../r2a_shadow_census_v2_20261119/report.md),
[prior review](../20261119_5432f636/report.md),
[artifact audit](artifacts.json), [independent cases](synthetic.json),
[reproduction script](verify.py), [pinned selftest](selftest2.log).

**The saved r2av3 receipt is supported. The stored-input rebuild, missing-source
statuses, refusal flags and support diagnostics are substantial fixes. Full-day
readiness is not yet closed: the collector still materializes the whole census
on return, and the venue-clock control needs a separate client-visibility test.
There is also a within-partition duplicate-accounting gap. No incorrect saved
label, current input leak or predictive improvement was established here.**

## Ranked findings

### P2 — the final return undoes bounded collection

`pf_r2_shadow.py:630` unconditionally returns `read_parts(pdir)`. That helper at
`:515–518` reads **every** partition into a list and concatenates them. This is
still reached when the caller ignores the return value. The partition loop now
releases frames, but the end of the production call reconstructs the full day.
The census receipt is written at `:623–628`, before this potentially failing
allocation, and the guard at `:608` does not cover it.

Independent bounded control: replace only `read_parts` with a sentinel. A
two-original, two-partition synthetic census reaches that sentinel **after both
the manifest and census receipt exist**. The ordinary small census returns
success, so this is a reachable production path, not dead code.

The saved 256-original frame occupies 167,398,415 bytes by deep pandas accounting.
Proportional scaling to 30,466 originals gives **18.55 GiB for the frame alone**,
before the read list/concat overhead, against the 4 GiB address-space limit.
This is an extrapolation, not a full-day memory measurement. The reported
1,246 MiB peak is valid for the four-part slice; it does not close this finding.

Required closure: full-day census returns paths/manifest and compact readings;
materialization must be an explicit, size-bounded test helper. Stream final
validation/support too. Prove the full-day route cannot call an all-part reader,
and do not publish a completed receipt before all required finalization passes.
The receipt-check scripts' `read_parts` calls are also slice-only until replaced
by partition-wise checks. No change to labels or training sampling is needed.

### P2 — preserving venue history leaves an unreceived-book blind spot

The venue-clock correction is appropriate **for reconstructing execution**.
It is insufficient as the only information-boundary check of the client context.

`venue_windows :364–368` preserves every frame with venue time <= decision.
`venue_future :378–382` modifies later frames and unnotified trade quantities.
`causal_rows :398–405` uses receipt-perturbed windows for market/order inputs,
but `rebuild_row :329` obtains the client context from the separately rebuilt
venue path. Thus a context that directly reads a venue-past frame not yet
received can escape both stored-vs-rebuild and stored-vs-perturbed comparisons.

Independent synthetic example, **no order yet at decision 0 ms**:

- A book frame has venue time 0 ms and receipt time 500 ms.
- Clean context has inventory 0; the clean causal check passes.
- A planted shared context error adds 0.01 × that unreceived frame's bid to the
  inventory input. Stored input becomes 0.64, and the entire current causal check
  still reports **1 checked / 0 failures / 0 not-evaluable**, with firing controls
  passing.
- Changing only that hidden frame's bid changes the leaked input to 0.65. There
  is no prior acceptance/fill notice in this example that could justify it.

This demonstrates a **test blind spot**, not an actual leak in the current
`client_snapshot`. Static review finds its normal fields using acknowledged
inventory/reservations and client state. The accepted R1 implementation remains
unchanged. The planted error is in the review process only; no source was edited.

Required closure: keep the venue-preserving replay check, and add an orthogonal
client-visibility control. At a captured decision state, preserve the delivered
public history and own-order notice history while making unreceived engine book
fields inaccessible or perturbing them **without rerunning past execution**.
The client observation/107-input construction must remain equal; a planted direct
read of an unreceived frame must fail. Include an absent/no-order case such as
this one and a live case, plus a delivered-information positive control. Avoid
rewriting already observed acceptance/refusal history to make the test fire.
The control must exercise the same observation path used by both builders.

### P2 — set reconciliation accepts duplicate originals within one partition

Missing-market and missing-pair exclusions are correctly retained now. However,
the duplicate claim is broader than the guard. The loader loop at `:549–554`
adds markets to `yielded` but does not refuse a second yield. The partition
reconciliation at `:600–601` compares **sets** of original IDs, which discards
multiplicity. It detects cross-partition overlap, not repeated status/action rows
within a partition. The full population set check at `:610` has the same limit.

Independent synthetic test: two selected original opportunities, both with
`missing_endpoint`, and one source pair yielded twice. The census returns three
original-status rows with **selected=2, represented=2, reconciliation=EXACT**.
The all-present positive control returns two rows. Missing-pair and missing-market
controls each return both originals with the intended missing-source status.

No duplicate exists in the saved r2av3 artifact; its canonical keys and exact
population check pass independently. The external receipt check detects repeated
original status rows, but the production collector's own “EXACT” reading does
not. It also needs candidate-key uniqueness for episode rows.

Required closure: before weighting/writing, enforce exactly one status row OR one
unique set of episode rows per original, never both; reject repeated source pairs
and duplicate canonical candidate identities. Keep the cross-partition check.
Add same-partition repeated-status and repeated-episode controls alongside the
existing dropped/cross-partition duplicate cases.

## Answers to the timing, retry and receipt questions

**(a) Venue versus receipt clock:** agree with the explanation for the rehearsal,
subject to the distinct visibility gap above. The saved decomposition attributes
all 51/1,372 differences to rebuilding the path, and none to market/order windows,
the fill-list perturbation or future fair alone. The recorded examples lose a
generation in the altered world. `Branch.view` reads `E.f[E.known]`
(`pf_branch_engine.py:298–305`), while refusal is legitimately known after
activation + POST_NOTICE 0 (`:244–258`). A frame can precede the decision at the
venue and still arrive later as public data. Changing that frame may change a
refusal already delivered to the client. That is not an invariance the original
replay check was entitled to require. The venue-clock fix does **not** establish
that every engine field is client-visible; the two checks serve different roles.
The broad “so nothing leaked” wording should be limited to the diagnosed
mechanism, rather than treated as a complete information-flow proof.

**(b) Retry/not-evaluable:** accepted. `:399–406` retries only when no path was
built, not after a failed comparison. At most eight deterministic seeds are used.
Unbuildable checks remain counted; `:617–621` requires zero such checks for PASS.
The earlier r2av2 FAIL remains disclosed and is superseded, not erased. The
pinned selftest exercises always-unbuildable, first-draw-only, missing-status and
shared-leak cases. r2av3 records two retries, zero remaining not-evaluable checks.
Retain this rule and record failed-draw reason/seed in the full-day audit trail.

**(c–d) Population, parts, memory and comparison:** the saved slice supports the
receipt, with the full-day caveats above. All 21 `sources.sha256` entries match;
r2av2/r2av3 output hashes and committed receipt/manifest/support copies match.
The pinned source, accepted engine, original orders and keep/pull receipt hashes
match. Four manifests each contain 64 deterministic originals and their union
equals the frozen selection, without overlap. Both missing-endpoint originals
have exactly one original-status row.

Saved as-of **2026-11-19T23:08:09.597010+00:00**, consumed Oct23:

| Item | Independently verified reading |
|---|---|
| Population | 256 originals, all represented; 213,349 census rows × 137 columns |
| Usable sample | 44,959 sampled ok rows; 508 original/side/head groups with total weight one |
| v3 comparison | All columns and dtypes exactly equal except `refused` on 289 joint POST sampled-ok rows |
| Refusals | Entry 312, joint_post_keep 6, joint_chase 12; retain rows carry no flag |
| Inputs | 91 export inputs + 16 context fields, float32 and unchanged ordered schema |
| Saved causal reading | 2,765 checked; zero in all failure/not-evaluable components; two retries |
| Saved label reading | 44,959 checked, zero failures, 32,902 no-fill outcomes retained |
| Resource evidence | Whole-process peak 1,246 MiB; partitions 326.85 / 352.73 / 303.90 / 252.57 seconds |

The nine receipt predicates reproduce. Independent saved algebra also reconciles
q/value/qm/severe, absolute decision/END times and horizon to the original END.
This review did not rerun the 2,765 checks on real tape or reconstruct discarded
per-fill lists; those counts are verified saved execution readings.

**(e) Support:** accepted as preparation diagnostics. All 28 coarse cells, category
and category-by-horizon distinct/nonzero-original counts, per-side/head numeric
ranges, missing-category statuses and WAIT/KEEP fallbacks reconcile. Inventory
sign uses the declared 1e-9 tolerance. Delayed entries/later generations are
reported separately; terminal other-side states are correctly disclosed as
unobserved. All 28 coarse cells remain insufficient because this slice has at most
127 usable originals per side/head, below the 200-original floor. These tables
are not a fitted or deployed support mask; the later teacher must enforce them.

## Prior findings: closed versus open

| d10d3698 item | This review |
|---|---|
| Stored 107-vector never checked | Closed for same-identity rebuild; residual client-visibility gap above |
| Missing source window silently drops original | Closed; multiplicity guard remains incomplete |
| Unbounded collector | Partition loop fixed; final all-part read remains open |
| Joint POST refusal flag | Closed; exact 289-row change and counts verified |
| 14/16 context typo and terminal coverage | Closed |
| Incomplete support diagnostics | Closed for this preparation stage |
| R1 preflight/engine | Remains accepted, source unchanged |

## ONE next priority

**Deliver the R2-A full-day launch gate: a genuinely bounded, resumable runner,
with the three controls above closed on synthetic fixtures and the same frozen
256-original partition artifacts.** Preserve valid labels/inputs/weights and the
explicit joint-refusal correction. Demonstrate uninterrupted versus interrupted/
resumed identity, checksums and aggregate readings; fail changed-input, corrupt-part,
overlap and stale-source resumes. A spy/sentinel must prove full finalization
never materializes all parts. Add a stop-rule firing fixture.

This is the first reviewable deliverable before launching all of Oct23. It does
not require another model, a target change, a full-day rebuild or an untouched
day. After it passes, run full Oct23 and assess support before any pi0-50 fit.
The following decisions settle the launch plan now.

## Full-day preparation decisions

**Population and existing gates:** all 30,466 original Oct23 opportunities
(15,233 per side), every status, deterministic canonical partitions <=64
originals (477 parts), unchanged shadow sampling, original END, size/cap 30,
episode/side/head weights and builder/J architecture. Keep the 200 distinct /
20 nonzero-original coarse support floors and explicit category/range fallbacks.
No new outcome-selected filtering and no teacher fit until its intended support
is assessed. Explicit zero fees/rebates remain accepted for **gross** research
values; 58/37/100 ms execution/notice effects remain in the engine.

**Full-day causal sample — declare `r2_full_causal_v1`:** within each original /
side / source / head stratum, let n be its sampled-ok row count. Select
`min(n, max(1, ceil(n / 64)))` rows by the smallest SHA-256 of a canonical key
containing this version plus every `SORT_KEY` field. Use a documented JSON array
encoding: dates as strings, identifiers/generation/times as integers in their
existing units, missing offset as JSON null; UTF-8, compact separators. Break
hash ties by the canonical key. No q, value, future mark or loss flag in selection.
This changes **audit sampling only**, not the training population or its weights.
All 107 same-identity/perturbed checks and firing controls run on selected rows;
all sampled-ok labels still get per-fill reconciliation. Synthetic visibility,
population, resource and resume controls run independently of this sample.

Record selected keys, seeds, draw statuses and component counts per part. Selection
must be identical after restart, repartitioning for a declared resource change,
or iteration-order change. Freeze this rule/source before the full-day run; never
resample around failed or not-evaluable checks. Their counts must be zero for
acceptance; a failure stops promotion and remains recorded.

Counting the rule on the saved slice, without replaying it, gives **1,598 checks**
(versus the previous 2,765). Linear full-day scaling gives about **190,174 checks**:
at 0.41 s/check plus 477 × 25 s of base work, **about 25.0 one-core hours**. This is
arithmetic, not a benchmark or confidence interval. Later-state density and
per-check cost may differ. Keep the approximately 40 core-hour planning ceiling;
update throughput estimates at checkpoints, not the audit rule to fit a budget.

**Resume contract:** before launch, commit an immutable plan containing all
selected IDs, partition memberships, source/dependency and input hashes, engine
settings, sampling version, ordered schema/dtypes, target/cost convention and
resource limits. Each completed part has an atomic completion record binding
that plan, its member IDs, parquet checksum, row/key/status/weight counts,
support contribution, label/causal readings, and elapsed/peak-memory measurements.
Commit these small manifest/checkpoint records through the development session;
large parquet files remain outside git. A path or filename alone is not a checkpoint.

Write a part to temporary storage; validate it before atomic promotion. Resume
accepts only validated completed records matching the exact frozen plan and actual
file hashes, refuses conflicts or overlapping originals, and reruns only missing/
unfinished parts. An interrupted write must never be marked complete. Failed
checks are failure records, not resumable passing parts. Recompute totals/support
from each accepted part exactly once; do not add old global counters again.
Publish the final COMPLETE receipt only when all 477 expected parts and all
acceptance checks reconcile. Current `:528` refuses existing output and `:611`
writes a manifest only after the loop, so this resume path remains to be built.

**Stop rule:** accept a checkpoint stop when **partition elapsed >900 seconds OR
measured peak RSS >1,536 MiB**, using a high-water measurement covering loading,
processing, sorting and writing. The current `rss_mib_after` at `:608` is not a
peak. Latch a breach when observed, finish an internally valid bounded checkpoint
only if the hard limits still permit it, then start no next partition and emit
`STOP_RESOURCE` (not COMPLETE). Retain the 2,048 MiB RSS guard threshold and
4 GiB address-space limit; enforce hard breaches during work rather than waiting
for checkpoint completion.
Include initial loading and final aggregation in the memory measurement. On a
breach, profile/checkpoint first; restarting does not authorize silently changing
membership, the audit sample or limits. Runtime/memory flags themselves need
known-bad controls, including a peak that recedes before the final RSS sample.

## Verification and limits

Pinned selftest independently passes (synthetic only, 67.41 s / 270 MiB). The
46/46 developer mutation outcomes and their source hashes were inspected; all
46 mutation replacement targets independently resolve. The full mutation suite
was not rerun. Independent artifact checks and synthetic cases used only the
codex slot via `run_light.sh`, with absolute script paths.

Five successful bounded invocations took **77.95 seconds**, maximum **694 MiB**.
One initial `/usr/bin/python3` invocation failed before testing because that
interpreter lacks numpy (0.17 s); the repository environment's `python3` was then
used. Final artifact audit as-of **2026-11-20T00:14:01.158499+00:00**. Evidence
hashes bind the final review script. No full labels, real-tape rerun, model fit,
untouched day, implementation edit, push or external message. Oct23 and the
previously consumed later development days remain consumed; none of these
implementation results establishes predictive skill or profitability.
