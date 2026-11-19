# R1 preflight review — 2026-11-19

Pinned target: **5a4242b4a0613523707a3c8c17507a32491de125**.
Range: **efac04521d7aa643efaeda1ad8cfe0fa987759b7..5a4242b4a0613523707a3c8c17507a32491de125**.
Source: detached checkout `/tmp/pf-review-5a4242b4`.
Prior review: [efac0452](../20261119_c70a5db4/report.md).
Receipt reviewed: [R1 preflight](../../branch_engine_r1_preflight_20261119/report.md).

**Clean within the reviewed scope. Prior P2 and P3 are closed. No new finding
was identified. R1 preflight is accepted for the declared bounded research
contract; the next priority is R2-A's adapter and Oct23 census receipt.**

Evidence: [independent cases](synthetic.json), [artifact audit](artifacts.json),
[selected mutation rerun](mutants.json), [final check log](final.log),
[reproduction script](verify.py).

## Findings and prior closure

There are no open severity-ranked findings in this commit. The committed
changes close the two prior findings at their source:

| Prior finding | Pinned source evidence | Verification |
|---|---|---|
| P2 effective engine state omitted — closed | `pf_branch_engine.py:368–379` lists and fingerprints `E.end`, `E.client`, `E.nf`, fallbacks and constraints alongside the earlier state. `:432–433` explicitly checks effective END, client mode, frame count and supported SETTINGS. | Both-side late-fill positive controls pass, all 20 corruption cases refused before arms with retained/recomputed fingerprint as applicable. |
| P3 missing marks become errors — closed | `Branch.__init__ :236`, `canon_mark :354–359`, `tape_id :365`, and `validate_ctx :430` use one missing representation. `finish :394–395` retains the missing-mark status. | None, NaN and both signs of infinity preserve missing statuses through direct and serialized contexts on both sides; finite marks still value fills. |
| Earlier structural-pull timing issue — stays closed | Existing `pull_effect_ok` timing checks unchanged; the timing mutant remains present. | Unchanged selftest passes and selected timing mutant is caught. |
| Prior hidden-fill P1 — stays closed | Client structural commands remain on client-known state; this patch does not change that logic. | Unchanged-module selftest passes, including F10. |

The execution guard runs before copying/executing either arm (`:448–456`). The
field inventory is descriptive; review also traced the actual fingerprint list
and `Engine`/`Branch` instance fields. Frame/trade arrays and mark are represented
by the recomputed tape hash, including the newly canonical missing mark.

The explicit contract checks remain effective even if someone recomputes a
corrupted context's fingerprint. Ledger logs are protected by the saved
fingerprint; the receipt correctly does not claim that arbitrary rewritten logs
plus a rewritten fingerprint are authenticated.

## Receipt, comparison and population

**The byte comparison and logs support the receipt.** The declared comparison
is against the original R1 v2 frozen Oct24 slice. It
requires identical orders CSV, forks CSV and histories JSON bytes, identical
readings and a changed engine source hash. The new module hash is
`8b7960ef4bbbc178b281e57d92d4795d7678436c0ca875e7199e24a6626af3a8`.

The independent audit verifies all seven source/output manifest hashes, the
committed JSON copy, old/new output hashes, dependency hashes and the actual
input receipt hash. All three ledger files are byte-identical; every reading,
setting and input identity agrees. The pinned comparator reruns successfully.
Independent unchanged/altered-byte and altered-reading controls prove that the
comparison detects differences. Prior reviews reconciled the identical saved
ledgers to fills and request clocks; no raw-tape rerun was needed here.

Saved slice as-of **2026-11-19T18:26:45.455460+00:00**: **256 orders** (128 per
side), **138 tapes**, **276 base runs**, **1,066 valid paired forks**, **38
not-live anchors**; the fork table contains **1,380 rows**. This is the same
consumed Oct24 implementation population. Base runs must not be counted as
additional paired labels.

The receipt retains the earlier failed tests. The one-frame count falsifier
was vacuous; the final fixture has two frames and changes the count to one.
The fingerprint-removal mutation was initially masked by explicit validation;
the added fingerprint unit test isolates it. The final developer log declares
the unchanged control passing and 56/56 mutants caught by selftest checks.

No performance claim follows from these controls. The policies are deterministic
stand-ins, values are the declared gross END marks, and Oct24 is consumed data.
This comparison is not untouched validation or evidence that a model selects
profitable quotes. As before, a commit containing comparison and outputs does
not independently prove the wall-clock order of the pre-run declaration.

## ONE next priority

**Deliver and review R2-A: the shadow-probe / feature / support adapter and
`r2_shadow_sampling_v1` manifest, synthetic controls, and the fixed first 128
original Oct23 opportunities per side, with no teacher fit.**

The existing [c68a072c contract](../20261119_fefa5aca/report.md) and
[efac0452 clarifications](../20261119_c70a5db4/report.md) remain in force.
No additional model, sampling, horizon, threshold or parameter amendment is
introduced by this review. Its acceptance receipt should make the following
concrete:

- Stable selected original IDs and hashes, all statuses retained, original
  absolute END preserved, and complete real-row identity/mark provenance.
- Correct absolute times, generation/side identity, acknowledged inventory,
  reservations and pending states; full-feature causal perturbations over
  public receipt times and private fill notices.
- Per-fill reconciliation of the neutral J moments and value, separate from
  the later paired-continuation labels; zero/no-fill/refusal outcomes kept.
- Unchanged market/known-state calculation semantics, ordered feature names
  and units, deterministic sampling and episode-balanced weights.
- Support counts by the predeclared strata, distinct original episodes and
  explicit fallbacks. The bounded slice cannot meet the 200-episode floor;
  report insufficient support without lowering the floor or treating probes
  as independent episodes. Full Oct23 support follows the preparation gate.

After R2-A passes, build full Oct23 teacher data and assess support before
teacher fitting/freezing and candidate paired-label generation. Cost convention
must be fixed before target/teacher fitting. Seen Oct23–28 remain development
data. No untouched-day result is claimed here.

## Checks and limits

Independent synthetic cases verify two late-fill controls (bid and ask), exact
direct/context/pickle results, 20 before-arm corruption refusals, and eight
side/missing-mark combinations. Each late-fill positive control gives the
expected **10 cents** on a planted 4,500 ms fill; these are synthetic values.
None/NaN/±inf cases retain missing statuses and contain no fabricated values.

The pinned unchanged-module selftest passes. All **eight selected mutants
(27–34)** are caught by selftest checks: seven new preflight mutations plus
the structural-timing regression. The developer's full **56/56** log was
inspected against the pinned harness names and expected outcomes; the full
suite was not rerun. F11/F7 source additions are at `:931–967`.

Two successful codex-slot invocations consumed **14.89 seconds**, peak
**210 MiB**. The first ran the synthetic check; after further capacity refusals,
the final invocation batched synthetic, artifact and selected-mutation checks
serially. This repeated only the small synthetic check and refreshed all
evidence against the final review-script hash. Refused launches did not start
research processes; capacity was retried through `run_light.sh`, with no bypass
or use of the other session's slot. Deferrals are retained in review evidence.

No fit, full label build, new tape experiment or untouched day was accessed.
This acceptance is limited to the reviewed research engine/preflight contract,
not venue-level fill realism or the correctness of the future R2 adapter. The
uncommitted `pf_r2_shadow.py` was outside scope and left untouched.

Only review evidence and review-owned state are committed. Other-session edits
and unrelated deletions are preserved. No implementation edits, push, external
message or scheduler action.
