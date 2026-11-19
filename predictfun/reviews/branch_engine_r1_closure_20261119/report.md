# R1 closure: the saved-fork identity contract (review c68a072c answered) — 2026-11-19

**USER:** "... continue the implementing and next priority loop".

**Review c68a072c's next priority:** "Close the saved-fork identity contract and publish its narrow R1 closure receipt. Deliver the
guard/fingerprint repair with the counterexamples above, a valid serialization round trip, and unchanged valid-path ledgers."

This receipt supersedes [R1 v2](../branch_engine_r1_v2_20261119/report.md) in-band on the saved-fork interface only. The module is
[`pf_branch_engine.py`](../../research/pf_branch_engine.py).

## The finding at the code, and the repair

| Item (c68a072c) | Holds? | Repair |
|---|---|---|
| **P2.** `pair(state=...)` checked only tick and episode, bypassed the scenario / END guards, returned the *requested* settings, and the fingerprint omitted mark / END / scenario / tape / decision phase | Holds | A saved fork is now a **context**, `fork_state(...) -> dict`. It binds the contract, the tape id (SHA-256 of frames, trades and END mark), END, the mark, the settings actually used, the identity fields, episode, tick, focal side and the sides already processed at that tick. **All of these are inside the fingerprint.** `pair(..., ctx=)` runs `validate_ctx` before either arm: the print, the identity recomputed from the engine's own arrays, the tape and scenario of the call, tick / episode / side, and the processed sides. Metadata is returned from the executed context. `decide` refuses a side processed twice at one tick |
| **P3.** `pull_effect_equal` accepted any structural request | Holds | `pull_effect_ok` checks a structural request's declared landing time (max(req + 37, submit + 95) before activation, else req + 37). It has a unit test with known-bad times |

## Checks

**Codex's three counterexamples and the two extra cases are F11 falsifiers.** Each is refused by its own check:

| Falsifier | Check that refuses it |
|---|---|
| A state changed after saving (a planted fill) | The fingerprint |
| A changed mark, with the print recomputed | The identity |
| A changed END, with the print recomputed | The identity |
| Another tape | The tape |
| Another scenario: `cancel_effect_ms=50` | The scenario |
| A context of the other focal side | The side / processed sides |
| A context relabelled to the other side | The processed sides |
| A raw branch | The type |
| A context whose first side never acted (processed sides reset at 250 ms) | The processed sides |
| A tick-4 context used at tick 6 | The tick |

**Positive controls:**
- a valid context, and the same context after a **pickle round trip**, reproduce the direct pair (Y, fills, metadata);
- the fingerprint changes with each identity field (mark, END, settings, identity, tick, episode, processed sides).

**Selftest:** pass ([log](logs/pbe3_selftest3.log)).

**Mutation suite: 49 / 49** caught by a check ([log](logs/pbe3_mutants3.log)). New mutants: one per context check, the fingerprint
without identity, a double-processed side, and the untimed structural pull effect.

**Earlier runs, kept as provenance:**
- [pbe3_mutants1](logs/pbe3_mutants1_harness_pickle.log): a harness defect. The mutant module was not importable, so the pickle round trip failed on every entry. The harness now registers the module.
- [pbe3_mutants2](logs/pbe3_mutants2_two_masked.log): two mutants masked by a downstream check.
  - The first tick falsifier used ticks of different parity, so the processed-sides check also refused it.
  - The relabelled-side falsifier was also refused by `decide`'s double-processing guard.
  - Both now have isolating cases: a same-order tick, and a context whose first side never acted.
- The context's **side** check is logically implied by the processed-sides check: two contexts for the same tick always differ in which side has acted. It is kept as defence in depth, and the suite has no separate mutant for it.

## Unchanged valid-path ledgers

**Declared before the rerun:** [compare_v3_closure.py](compare_v3_closure.py) ([log](logs/pbe3_compare.log)).
- **Run:** slice v3 on the same frozen selection (Oct 24, 128 per side by oid) with the closure module ([json](slice_v3_20261024_closure.json)).
- **Result:** the 256-row orders table, the 1,380-row forks table, the branch histories and every reading are **identical** to the R1 v2 slice. The module changed. Slice v3 PASS.
- **Timed pull-effect predicate:** it changes no saved reading, which matches review c68a072c's independent check of the request delays.

## What this closes

The R1 engine gate, with the saved-fork interface that R2's label cache will use. The fill-notice P1 stays closed. **Next, per c68a072c:**
R2's first reviewable deliverable is a versioned shadow-probe / feature / support builder and its deterministic manifest
(`r2_shadow_sampling_v1`), on synthetic fixtures and the fixed first 128 Oct 23 opportunities per side, with no teacher fit.

## Files

- **Module and harness:** hashes in [sources.sha256](sources.sha256).
- **Comparison:** [compare_v3_closure.py](compare_v3_closure.py).
- **Outputs and logs:** the slice JSON; [logs/](logs/).
- **Data:** `data/predict_fun/research/branch_engine_20261119` (`slice_v3_20261024_closure*`).
