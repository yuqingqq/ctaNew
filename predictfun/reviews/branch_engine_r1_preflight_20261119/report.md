# R1 preflight: the effective engine-state guards and missing marks (review efac0452 answered) — 2026-11-19

**USER:** "... continue the implementing and next priority loop".

**Review efac0452's next priority:** "Complete the saved-context preflight with the effective engine-state guards and missing-mark handling,
then publish its narrow closure evidence."

This receipt supersedes the [R1 closure](../branch_engine_r1_closure_20261119/report.md) in-band on the saved-context guards only. The module
is [`pf_branch_engine.py`](../../research/pf_branch_engine.py).

| Finding (efac0452) | Holds? | Repair |
|---|---|---|
| **P2:** the engine's own END (`E.end`), its client mode and frame count were outside the fingerprint and validation. A context with `E.end` = 4,000 ms skipped a 4,500 ms fill and returned 0 c instead of 10 c. The fallback / constraint logs and a check against the *supported* settings were also missing | Holds | `FINGERPRINT_FIELDS` inventories the fingerprint once. It now includes `E.end`, `E.client`, `E.nf`, the fallback and constraint logs, and the earlier fields. `validate_ctx` also requires `E.end == B.end == END`, `E.client is True`, `E.nf == len(E.f)`, and settings equal to the supported `SETTINGS`, not only to their own record |
| **P3:** a NaN mark became a context error instead of a `missing_mark` status | Holds | `canon_mark` maps None / NaN / ±inf to None at the input boundary (Branch, tape id, validation). Pairs with a None / NaN / inf mark return status `missing_mark`; a finite mark is the positive control |

## Checks

**F11 additions:**
- **Positive control:** a late fill at 4,500 ms gives Y = +10 c, directly and after a pickle round trip.
- **Corrupt contexts:** each is refused before either arm runs, both with the fingerprint kept (the fingerprint catches it) and with it recomputed (the explicit check catches it).
  - the engine END changed;
  - the client mode changed;
  - the frame count changed;
  - the scenario changed in both the record and the state;
  - a planted fallback or constraint (fingerprint only).
- **Fingerprint unit test:** extended to every new field.

**Missing marks:** a None, NaN or inf mark gives the `missing_mark` status (F7).

**Selftest:** pass ([log](logs/pbe4_selftest4.log)).

**Mutation suite:** **56 / 56** caught by a check ([log](logs/pbe4_mutants2.log)). New mutants:
- the engine END check removed;
- the client-mode check removed;
- the frame-count check removed;
- the scenario checked against its own record only;
- the engine execution fields outside the fingerprint;
- the ledger logs outside the fingerprint;
- a non-canonical NaN mark.

**Kept as provenance:**
- One masked mutant: the execution fields outside the fingerprint were hidden by the explicit validation until the fingerprint unit test covered them ([log](logs/pbe4_mutants1_one_masked.log)).
- One vacuous falsifier: the frame-count tamper on a one-frame tape. It now uses two frames ([log](logs/pbe4_selftest2.log)).

**Valid ledgers byte-identical** ([compare_bytes.py](compare_bytes.py), declared before the rerun; [log](logs/pbe4_compare.log)):
- slice v3 on the frozen Oct 24 selection with this module: the orders CSV, forks CSV and histories JSON have the same SHA-256 as the R1 v2 slice;
- every reading is identical;
- the module changed;
- slice v3 PASS ([json](slice_v3_20261024_preflight.json)).

**Next, per efac0452:** R2-A's first deliverable, the shadow-probe / feature / support adapter and the `r2_shadow_sampling_v1` manifest.
It runs on synthetic fixtures and the fixed first 128 original Oct 23 opportunities per side, of any status, with no teacher fit. The
review's amendments apply:
- `ident` is populated at the adapter boundary;
- missing marks stay `missing_mark` rows;
- support at this size is a diagnostic (`insufficient_training_support` where it applies).
