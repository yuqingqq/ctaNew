# REVIEW 111 — the pin sites, enumerated from the code

**REV, 2026-09-09T07:21Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. Enumerated in a scratch clone at the tip; the artifacts I read are real.

## THE ANSWER: REFUTED. THERE IS A THIRD SITE, AND IT IS THE SILENT ONE

**BE 111's site list — `be_cascade.modules` and `be_module` — is incomplete, and the site
it misses is the one that costs five hours rather than one night.**

**SITE 3: the day book's own `producing_code.import_closure` — 49 module digests, RECORDED
by the builder and READ BY NOBODY.** The book carries the scores (`asm["by_arm"]`); those
scores are computed at BUILD time by the scoring closure; the receipt records exactly which
bytes did it; and **no consumer ever compares that record to the code that consumes the
book.** The day run verifies the book's BYTES (`verify_day_inputs`) and the params' cascade
pins against the files on disk (`verify_be_module`) — both loud — so the receipt ends up
citing the CURRENT scoring code as though it produced numbers that older code produced.

**This is not hypothetical. It is true of every book on disk right now.** The newest book
receipt, `be_daybook_receipt_20260907_btc__L250ms.json`, records:

| module | recorded in the book | at the tip |
|---|---|---|
| `de_phase4_diag_runner.py` | `52e76689…` | **`c979fda5…`** |
| `de_head_scoring.py` | `60ef48fe…` | **`53a406a0…`** |
| `de_score_stream.py` | `f85be335…` | **`350b88ca…`** |
| `harmful_stateful_policy.py` | `14e669b1…` | **`1d10467f…`** |

**All four scoring modules have moved.** The only thing between that and a silently wrong
run is DE 155's shape guard (`ASSEMBLY_PREDATES_CAUSAL_SCORING`), which fires on the
float→dict transition — **a one-off**. Any future scoring change that alters values and
keeps the dict shape passes it silently, and the run's receipt will name the new pins.

**The fix is one predicate, and the information already exists in the artifact.** At the
start of a day run (and in `de_cancel_count_delta.measure_book`), compare the book
receipt's `producing_code.import_closure.modules` for the scoring-path modules against the
files on disk and REFUSE by name — `BOOK_BUILT_BY_DIFFERENT_SCORING_CODE`, naming each
module and both digests. That converts the silent site into a loud one at the run's start,
which is the survivable class.

**SITE 4, the same defect wearing a different filename: `de_section81_cache_12.pkl` has no
code pin at all** — not recorded, not checked — and is shape-guarded in **one of its
thirteen readers** (REVIEW 109). Same class, same fix: record the closure that built it and
check it at the load site.

---

## HOW I ENUMERATED, AND WHY IT IS CLOSED RATHER THAN MERELY UNFALSIFIED

**I did not read v23 or v31 to build the list** — that is the artifact under test. I
enumerated from the two operations that *constitute* a pin:

1. **a stored identity** — a digest, or a `{path, sha256}` pair, written into a declaration
   or a receipt; and
2. **a recomputation** — a consumer that hashes the thing again and compares.

**A pin can only refuse if some consumer performs (2).** So the silent set is exactly the
identities that are recorded and never recomputed — which makes the search constructive
rather than an argument from absence: I listed every identity-bearing key in the artifacts
on the critical path (the book receipt has twelve, plus the closure's 49 module digests),
then searched every consumer for a reader of each. Where the only readers name their **own**
identity — which is what `da_book_verify.py` and `de_multiday_gate1_runner.py` do with
`import_closure`, `producing_code` and `closure_drift`, every hit being
`source_identity_at_launch()` or the verifier's own `idy[...]` — the recorded value has no
recomputation and cannot refuse.

**Scope, stated so the closure claim is checkable:** closed over the modules that build or
consume a day book — `be_daybook_build`, `be_cancel_axis_null`, `de_multiday_gate1_runner`,
`de_phase4_diag_runner`, `de_head_scoring`, `de_score_stream`, `harmful_stateful_policy`,
`de_cancel_count_delta`, `de_early_read`, `de_decision_ledger`, `declaration_chain`,
`de_multiday_design_declaration`, `harmful_candidate_manifest`, `phase2_arms`, and the two
verifiers. Not closed over the rest of the package (there are ~280 digest sites in
`live/pm_research` overall, most in modules no day run touches).

## THE FULL SITE LIST

**LOUD — a consumer recomputes and refuses:**

| # | site | who recomputes | refusal |
|---|---|---|---|
| 1 | `params.be_module` (entry pin) | `verify_be_module`, first, on every path | *"BE's cascade module digest differs"* — **unnamed**, see below |
| 2 | `params.be_cascade.modules[0..9]` | `verify_be_module` when `actual_sha is None` | `BE_CASCADE_DIFFERS` |
| 5 | `stream_provenance.reader_module` → `exp_m6_settlement.py` | `chainlink_stream_provenance_check` (`reader_sha_matches_current_file`) | `SETTLEMENT_VERIFICATION_PROVENANCE_INCOMPLETE` |
| 6 | fit files: `arms[*].model_digests`, the manifest, `verify_head` | `_fits_digest`, `verify_pinned_models`, `verify_pinned_thetas` | `LGBM_NORMALISER_DIGEST_DIFFERS`, `…NOT_IN_MANIFEST`, pinned-model refusals |
| 7 | params ↔ design pair | the battery's chain check | fails the walk on both halves |
| 8 | declaration chain `supersedes {path, sha256}` | `declaration_chain` | chain refusals |
| 9 | book bytes ↔ BE's sidecar | `verify_day_inputs` | day refused before any stage |
| 10 | tape / fragment `expect_sha256` | the builder, at read time | build refuses |
| 11 | ledger `expect_sha256` | `read_ledger` | `DECISION_LEDGER_DIGEST_MISMATCH` |
| 12 | early-read supersession `{path, sha256}` | `early_read_preconditions` | `EARLY_READ_…` refusals |
| 13 | each runner's / verifier's OWN source, during its run | rule 22 drift | `producing_code_is_the_committed_bytes` |

**SILENT — recorded and never recomputed:**

| # | site | what it governs |
|---|---|---|
| **3** | **book `producing_code.import_closure` (49 modules)** | **the scores in `asm["by_arm"]` — the numbers every arm result is built from** |
| **4** | **`de_section81_cache_12.pkl`** | **a cached assembly with no recorded closure at all** |

**One loud site with a soft edge, carried forward from REVIEW 109:** site 1's refusal text
is *"BE's cascade module digest differs"* and carries **no `BE_CASCADE_DIFFERS` name** —
which is exactly how params v22 read clean to anyone grepping the named refusal. It refuses,
so it is survivable; it should still be named.

## WHAT THIS MEANS FOR THE BUILD THE USER STOPPED

The 09-03 build was stopped with nothing written. **Building it now is safe** — the book
would be built by the current closure and consumed by it. **The exposure is what happens
after**: DE's batch is in flight (findings B and C from REVIEW 110), DE 160 is editing the
null's sampling unit, and the scoring path has moved four times in one day. **A book built
this morning and consumed this afternoon, after any scoring-path edit, is the silent case** —
and the receipt would name the afternoon's pins.

So the ordering that makes the five hours worth spending is: **land site 3's predicate
first, then build.** It is one predicate over information the artifact already carries, and
it turns the only silent site on the critical path into a refusal at the run's start.

## ROUTED

1. **BE 111 — the site list of two is wrong; add site 3.** This is the finding worth
   interrupting for: a repoint guard over `be_cascade.modules` and `be_module` will not see
   a book built by different scoring code, because that identity is not in the params at all.
2. **DE — the predicate**: `BOOK_BUILT_BY_DIFFERENT_SCORING_CODE`, at the start of
   `run_day` and in `measure_book`, from `producing_code.import_closure.modules`.
3. **DE — site 4**: record the closure that builds `de_section81_cache_12.pkl` and check it
   at the load site (which REVIEW 109 already routed for the shape check).
4. **DE — name site 1's refusal.**
