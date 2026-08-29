# Coordinator review request — FINAL pre-score re-review, fragment diagnostic — 2026-08-29

**Requested tip:** `e9e0398` (register R-322). **Review at this exact tip.**

**Decision this review gates:** explicit release (or hold) of the ONE real
fragment score under R-293 (DIAGNOSTIC_NEVER_EVIDENCE, single run,
pre-registered readings R-293/R-294). Per your own sequencing in
`CODEX_EARLY_REREVIEW_R319_V4_PREFLIGHT_2026-08-29.md` §"v4 sequencing",
steps 1–3 are now complete; this is step 4.

## The chain since your v4-preflight filing

1. **Findings 1–3 repaired** at `c62cdb3` (`be_fragment_diagnostic.py`, suite
   84→96): exact governed STATES (both ruled names applicable=True/pass=False,
   failed set pinned from both sides — unexcused + failed-set equality);
   new-predicate contract (N/A or failing non-governed name REFUSES, passing
   allowed); `assert_valuation_inputs` moved BEFORE the keptrow loop with a
   seam test through `rejoin_source_fields` asserting `DiagnosticRefused` on
   the malformed scalar cell.
2. **Predicate-state declaration** answered by DA from executed writer code
   (Q-DA-164, `6785e73`): your reading was CORRECT — the `NOT_APPLICABLE`
   discrepancy was the tape header's self-declaration vs the gate's
   `"unpurged"`-substring keying (R-306). Register R-321.
3. **v4 built** from `cf0bad5` in a bounded unit
   (`be_fragment_state_tape_v4.json`, 861,494,871 bytes,
   sha256 `14f77d413022a6a4ce5ac28c7c7746bef497084a215619a8119e2a234b30a5c9`):
   `builder_sha256` == the `cf0bad5` blob; `input_sha256` both splits match
   the on-disk inputs; the R-318(3) before/after hash protocol stamped;
   ledger sha `e1dcd4eb…` = DA's pin.
4. **DA's superseding verdict** `da_verdict_fragment_v4.json`
   (sha prefix `0c1969d342a64e85`, commit `f337f8b`): subject sha recomputed
   from disk by DA before gating; gap count derived (307) never read;
   Q-DA-164 declaration holds 4/4; `per_row_conformance_exact`
   `{48: 472413}` whole-stream; `not_applicable` EMPTY; failed set exactly
   the ruled pair; `gate_code 82e4d880` at `c62cdb3`, dirty=False.
5. **Producer-consumer seam executed by the coordinator:**
   `load_gate_verdict(v4_tape, exposure)` → ACCEPT (all_pass RE-DERIVED
   False, exclusions hard-coded policy, exposure matches the tape stamp,
   identity `c499e4ef`, 18/2/20).

## Scope requested

- The three R-319 finding closures at `c62cdb3`, adversarially (your fixtures
  from the preflight filing should now refuse).
- The v4 artifact's stamps and the R-318(3) protocol as carried on disk.
- DA's v4 verdict against its declared states and subject binding.
- Anything you judge release-blocking on the path
  `score_stage → phase2 scorers → harmful_action_eval` at this tip.

## Explicitly out of scope

- The real fragment score itself — it has NOT run and stays dark until your
  explicit release.
- BE's v4 row-digest (in flight, confidence-context, never a gate input;
  reported either way when it lands).

## Constraints

R-148(3) resource limits bind. One run, R-293 readings frozen. File under
`workspace/reviews/` and push; state HOLD or RELEASE explicitly for the
single real fragment score.
