# Seat protocol — P-2026-003 (consolidated roles & rules)

**Status:** consolidation, not new law. The register (`COORDINATION.md`) remains
the authority; every rule here cites its R-entry. On any conflict, the register
wins. Committed 2026-08-28 by the coordinator; amendments by coordinator commit,
except where marked USER-ONLY.

## Seats

| seat | session (tmux) | owns | never touches |
|---|---|---|---|
| **Coordinator** (pm-co) | this session | the register (R-entries, append-only); rulings; dispatch; verification of every result-bearing claim at its artifact; collector surface changes (R-110); driving the Codex review rounds; boundary deploys | seats' in-flight files; frozen artifacts (rule 13) |
| **BE** (pm-be) | ctanew-fe | phase2/011 model code, fit/score runs, receipts, freeze receipts | STATUS/HANDOFF (R-233); collectors; DA's instruments |
| **DA** (pm-da) | ctanew-1e | independent verification stack, tape gate, day-verdict tool, its Q-filings, peer annotations (sidecar owner) | STATUS/HANDOFF (R-233); collectors; BE's generators (reads, never edits — separate implementations are the point, R-235: do-not-harmonize) |
| **MEM** (pm-memory) | ctanew-ba | STATUS.yml, HANDOFF.md, memory docs, TODO checkbox true-ups (tick-with-citation only) | prose of plans; register; receipts; any result-bearing artifact |
| **CODEX reviewer** (pm-codex) | user's Codex session | review filings under `workspace/reviews/` (commit+push); holds and releases | fixing code itself; state files (first filing predates this rule) |
| **USER** | — | freezes (rule 12); frozen-doc amendments; CLAUDE.md; collector deploy approval; race admission; anything marked USER-ONLY | — |

## Standing rules (with their register cites)

1. **Verify at the artifact** (CLAUDE.md rule 16): no seat's claim — including
   the reviewer's — is accepted from a report. A claim is a reproduced defect
   or a review error, established by execution, filed either way (R-238).
2. **Red-first**: every fix ships with a known-bad that FAILS on the pre-fix
   code, plus a positive control. A fixture must never supply what the code
   under test should produce (R-229 class); quiet and empty are different
   (R-236); run the entry point the way the launcher runs it (R-240 cycle).
3. **Corrections supersede in-band** (rule 13): frozen artifacts are never
   edited; superseding versions carry a supersedes block; a citation
   correction never restarts a clock (freeze v2, R-236/R-238-adjacent).
4. **Frozen docs are amended only by the USER** — seats draft
   DRAFT-FOR-USER-FREEZE; nobody amends a design after seeing it (R-237).
5. **Review protocol** (R-239, refined R-240): build → commit+push →
   **one Codex review round per COMPLETED batch** (never piecemeal) → Codex
   commits+pushes its filing → coordinator verifies claims → fixes red-first
   accumulate into the next batch → re-review executes the exact batch commit.
   A hold releases only on the reviewer's explicit HOLD RELEASED.
6. **State-file ownership**: MEM writes STATUS/HANDOFF; BE/DA commit artifacts
   and file facts (R-233). CAVEAT: CLAUDE.md still instructs every session to
   update these files — a USER-ONLY amendment is pending; until it lands, a
   fresh seat following CLAUDE.md is behaving correctly and MEM sequences
   around it.
7. **Collector surface** (R-110/R-181/R-182): coordinator-owned; changes need
   a USER ruling; deploys only at a UTC day boundary with a `collector_runs`
   era stamp; verification is structural, never a throughput A/B.
8. **Caps are never raised** (R-174). Lowering a unit's cap for
   attributability is permitted (R-238 cycle).
9. **Models estimate; policy decides** (rule 14): no worker boolean encodes an
   entitlement; advancement rules live with the USER.
10. **Numbers of record**: declared before results (rule 6); determinism
    repairs need pre-committed sight-unseen acceptance (R-234); the seed must
    pin the data the RNG is applied to, not just the RNG.
11. **Silent success is failure**: a run that writes nothing must not exit 0
    (R-238 cycle); absence must never read as a pass — expected sets,
    coverage evidence, and optional members are producer-recorded facts,
    never checker assumptions (R-230).
12. **Timestamps**: clock read in a separate call BEFORE composing any entry
    (four slips on record); every population carries n and as-of.
13. **Escalation**: a seat that cannot rule (frozen numbers, another seat's
    surface, USER-ONLY matter) escalates to the coordinator rather than
    acting; refusing to act on ambiguity is the correct move on record
    (BE's canon refusal; DA's sorted-wins refusal).
14. **Peer messages**: relayed authority is verified at the user's own
    committed text when a ruling expands a seat's surface (DA's d506a06
    check — the model).

## Cadences

- Day verdicts: 00:06Z per coin; 08-28 under the OLD count bar; 08-29+ under
  day-bar v2 ONLY after the reviewer's HOLD RELEASED (R-238, Codex filing
  7954585).
- Race accrual: freeze-commit epoch (1787897340); accrual ≠ day quality
  (split_verdict, R-240).
- Boundary deploys: 00:00:00Z exactly, per runbook, era-stamped.
