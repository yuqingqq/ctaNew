# EV_REPLAY_REVIEW_LOOP — dedicated review of `plans/EV_REPLAY_PLAN.md`

**Chartered 2026-08-23 under the coordinator's B3 tick + Ruling R-77.**
The plan predates this loop (it was reviewed inside the general
`DE_PLAN_REVIEW_LOOP` through its iteration 10, revisions 3–7 of the plan)
— this loop gives it the dedicated instrument the tick asks for, under the
R-61/R-62/R-77 regime from iteration 1.

**FROZEN LENS SET, declared at loop start (R-61(1)/R-77 — the set may not
grow without vacating the streak, and the loop may not close while any
lens here has never run):**
1. couplings (plane boundaries, engine/config seams)
2. soundness (claims are internally consistent and enforceable)
3. claims-vs-artifacts (every status cell checked against the code/receipts)
4. decision-variable (every pinned value: where was it ever varied)
5. graveyard (every dropped/deferred item: does its dropping reason hold)

**Stop rule**: two consecutive zero-confirmed-MUST-FIX iterations, or
marginal value per R-61(3). Every MUST-FIX names WHAT DECISION CHANGES AND
WHOSE (R-62). Fix-holder is the plan; DE is both author and reviewer
(the original DE loop's precedent), so applications land as plan revisions
and the next iteration re-reviews with fresh eyes.

---

### Iteration 1 — 2026-08-23 — verdict: `DEFECTS_FOUND` (2 MUST-FIX + 3 SHOULD-FIX). All five lenses RAN (R-77 satisfied from the start).

**MUST-FIX:**
1. **The header contradicts itself and has already misled a ruling.**
   Line 3 says "Revision 3"; line 9 says "Now Revision 7". R-67's
   programme-state line read "EV_REPLAY at 3" — the stale first line —
   and this tick's "draft the PLAN first" premise descends from the same
   misread. Decision changed and whose: the COORDINATOR'S programme-state
   accounting, twice. (The premise gap in the tick is therefore DE's own
   artifact defect, not a coordinator error — the correction is owed in
   both directions.)
2. **§1's dialect census is stale in the direction that matters.** "Five
   ad-hoc replay dialects" is now EIGHT: since the plan was written this
   session added `policy_bounds_v1.replay_multi`,
   `state_gate_v1.replay_sg` and `ww_ebx_v1.replay_ebx` — three more
   instrumented-copy engines, each conformance-locked per window to the
   reference under its own frozen protocol. The convergence claim ("the
   single environment these five converge into") is further from true
   than when written, and the coordinator's "everything downstream is
   measured through ad-hoc paths" is CORRECT in exactly this sense.
   Decision changed and whose: the NEXT replay author's copy-vs-env
   choice reads this section; as written it understates the divergence
   and does not name the de-facto pattern (conformance-locked copies
   under frozen protocols) as either adopted or debt.

**SHOULD-FIX:**
3. The precedence banner pins "contracts.yaml v22" — v23 is in force
   (R-68). Version-label staleness; the file pointer is unambiguous, so
   no decision changes.
4. §4.1's own trigger ("parity becomes a real gate at the FIRST
   non-reference engine") FIRED three times and the plan does not record
   it — each new engine satisfied parity via its per-probe `conformant()`
   gate rather than via `ev_replay`'s machinery. Record the mechanism
   that actually discharged the trigger.
5. §6.4's revive condition cites "the coordinator's 680-window re-sample
   ruling" — FORECLOSED (R-9 ran the day series instead; R-11 closed the
   family; DEAD across four channels at the achievable rungs since).
   The revive trigger should cite what could actually revive it now
   (nothing on current evidence).

**Lenses 1/2/4 returned clean**: the §0 boundary rules hold in all eight
dialects (no EV output enters any policy loop; receipts read post-run);
§4's status table matches the artifacts cell-for-cell (23 checks verified
this tick); every pinned value (seed, N_BOOT, 250 ms lag) is documented
with its reason.

**Applied as plan Revision 8 same-tick** (author-reviewer precedent);
iteration 2 re-reviews Revision 8 with fresh eyes next tick.

---

### Iteration 2 — 2026-08-24 — verdict: `DEFECTS_FOUND` (2 MUST-FIX + 1 SHOULD-FIX; decision-readiness ran per R-82, as in iteration 1)

1. **MUST-FIX (coordinator-required): the pattern disposition was spread
   across two sentences, not one unmissable word.** Decision changed and
   whose: the next replay author's default. Fixed — §1 now carries
   **"PATTERN DISPOSITION: DEBT — not ADOPTED"** as a standalone marker
   with the copy-conditions stated (frozen protocol + conformance gate
   only) and convergence named as this plan's open debt.
2. **MUST-FIX (self-caught): Revision 8's repair REINTRODUCED the defect
   class it fixed.** The history block retained a live-looking "**Now
   Revision 7**" — the exact string a prose-number reader (R-67's method,
   the R-79 class, four instances programme-wide) would hit next. Decision
   changed and whose: any accounting reading the file — the same failure,
   third time waiting to happen. Fixed structurally, not by another prose
   patch: an authoritative machine-readable `REVISION:` field is now the
   FIRST line under the title, the only place the number lives; every
   number in the history narrative is demoted to quoted history
   ("(then-marked: ...)").
3. **SHOULD-FIX (process): adopt `da_freeze_pin.py` for this loop from
   iteration 3** — sha256 pinned at dispatch, verified at report, breached
   iterations not streak-eligible (DA's instrument, 8 checks; the
   author-reviewer-same-session shape of this loop is exactly the pattern
   the pin exists to make evidence-able rather than trusted). ADOPTED in
   this charter effective iteration 3.

**Applied as plan Revision 9 same-tick.** Streak: 0 (both iterations
found defects). Iteration 3 runs against a PINNED Revision 9 next tick.
