# DE_OP_CROSS_REVIEW_LOOP — charter

Cross-plane review loop: the DE session reviewing `plans/OP_PLANE_PLAN.md`,
assigned by the coordinator 2026-08-23. Started 2026-08-23.

**Why DE reviews OP:** DE consumes the OP plane in three couplings its own
plans depend on — (1) the halt port and its two edges (the `DE-Constraints`
hard constraint stopping new risk; the priority `cancel_all` bypassing the
solver but not the Actuator); (2) the declared telemetry ports DE's four
acting modules publish to; (3) the ack-latency→τ-rung seam, where a
deployment-measured bound above the Class-D-frozen 1000 ms rung kills the
cancellation lever independent of any replay result. **If OP's plan does not
support those couplings as DE's plans assume, that is a MUST-FIX in one of
the two documents, and the finding must name WHICH.**

**Calibration, from the coordinator:** both coordinator-written plans failed
badly under cross-review (SP: 31 MUST-FIX; EV_GATES: 20 + REFUTED_IN_
SUBSTANCE). OPS wrote this plan in one pass under time pressure during a live
outage. Review it exactly as the DE corpus was reviewed — no softening.

**Method:** as `DE_PLAN_REVIEW_LOOP.md` (closed): independent reviewers with
distinct lenses; the coordinator-of-the-loop (DE) verifies every finding
against the files before recording it. Finding classes MUST-FIX / SHOULD-FIX
/ NOTE, each with a concrete failure case.

**Cross-plane boundary, binding:** `OP_PLANE_PLAN.md` is OPS-owned. **DE
records findings in this charter and the ledger; DE never edits another
plane's plan.** Findings whose fix belongs in DE-owned documents are applied
by DE directly, per the closed loop's practice.

**Stop rule:** two consecutive iterations with zero confirmed MUST-FIX
(zero-confirmed-MUST-FIX counts, the pinned semantics), or the coordinator
stops it. Verdicts: `DEFECTS_FOUND` (recorded for OPS) /
`DEFECTS_FOUND_AND_APPLIED` (DE-side fixes) / `CLEAN` / `BLOCKED(reason)`.

---

## Iteration log

(appended per iteration)
