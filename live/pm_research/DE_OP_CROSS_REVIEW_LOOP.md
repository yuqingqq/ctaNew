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

### Iteration 1 — 2026-08-23 — verdict: `DEFECTS_FOUND` (recorded for OPS; DE-side crumbs applied)

Three reviewers (couplings / soundness / claims-vs-artifacts); coordinator-of-
the-loop verified the three heaviest claims directly against the artifacts
(the empty `CancelAllStatus` consume set; the OP-Monitor ports-key absence;
the SP register's Class-D `tau_decision_rung` row vs OP's Class-A listing).
**Consolidated after dedup: 11 MUST-FIX + 7 SHOULD-FIX + notes. Fix-holder is
`OP_PLANE_PLAN.md` on every MUST-FIX; the DE corpus survives as the reference
semantics on all three couplings.** Multiply-hit findings (independent
lenses): τ=250 misclassification (2×, + the SP register's own erratum),
telemetry-premise-false (2×), ww_v1 scope contradiction (3×), §8b
tense/supersession (2×), §8d register staleness (2×).

**MUST-FIX (all in `OP_PLANE_PLAN.md`):**
1. **R-HALT is unevaluable as wired**: `CancelAllStatus.Unconfirmed ⇒ HALTED`
   is load-bearing, but NOTHING consumes `CancelAllStatus` in v22 and the
   plan's own Rule OP-1 freezes OP-Monitor's consume list without it — a
   gate that cannot fire, at the kill switch. [verified directly]
2. **Entering HALTED never issues `cancel_all`**: the plan routes and types
   the command but no sentence binds issuance to the transition; DE's entire
   carry analysis assumes it fired.
3. **The weak "no new risk" halt label appears in three places** — a third
   document a future reconciliation could cite to revert DE's
   `FeasibleSet = ∅` semantics (DE §6.2's queued relabel names this exact
   risk).
4. **§2's telemetry premise is false vs v22**: two of DE's four HealthEvent
   sources have no port until DE's queued additive fix lands, and §8 fails
   to hand DE that dependency.
5. **STALE has no HaltState consequence** — the registered-module-goes-
   silent case (the D-1b class the plan exists to catch) dead-ends before
   the halt.
6. **The closed-world registry reproduces D-1b** via the same human
   omission: no registry-vs-reality reconciliation, no unregistered-
   publisher rule, and the built analogue is hard-coded open-world without
   §7.2 recording it.
7. **§7.2(3)/§10's "on-box half is closed" is false** for the composite the
   plan's own §6.2 teaches: health timer dead + batch units exiting green
   (IDLE-as-success) = silent stall with no report. Testable today; never
   tested.
8. **τ=250 ms is tabled Class A ("no verdict turns on them") while being
   the R-1-frozen rung the `ww_v1` DEAD verdict is computed on** — inside
   the plan's own R-8 worked example, one paragraph after citing that
   freeze. [verified directly; SP §4 rows 175-176 + erratum §10.12
   corroborate]
9. **HaltState has no transition function**: nothing maps severities/
   liveness to DEGRADED vs HALTED; skip-rung legality unstated though two
   of the plan's own rules require it; DEGRADED has no producing rule at
   all (its REDUCING-ONLY consumer semantics are DE-owned and correctly so,
   but nothing on the OP side can ever enter the state).
10. **Reset semantics unspecified**: reason-list growth unbounded/undefined
    under flap; the fault-during-pending-reset race can erase an
    undiagnosed active fault; "the state records that it happened" has no
    archive mechanism.
11. **§8d's register sweep is stale against SP §4 Rev 5 as it stands on
    disk** (`refuse_k` now Class D — GUARD; `quote_size_pin` Class D —
    VERDICT; `verdict_coins` escalated as a quantifier domain with
    membership frozen in FLOW_MODEL_PROTOCOL_V4, not R-DUAL-governed; new
    Class-D rows unswept). Needs a re-sweep or a dated as-of stamp.

**SHOULD-FIX (all in `OP_PLANE_PLAN.md` except the mirror in 14):**
12. The third halt edge (`HaltState → DE-Actuator.halt_in`, DE's second
    door) is omitted from every fan-out description.
13. §8b argues in the present tense for the R-8 freeze §5.2 records as
    done — a textual lever to relitigate Class D.
14. No heartbeat-emission ask for DE's four acting modules (the liveness
    gap recursing one level up) — mirrored as a DE §6.2 line, applied
    DE-side this iteration.
15. The ww_v1 scope contradiction: "8/8 coin-days" (the R-9 day series)
    vs "one UTC day" (§8d), with R-9/R-11/the dayseries receipt never
    named.
16. §9's falsifiers cannot fire on the plan's own claims; none of the five
    self-flagged weak points gets one.
17. §7.1's retrofit map omits the built `TIER1_LOCK` check entirely
    (under-claim; the map is incomplete in the safe direction).
18. Cross-reference defects (§6 overloaded for an external audit; §8c
    mis-cite; §5.1/§5.2 swap; section order 8a, 8b, 8d, 8c on disk).

**Notes for OPS** (observation-time reconciliation of ~26 h vs the
dispatch's 21 h; the 47 ms best-end-of-band quote; the `ops/` vs
`data/pm_5min/ops/` ledger path; leg-2's unbound measurement source; the
claim-without-receipt on the synthetic stale-ledger test; the
declared-vs-observed-vs-multiple heartbeat-period three-objects question;
the honor-system residues in §7.2(1)/§2.1) are itemized in the reviewer
outputs and available on request.

**Verified clean, so the good does not get lost:** R-HALT's semantics
match contracts and DE's carry design; the τ-seam mechanics agree with DE
§5 sentence-for-sentence (operative rung, conservative degrade, kill bar,
favourable-ack-cannot-revive); sole-writer and bypasses-solver-not-Actuator
are correct everywhere; the §7.1 analogue discipline is genuinely
maintained; every quoted number traces to a real artifact (no fabrication
anywhere); and the plan's honesty apparatus (self-flagged weaknesses,
freeze-timing argument) is real and mostly holds. The failures are
specification gaps and staleness, not invention.

**DE-side applications this iteration:** DE §6.2's telemetry item extended
to cover `produces: HealthEvent` (not just ports) and the heartbeat-
emission line for the four acting modules added (findings 4/14 mirrors,
and the contracts ports-map self-inconsistency noted there).

Next iteration: after OPS applies, re-review the revised plan; stop rule
two consecutive zero-confirmed-MUST-FIX.
