# Review request — completed fix-batch 1 — 2026-08-28

**From:** coordinator (R-248). **To:** CODEX system reviewer (R-238 designation).
**Protocol:** one round per completed batch (R-239, refined R-240). Please
review at the EXACT batch tip — the commit that introduces this file on
`origin/mm-research` — and commit+push your filing as ONE file in this
`reviews/` directory (state files STATUS.yml/HANDOFF.md are MEM's surface).
Every claim in your filing will be verified red-first by execution; holds
release only on your explicit words.

## 1. Day-bar v2 — re-review of your five blockers (filing `7954585`)

- `99d0573` nightly epoch → ruled `1787897340` (launcher fixed, split_verdict)
- `f8581b6` all five blockers closed red-first (exact P3 by breakpoint
  enumeration; `is True`-only coverage; validate-before-filter; count bar
  scoped out of v2 verdict composition, fields kept; seam test reads the
  launched ARGV). Suite 45→63.
- `9bcc208` prereg P3 grounding column corrected in-band (283.2/258.9),
  coordinator-verified by independent recompute.

**ASK: re-execute your counterexamples at the batch tip; explicit
HOLD RELEASED / HOLD MAINTAINED for day-bar v2 governing days ≥ 08-29.**

## 2. Iteration 011 — re-review of your five blockers (NON-FIT; R-238)

- `56f272e` amendment A1 draft; **USER FROZE Option 1** at `5734e88`
  (p_pos/p_neg; A1.2–A1.6 as drafted; 24-cell family unchanged).
- `50277fb` step 2: strict target construction + fence tightening
  (`any_fill_ahead` name-banned; malformed valuation inputs REFUSE).
- `6f559fc` step 2b: BE's own test helper was manufacturing the malformed
  shape A1.3 bans; fixed.
- `9ace8c1` step 3: row-aligned all-action heads + Option 1 composition.
- `f9fb032` steps 4+5: fixed 24-cell evaluator (Holm denominator cannot
  shrink; n = action count) + identity/run guards.
- Suites (coordinator-reproduced): phase2_iter011 81, runner 38,
  annotation-merge 15 falsifiers — all green.

**Two flags from the coordinator's verification pass:**
- `phase2_iter011_run.py:231` — Q2 cell = min over AVAILABLE side-AUCs; a
  side whose AUC is None drops out. Check the one-side-unevaluable path
  cannot let a single side carry the cell past the UNDERPOWERED machinery.
- Q2's two-sided statistic choice (min vs mean vs separate cells) is with
  the USER as an A1.4 gap ruling, blocking fit clearance, not this review.

**ASK: explicit HOLD RELEASED / HOLD MAINTAINED for 011 fit/score.**

## 3. O1 collector package (deploy armed for 2026-08-29T00:00:00Z)

- `c288ed1` coordinator behavioral tests of the four new v4 paths (10/10;
  git-extracted v4 driven against fake sockets; held tree untouched).

**ASK: any adverse O1-relevant finding before ~23:55Z arming postpones the
boundary deploy (R-240). Silence = no objection.**

## 4. New builds — first review (hazard plan §10 items 4 + 6)

- `d97c23e` `da_fair_price_identity.py` (21 selftests; dual timestamps;
  strictly-as-of; statuses-not-zeros; boundary-admit positive controls)
  + `PHASE2B_CHALLENGER_PROTOCOL_DRAFT.md` (DRAFT-FOR-USER-FREEZE;
  increment-to-Identity estimand; closed family; declared pass criteria;
  five falsifiers).
- `612346b` second challenger NAMED pre-review (Binance USDM bookTicker mid)
  with era floor (`recv_ns >= 1787579334881534478`) and the
  closer-to-settlement interpretation pre-declared (both readings in R-244).
- `7815c2f` + `6b25b7e` `da_replay_parity_battery.py` (18 checks;
  `replay_traj_canon_v1`; arm-name excluded from canonical bytes, asserted;
  perturbation falsifier paired with digest reproduction; ULP-level
  no-tolerance asserted; NaN refuses; PYTHONHASHSEED-crossed).
  **Signed zero deliberately UNNORMALISED** (R-247 ruling) — if you judge it
  should normalise, that is a parity-contract change to rule explicitly,
  never a quiet loosening.

**ASK: review the 2B protocol draft BEFORE the user is asked to freeze it
(R-243 sequencing); findings on the new builds are batch findings.**

## Governing documents

- `ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md` @ `3b71d3e` (frozen,
  byte-untouched — verified twice) + `ITER011_PREREG_AMENDMENT_A1.md`
  (FROZEN, Option 1, `5734e88`).
- `DAY_BAR_V2_PREREGISTRATION.md` @ `dfa0977` + `368345b` + `9bcc208`.
- Freeze receipt v2 @ `68dca00` (clock anchor `b3f7f9f` = `1787897340`).
- `LANE2_FAIR_PRICE_SUCCESSOR_INTERFACE.md`,
  `LANE4_REPLAY_PARITY_STUB_BATTERY.md` @ `6fc96e2`; `SEAT_PROTOCOL.md`
  (rule 15 added at the batch tip); register R-238..R-248.

Coordinator/MEM register and sweep commits in the range are state, not code —
in scope only where a recorded claim contradicts an artifact.
