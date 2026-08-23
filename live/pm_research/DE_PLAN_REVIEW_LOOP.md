# DE_PLAN_REVIEW_LOOP — charter — **LOOP ENDED 2026-08-23, stop rule met**

Ten iterations, ~57 defects found and applied, independent auditor sign-off
on the closing review. Final state: `plans/DE_MODULE_PLAN.md` Rev 7 ·
`plans/DE_PLACEMENT_POLICY_PLAN.md` Rev 8 · `plans/EV_REPLAY_PLAN.md` Rev 7 ·
`CANCEL_POLICY_PROTOCOL.md` (§1 frozen+answered) · `ev_replay.py` ·
`warning_window.py`, committed through the closing commit. See the closing
entry at the bottom of the iteration log.

Self-paced review loop over the DE plane plans. Started 2026-08-23.

**Objects under review:**
- `plans/DE_MODULE_PLAN.md` — module structure (ActionSpace · Constraints ·
  DecisionScheme · Allocator · Actuator)
- `plans/DE_PLACEMENT_POLICY_PLAN.md` — DecisionScheme policy content
- since iteration 5 also: `plans/EV_REPLAY_PLAN.md` and
  `CANCEL_POLICY_PROTOCOL.md` (record coherence)
(current revisions per each file's status line — revision-free by convention)

**Goal:** the plan pair is (1) **complete** — covers every content the DE plane
needs, with an owner for each; (2) **modular** — integrates into the
`PM_ARCHITECTURE.md` type algebra without violating the rules (plane ordering,
R-COMPAT, R-SSOT, R-KNOW, halt wiring, ports); (3) **sound** — survives
adversarial concrete cases.

**Method (the split that has already stopped rules being re-cut):** independent
review agents probe with distinct lenses and return findings; the coordinator
verifies each finding against the files before acting, applies confirmed fixes,
and records the iteration here. Reviewers never edit; the coordinator never
grades their own homework as a reviewer.

**Finding classes:** `MUST-FIX` (a defect that would propagate into code or
contradicts a measured fact / architecture rule) · `SHOULD-FIX` (gap or
ambiguity that invites a future defect) · `NOTE` (recorded, no change).
Every finding needs a concrete failure case, not a preference.

**Binding constraints on fixes:**
- Plans stay **DESIGN** — no fix may record as settled what a measurement could
  settle (standing rule), invent a measured number, or add a PnL/capacity claim.
- `FLOW_MODEL_STATE.md` wins on facts; `PM_ARCHITECTURE.md` wins on types.
- Fixes must not create a `DE → BE` edge or an ownerless/two-owner quantity.

**Stop rule:** the loop ends when an iteration produces **zero confirmed
MUST-FIX findings twice in a row** (loop-until-dry, K=2), or the user stops it.
Verdict recorded per iteration: `DEFECTS_FOUND_AND_APPLIED` | `CLEAN` |
`BLOCKED(reason)`.

---

## Iteration log

### Iteration 1 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

Three independent reviewers (completeness / integration / adversarial-cases).
**12 MUST-FIX-class findings confirmed after coordinator verification against
`contracts.yaml` v22 and `PM_ARCHITECTURE.md`; applied as
`DE_PLACEMENT_POLICY_PLAN.md` Revision 3 and `DE_MODULE_PLAN.md` Revision 2.**
Three findings converged from two lenses independently (terminal-vs-resting-
quotes, illegal `TOLERATE_UNUSED`, wrong §6.2 contract list) — treated as one
each.

**Confirmed and applied (MUST-FIX):**
1. Terminal "no new risk at `r<60`" could not stop resting adding quotes from
   filling — a feasibility oracle only refuses new actions. Fix: ONE "new
   risk" predicate (increase in contingent `L_adv`, position + worst-case fill
   of resting quotes), scheme-owned retraction of the adding side at `r≈60`,
   reducing-only book below it.
2. Halted-inventory disposition undefined. Fix: HALTED blocks everything incl.
   reducing `CROSS` (untrusted state must not trade, even to reduce);
   `cancel_all` fires via the halt port; carry-to-resolution is the designed
   degradation bounded by `L_adv`.
3. Cap breach via burst fills between decisions had no defined response. Fix:
   REDUCING-ONLY state (shared with `r<60` and DEGRADED), breach is a
   `HealthEvent`; feasibility prices contingent exposure.
4. §8.1 warning-window thresholded at `τ` alone, omitting the 250 ms knowledge
   lag — larger than the whole measured burst timescale (btc half-life
   80.8 ms). Fix: branch statistic is drift share with `W > lag + τ`; the
   `τ=0` rung is the lag floor, not "zero latency"; §4.1's "first trade is a
   warning" re-scoped to multi-cluster runs and queue drain.
5. Dump could lift our own resting quote; self-trade prevention unowned. Fix:
   cancel-before-cross sequencing rule (§4.7), order preserved by Actuator;
   venue self-match handling added to unverified facts.
6. `NEW_BBO` treated as an unconditioned action at a 1-tick book where
   fronting is only executable at level re-formation. Fix: placement values
   `JOIN | FRONT_ON_FORMATION`; the composed policy explicitly quotes the
   measured `SKEW_LB` arm. (Name-vs-definition instance.)
7. `TOLERATE_UNUSED` is not a legal `UnavailableAction`
   (`Halt|RefuseAction|FallBack`), and `DecisionProblem.belief` has no
   `Unavailable` arm at all — `RulePolicy_v1` was unwireable while BE-Belief
   is a seam. Fix: consumed-inputs manifest licenses omission; belief-widening
   listed as a MIGRATION (not additive).
8. §6.2 contract list wrong in both directions: `CANCEL` already exists in
   `ActionSpace.verbs` (`QUOTE|CANCEL|MINT|MERGE|CROSS|WAIT`); the real gaps —
   `Action.order_ref`, `Action.placement` — were unlisted; plan verbs renamed
   to the contract's. (Second name-vs-definition instance, inside a plan
   warning about it.)
9. No DE module except the Actuator declares `telemetry_out` — scheme/
   constraints failure could never reach OP; fail-closed silently not
   applying. Fix: telemetry ports + named `HealthEvent` sources for all four
   acting modules.
10. Quote SIZE and total capital had no owner; every measured number is
    conditional on 5-share quotes. Fix: SP-Params → `CapitalBudget` (incl. max
    quote size) → scheme sizes ≤ budget; live size pinned to measured support
    until re-measured (gate note).
11. Live collector gap: venue keeps filling while we are blind and post-gap
    `SelfState` is silently wrong. Fix: Actuator-owned
    reconcile-before-quote on every recovery/restart/halt-reset; divergence is
    a `HealthEvent`; lifecycle section added (§5b).
12. MERGE had two writers and no channel. Fix: Allocator is sole issuer of
    `MINT/MERGE` via a typed `CapitalOpCommand` channel to the Actuator; the
    scheme never emits capital ops.

**Also applied (SHOULD-FIX/NOTE):** `DecisionSchemeConfig` reconciled to the
YAML (`coupling_mode: Dynamic` for a universe that resolves every 5 minutes;
`incentive_model`); `utility_none`/`incentive_none` registered-null pattern
instead of silent misdeclaration; T-AGE trigger REMOVED (waiting never worsens
queue position — its rationale was backwards); §8.1 redefined on a
parameter-free permissive envelope (the "policy-free" claim was leaking trigger
parameters); re-arm/re-post semantics pinned per trigger type + partial-fill
row type; band hysteresis is a POLICY grid dimension, Actuator debounce
identity in replay, flip/cancel counts a mandatory report axis; τ-rung
observation mechanism named (reconciliation upper bound → conservative rung,
stored in SP-Params); side-aware `N*` recorded beside the variance form;
SSOT handles for `r=60`/fees/`verdict_coins`; resolution-edge facts +
settlement booking + MINT-first conditionality; replay/live parity as a
promotion rule; composition-root duties named (per-decision coupling graph,
`horizon` = `r`, `spec_snapshot`); `Decision.duals` declared empty for a rule
policy; falsifier added (contingent-`L_adv` feasibility unworkably tight).

**Rejected/corrected reviewer claims (logged so they are not re-derived):**
- "`cancel_all` is bound to no trigger" — WRONG: the architecture's halt port
  routes both the constraint edge and the priority `cancel_all`. The real gaps
  were DE telemetry sources (finding 9) and reconciliation (finding 11).
- One reviewer read the harness `front=True` as a same-price queue-jump
  idealization "physically realizable only at formation events" — adopted, but
  its sharper claim that the prose "names the UB arm" was already half-true in
  Rev 2 §4.6; the fix (finding 6) supersedes rather than contradicts.

**Not applied:** nothing that would settle by measurement was recorded as
settled; no measured number invented; scope (btc/eth, mechanism-first, no PnL)
unchanged.

Next iteration: fresh re-review of Revision 3 + Revision 2 — verify the fixes
landed coherently, hunt residuals introduced by the rewrite.

### Iteration 2 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

One fresh-eyes verification reviewer over placement Rev 3 + module Rev 2.
**2 MUST-FIX + 4 SHOULD-FIX confirmed; applied as placement Revision 4 and
module Revision 3.** 10 of 12 iteration-1 fixes verified present and coherent;
every specific contract claim verified exact against v22; every quoted number
verified against the fact authority except the two below. Stop counter: 0
consecutive clean.

**Confirmed and applied:**
1. (MUST-FIX) Rev 2's capital/size chain was unwireable — v22 `CapitalBudget`
   has no consumer, no max-quote-size field, no path into a scheme whose
   manifest deliberately cannot grow. Fix: route the budget through
   `DE-Constraints` (one ADDITIVE module-record change) and out via
   `FeasibleSet.max_size`, a field v22 already has; manifest unchanged. The
   reviewer correctly named this a reintroduction of the iteration-1
   missing-from-the-list defect class.
2. (MUST-FIX) The §2 constraint table encoded `HALTED ⇒ no new risk
   (predicate)` — under which a reducing `CROSS` is feasible — while three
   other statements said HALTED blocks everything; and the block-everything
   rule had no named enforcer. Fix: `HALTED ⇒ FeasibleSet = ∅` in the table,
   plus the Actuator as second enforcer (refuses all venue writes except
   `cancel_all` while `halt_in = HALTED`).
3. (SHOULD-FIX) The 1-tick "99.9 % of the time" swapped its denominator (that
   figure is conditional on the tails-only 0.001 regime). Fixed to the modal
   population (median 1 tick, p90 2 ticks). Seventh logged instance of the
   denominator/population defect class.
4. (SHOULD-FIX) "4.5 fills destroyed per dump" back-derived from the naked
   half-spread, contradicting the measured capture quoted forty lines earlier.
   Fixed to 3–3.7 on measured capture.
5. (SHOULD-FIX) REDUCING-ONLY's permitted-action set differed across three
   statements and had no breach scope. Fixed: one canonical definition
   (reducing quotes sized ≤ |net|, reducing CROSS, capital ops, CANCEL, WAIT)
   + scope rule (per-market breach and r<60 → that window; scenario breach and
   DEGRADED → global).
6. (SHOULD-FIX) `FLOW_MODEL_STATE.md` §1e's closing "Two days" is stale
   against the `edge_l1_v1` receipt (`source_days` n=4). NOT edited by DE —
   the state page is not this plane's to correct; surfaced in
   `COORDINATION.md` D-4 per the page's own "say so rather than reconciling
   privately" rule. Plans carry the receipt figure with a provenance note.
7. (NOTEs) §3.5 telemetry premise softened (explicit records vs wildcard
   default); 75–350 → 75–352 ms; §3.1's opening clause contradicted its own
   manifest (rewritten); reducing-quote size ≤ |net| qualification added at
   r<60 (an oversized reducing quote is itself contingent new risk).

**Process note:** the roles changed mid-iteration (four-plane split,
COORDINATION.md §0); this loop and the plans are DE-owned, the three
iteration-1 design calls remain coordinator-gated PROPOSALS (COORDINATION.md
§2.6), and the `CANCEL_POLICY_PROTOCOL` freeze including the §8.1 branch
threshold is explicitly the coordinator's (§2.1).

Next iteration: fresh re-review of Revision 4 + Revision 3. A clean result
starts the two-clean-in-a-row stop counter.

### Iteration 3 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

Fresh-eyes reviewer over placement Rev 4 + module Rev 3. **3 MUST-FIX + 2
SHOULD-FIX + 4 NOTEs confirmed; applied as placement Revision 5 and module
Revision 4.** All six iteration-2 fixes verified present; two of the new
mechanisms failed under logic attack. Stop counter: 0 consecutive clean.

**Confirmed and applied:**
1. (MUST-FIX) **The ≤|net| enforcement claim was FALSE under the plan's own
   predicate.** Counterexample: net +10 Up at ~0.50 (`L_adv` ≈ $5.00), reducing
   ASK sized 18 → worst-case flip to 8 Down ≈ $4.00 < $5.00 → contingent
   `L_adv` DECREASES → feasible under the predicate, yet flips past flat. The
   cap only follows from the predicate when the flipped side is more
   expensive. Fix: ≤|net| is an explicit HARD rule of REDUCING-ONLY, owned by
   the oracle's state branch; "the oracle will refuse it" withdrawn in both
   plans.
2. (MUST-FIX) **`FeasibleSet` semantics were unpinned and the natural reading
   inverts the halt door**: `FeasibleSet{max_size, binding}` has no action
   list; missing-key-=-uncapped makes an EMPTY set fully PERMISSIVE, and
   DEGRADED/REDUCING-ONLY had no second enforcer. Also verb-only keys cannot
   express REDUCING-ONLY (both sides are `QUOTE`). Fix: new module §2a —
   side-keyed key domain (`"<verb>:<side>"`), default-DENY (missing key = 0),
   one-`DecisionProblem`-per-`(coin, window)` stated as a rule (§3.3), §6.2
   notes row added.
3. (MUST-FIX, overtaken-by-events) **The Layer-1 population fact was wrong in
   a new direction**: `FLOW_MODEL_STATE.md` §1f (BE, measured) — the shared
   sampler is earliest-first, so `edge_l1_v1` spans ONE UTC day (2026-08-20);
   the receipt's `n_days: 4` counted days READ. Iteration 2's provenance note
   bet on the receipt; both sides of that conflict were wrong. Fix: population
   corrected in the plan and the protocol draft (blind), within-day-interval
   caveat added, re-sample question routed to the coordinator's freeze
   (R-ADMISS).
4. (SHOULD-FIX) The strengthened `HALTED ⇒ ∅` row cited architecture §1, whose
   halt-edge label states the WEAKER predicate rule — a future reconciliation
   citing it would revert the fix. Recorded as a divergence row in §6.2 for
   coordinator reconciliation.
5. (SHOULD-FIX) Stale revision labels (policy H1 said Rev 3 while its status
   said Rev 4; module cross-ref pinned Rev 3). Fix: title and cross-references
   are now revision-free; the status line is the only revision carrier.
6. (NOTEs) Dump arithmetic low end 2.89 → range 2.9–3.7; module §1.2 aligned
   to the modal-population phrasing; capital-ops "always permitted" scoped to
   REDUCING-ONLY with the Actuator named as their only HALTED door
   (`CapitalOpCommand` bypasses the oracle; DE-Allocator has no `halt_in`);
   §3.4's licensed-omission list completed with `coupling` + the
   can-be-Unavailable validation rule stated precisely.

**Also verified clean by the reviewer:** the CapitalBudget routing is genuinely
additive; ports exactly as §3.5 states; all quoted numbers against the
authorities; iteration-1 fixes intact; HALTED/REDUCING-ONLY textually
consistent across both plans.

Next iteration: fresh re-review of Revision 5 + Revision 4, focused on the §2a
pin and the corrected population text. Stop rule unchanged.

### Iteration 4 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

Fresh-eyes reviewer over placement Rev 5 + module Rev 4 + the protocol draft.
Reported 1 MUST-FIX + 4 SHOULD-FIX + 4 NOTEs. **The MUST-FIX was a
READ-TIMING ARTIFACT, resolved not applied:** the reviewer snapshotted
`CANCEL_POLICY_PROTOCOL.md` seconds before Ruling R-1 was applied to it
(protocol mtime 02:53:01 vs its evidence; the R-1 edits landed minutes later,
BEFORE the `ww_v1` receipt was read). The current file carries the frozen
three-way rule, the §1.3 statement, the §2.3 amendment and the verification
note; the receipt was read against the THREE-WAY rule; the blind chain is
intact. Logged so the mtime evidence is explained, not just contradicted.

**Confirmed and applied (as placement Revision 6 + module Revision 5):**
1. (SHOULD-FIX) The ≤|net| HARD cap covered reducing quotes but left reducing
   `CROSS` unqualified in the canonical prose — an oversized reducing CROSS
   dumps past flat as a TAKER through the same predicate hole. Cap now covers
   both verbs, unconditionally, in module §2 and policy §7.
2. (SHOULD-FIX, and mathematically the sharpest catch of the loop) **The
   iteration-3 rationale was FALSE:** "the cap only fails to follow when the
   flipped side is cheaper" is wrong — for ANY reducing size `s` with
   `|net| < s < 2|net|`, the flipped magnitude `|net−s| < |net|`, so
   contingent `L_adv` falls at ANY price pair; the plans' own counterexample
   sits at ~0.50 where the sides are equal-priced. The HARD rule was already
   unconditional (correct); the rationale invited a future "relax when
   pricier" argument that would reopen the hole. Corrected in all three
   documents; this charter entry is the iteration-3 correction of record.
3. (SHOULD-FIX) "Refuse ALL actions" / "dies at either door" contradicted
   §2a's CANCEL/WAIT carve-out at the letter. Re-scoped to venue-write /
   size-bearing verbs; CANCEL-only decisions pass the oracle and are
   harmlessly absorbed (book already retracted).
4. (SHOULD-FIX) Two stale revision pins (module §8 → "Revision 3"; protocol
   header → "Revision 4") — both now revision-free.
5. (NOTEs) `MINT`/`MERGE` dropped from the §2a pin scope (side-less verbs
   cannot appear in a `"<verb>:<side>"` map; capital ops route around the
   oracle entirely); §8.1's E-FLOW envelope aligned to the frozen "at our
   level or better"; the `COORDINATION.md` path added to both plan headers;
   R-2's literal-execution risk (reintroducing n=4) was already mooted by
   BE's §1f landing first — noted, no action.

**Also this iteration (outside the loop's own scope, recorded for
completeness):** §8.1 was RUN under Ruling R-1 and ANSWERED — DEAD on both
verdict coins, doubly; §9.1 FIRED; the §8.2 grid is not built. The plans and
protocol now carry the outcome as a measured result citing the receipt.

Stop counter: 0 consecutive clean (4 substantive SHOULD-FIX this round;
trend 12 → 2 → 3 → 0 MUST-FIX after staleness resolution). Next iteration:
fresh re-review of Revision 6 + Revision 5 + the answered protocol.

### Iteration 5 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

Fresh-eyes reviewer over placement Rev 6 + module Rev 5 + the answered
protocol + (first review) `EV_REPLAY_PLAN.md`, with SP-consistency checks.
**1 MUST-FIX + 3 SHOULD-FIX + 4 NOTEs confirmed; applied as placement
Revision 7, module Revision 6, EV-Replay plan Revision 2, plus corrections to
`EDGE_LAYER1_RESULTS.md` and the frozen protocol's record text.**

1. (MUST-FIX) **"Six of eight cells negative" is FIVE of eight** — the
   receipt's `signs` block shows three spanners (btc h=60, eth h=15, eth
   h=30); the origin (`EDGE_LAYER1_RESULTS.md`) even listed all three while
   saying "the two". Survived FOUR iterations. Verified against the receipt
   before applying; corrected at the origin with a dated note and in the
   placement plan; the `FLOW_MODEL_STATE.md` §1e propagation routed to the
   page owner via D-4 (R-2 pattern). No verdict moves — f* and DEAD/DEAD
   rest on h=5 only.
2. (SHOULD-FIX) **Iteration 4's "generic" rationale was ITSELF false** —
   `L_adv` is dollar cost basis, not share magnitude, so a flip past flat can
   RAISE contingent `L_adv` when the flipped side is expensive (counterexample:
   basis 0.10, flip to ≈0.90 → $7.20 > $1.00, predicate over-refuses). Two
   successive false derivations of the same rule. Landing: the ≤|net| cap is
   DEFINITIONAL to REDUCING-ONLY (never past flat, maker or taker) and the
   predicate is recorded as unable to substitute in EITHER direction; no
   derivation from `L_adv` arithmetic is attempted again.
3. (SHOULD-FIX) The protocol header's "granted BEFORE the measurement ran"
   overstated the sequence — the truth (drafted before any measurement
   existed; run started under §3.1's license; frozen before the receipt was
   READ at 175/210 in flight) is stronger-documented and now stated
   precisely. R-1's own opening phrase carries the same slip — the
   coordinator's text, flagged in D-4, not edited.
4. (SHOULD-FIX) EV-Replay plan cited R-IMPUTE for the fit-cutoff guard; the
   governing rules are R-WFWD/R-REQ, with R-IMPUTE the *other* look-ahead
   (t_known laundering) the env also refuses — two classes, two refusals,
   both to be selftested.
5. (NOTEs) §2 table "venue-write" → "size-bearing"; §8.1's stale "T-FLOW
   envelope / at our level" aligned to frozen `E-FLOW … or better`;
   `EventTimeView` license narrowed back to canary-only; the per-window
   purity discharge given its lapse boundary (capital-coupled replays);
   §6.4 dead-lettered (B7 returned DEAD-DEAD); receipt-key mapping
   `join = BACK_DISPLAYED` pinned in protocol §1.3; garbled ledger path
   fixed; charter header made revision-free.

Stop counter: 0 consecutive clean. MUST-FIX trend 12 → 2 → 3 → 0 → 1 (the 1
being a four-iteration-old fact error, found by the loop working as
designed). Next: iteration 6 over placement Rev 7 + module Rev 6 + EV plan
Rev 2.

### Iteration 6 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

Fresh-eyes reviewer over placement Rev 7 + module Rev 6 + EV plan Rev 2 + the
protocol + (first code review) `ev_replay.py` + the corrected results file.
**2 MUST-FIX + 3 SHOULD-FIX + 4 NOTEs; applied as EV plan Revision 3,
`ev_replay.py` hardened (13 → 22 selftest checks), `EDGE_LAYER1_RESULTS.md`
population corrections, and small protocol/policy-plan wording fixes.**

1. (MUST-FIX) **The EV plan header claimed "gated green against §4" while
   §4.3's must-fail controls did not exist** — and v1's parity gate
   structurally cannot fail (two invocations of the same reference engine;
   honest in the code and the ledger, overstated in the plan). Fix: §4 is now
   a status TABLE with claims matching artifacts exactly; §4.3 is named OPEN
   ACCEPTANCE DEBT that BLOCKS any engine change and any B2 event-loop
   change; §2's plugin-path rows (artifact resolution, RNG, τ-as-parameter)
   are marked NOT IN v1 rather than described as present.
2. (MUST-FIX) **`EDGE_LAYER1_RESULTS.md` still asserted the four-day
   population in three places** — in the very file iteration 5 had just
   corrected for the sign count, where a fresh dated note makes the rest read
   as vetted. All three corrected to the §1f fact (one UTC day; the stamp
   counts days read); the void "three-vs-four days" comparability note
   corrected too.
3. (SHOULD-FIX, code) `run_hash` did not cover mid paths or gap-interval
   endpoints — two runs differing only there hashed identically, blinding the
   determinism gate to exactly the inputs `evaluate_markout` consumes. Fix:
   per-record content hashes over the FULL record + an `engine_hash` from
   source (replacing a hand-written label that would not change on an engine
   edit) + three hash-sensitivity must-fail controls in selftest.
4. (SHOULD-FIX, code) The boundary selftest scanned `ReplayEnv` methods but
   only `RunRecord` FIELDS — a markout method on the record type passed. Fix:
   both class namespaces scanned, methods included; the plan's §0.1
   `env.close()` phantom removed; §4.4 status stated as partial (import-level
   separation arrives with the plugin-path module split).
5. (SHOULD-FIX) §2's artifact-resolution row claimed "both selftested" for
   refusals v1 cannot express. Re-scoped to plugin-path contract with the
   THEN explicit.
6. (NOTEs) Duplicate envelope statement in policy §8.1 collapsed to one; the
   protocol's "~5× front-of-queue ratio" renamed fill-count ratio and named
   apart from the 9.4× inventory ratio; R-IMPUTE mechanism phrase precision;
   `FLOW_MODEL_STATE.md` §1e's propagated "Six of eight" re-confirmed as
   routed (BE's page).

Stop counter: 0 consecutive clean. MUST-FIX trend 12 → 2 → 3 → 0 → 1 → 2 —
the last two rounds' MUST-FIXes are claim-vs-artifact and stale-fact classes
in NEWLY ADDED surface (code + a results file pulled into scope), not
regressions in the core plans, which have been stable since iteration 4.
Next: iteration 7 over EV plan Rev 3 + the hardened code + spot re-checks.

### Iteration 7 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

Fresh-eyes reviewer over EV plan Rev 3 + hardened `ev_replay.py` + the
corrected results file + spot-checks. **2 MUST-FIX + 3 SHOULD-FIX + 4 NOTEs;
applied as EV plan Revision 4, module Revision 7, placement Revision 8,
`ev_replay.py` re-hardened, smoke re-run, and — the structural fix — the DE
corpus COMMITTED, per-iteration from now on.**

1. (MUST-FIX) §4.2's "PASS" cited the pre-hardening smoke — the
   extended-coverage determinism gate had never run on real windows (the
   claim-stronger-than-artifact class recurring INSIDE the fix for it). Fix:
   smoke re-run post-hardening — parity 14/14 both arms, determinism PASS
   under the full-coverage hash, receipt regenerated with
   `engine_hash`/`record_hash`/era/bound stamps.
2. (MUST-FIX) **Every "Revisions N–1 in git history" claim was FALSE — the
   loop's files were untracked, zero commits; the audit trail existed only in
   the working tree.** Fix: headers reworded to the truth (prior revisions
   NOT preserved), and the corpus is committed with per-iteration commits a
   standing loop practice from iteration 7 onward. The unrecoverable priors
   are the cost of six iterations of not noticing; recorded, not excused.
3. (SHOULD-FIX) `engine_hash` hashed only `replay_window`'s own body while
   the load-bearing logic lives in `RestingSide`/`BookState`/`fold_*` — an
   edit to the queue-drain rule would change fills with the hash unmoved,
   leaving §4.3's BLOCK without a tripwire for the likeliest change. Fix:
   transitive-source hash (six functions/classes + the four constants).
4. (SHOULD-FIX) The boundary scan was blind to annotation-only dataclass
   fields — `markout_h5: list` on `RunRecord` passed all checks (fields
   without defaults never appear in `vars(cls)`). Fix: scan
   `__dataclass_fields__` of both classes too.
5. (SHOULD-FIX) The receipt promised but did not stamp collector era, and
   the queue bound was inferable-not-stamped (violating the plan's own §3.4).
   Fix: `collector_era: fi.ERA` + explicit per-record `queue_bound`.
6. (NOTEs) Placement's unbumped iteration-6 edit absorbed into Rev 8; the
   "13" prior selftest count is unverifiable (a consequence of finding 2 —
   no prior version exists); `record_hash` survived the collision attack
   (float repr injective, str-keyed diagnostics, no bloat);
   `FLOW_MODEL_STATE.md` §1e's "Six of eight" remains live on the fact
   authority — routed to BE, re-flagged.

Stop counter: 0. Trend 12 → 2 → 3 → 0 → 1 → 2 → 2. Standing practice added:
**commit per iteration; a revision claim about history must be checkable in
history.** Next: iteration 8 over the committed corpus.

### Iteration 8 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED`

**1 MUST-FIX + 2 SHOULD-FIX + 4 NOTEs; applied as EV plan Revision 5,
`ev_replay.py` re-hardened, this correction of record, and the stop-counter
pin below.**

1. (MUST-FIX — **correction of record for iteration 7's finding 2**) The
   blanket "the loop's files were untracked, zero commits" was FALSE for two
   of the eight files: `plans/DE_PLACEMENT_POLICY_PLAN.md` was tracked since
   `f46379f` (2026-08-22, the pre-revision original) and
   `EDGE_LAYER1_RESULTS.md` since `a460ccf` — my own Rev-8 placement header
   stated this truth while this charter contradicted it one entry earlier.
   Consequence an auditor should know: **the cumulative Rev 1→8 placement
   diff and the five-of-eight corrections ARE recoverable**
   (`git diff f46379f c0bae24 -- plans/DE_PLACEMENT_POLICY_PLAN.md`;
   `a460ccf..c0bae24`); only the six genuinely-untracked files' intermediate
   revisions are lost. The `c0bae24` commit message carries the overstated
   claim immutably — this entry is its annotation. The standing practice
   iteration 7 added banned exactly this defect; iteration 7 committed it in
   the same breath. Propagations fixed: EV plan header, D-4 (report #13).
2. (SHOULD-FIX) `engine_hash`'s closure was incomplete: `el.HORIZONS` (sets
   tick-window `unavailable_iv` spans), the four line-filter marks (drop
   whole event classes), and `fi._gz_lines` (the tape decoder) could each
   change records with the hash unmoved. Closure completed; the hash moved
   (`190fc906…` → `c5c405bc…`), which is itself the demonstration.
3. (SHOULD-FIX) **Stop-counter semantics PINNED, prospectively:** the counter
   counts **zero-confirmed-MUST-FIX iterations** — the header rule as
   written. Iteration 4 (0 MUST-FIX after staleness resolution, SHOULD-FIXes
   applied) would have counted under this pin; the logs said "0 consecutive
   clean" instead. The pin changes nothing retroactively — iterations 5–8
   each carried ≥1 MUST-FIX, so today's streak is 0 under either reading —
   and it is being pinned while it decides nothing, which is the only
   legitimate time (the R-6/Class-D lesson, applied to our own rule).
4. (NOTEs applied) Gate outcomes now persist in the receipt (`gates` block —
   they were stdout-only, PASS checkable only by entailment); §2's spec-hash
   row corrected (receipt-level stamps, protocol NAME); §4 vintage labels;
   the queue-bound comment reworded (consumer never infers; the v1-internal
   map becomes a run parameter when both bounds share a run).

Stop counter: 0. Trend 12 → 2 → 3 → 0 → 1 → 2 → 2 → 1. Next: iteration 9
over the re-committed corpus; two consecutive zero-MUST-FIX iterations end
the loop.

### Iteration 9 — 2026-08-23 — verdict: `ZERO CONFIRMED MUST-FIX` — **streak 1 of 2**

2 SHOULD-FIX + 4 NOTEs, all applied (EV plan Revision 6, `ev_replay.py`
re-hardened, this entry). First streak increment under the pinned rule.

1. (SHOULD-FIX) `engine_hash` residue after "closure completed": the record
   SHAPES (`el.Fill`, `el.WindowFills` — a field reorder transposes every
   tuple positionally with the hash unmoved), the env's own record mapping,
   and (N3, preempted) the parity comparator. All added. **The wording
   "closure completed" is retired**: closure is a property review earns
   per-iteration, not a state — third residue round in the same class says
   the claim form itself was the defect.
2. (SHOULD-FIX — correction of the iteration-8 correction's scope clause)
   "Only the six genuinely-untracked files' intermediate revisions are lost"
   was over-scoped: the two TRACKED files' intermediate revisions (placement
   Rev 1–7, results-file intermediates) are equally lost — git holds two
   snapshots of each; what is recoverable is the cumulative endpoint diffs,
   as the previous sentence stated correctly. Corrected here, additively —
   past log entries are never edited in place.
3. (N1) The iteration-8 recovery command needs the `live/pm_research/`
   pathspec prefix — from the repo root it silently returns an empty diff:
   `git diff f46379f c0bae24 -- live/pm_research/plans/DE_PLACEMENT_POLICY_PLAN.md`.
4. (N2, applied) Per-window `inputs_hash` (gaps + token ids) now stamped in
   receipts — those inputs shape `RunRecord`s and previously arrived
   unstamped, so an input-side change shifted `run_hash` unattributably.
5. (N4, recorded) The reviewer independently recomputed both hash vintages
   bit-exact and verified the engine modules unchanged since `c0bae24` —
   the other-plane working-tree edits touch nothing in the closure.

Stop counter: **1**. Trend 12 → 2 → 3 → 0 → 1 → 2 → 2 → 1 → 0. Iteration 10
clean of confirmed MUST-FIX ends the loop.

### Iteration 10 — 2026-08-23 — verdict: `DEFECTS_FOUND_AND_APPLIED` (SHOULD-FIX only) — **ZERO confirmed MUST-FIX → streak 2 of 2 → LOOP ENDS**

*(Vocabulary note, amended prospectively per this iteration's N3: the
iteration-9 entry's verdict string `ZERO CONFIRMED MUST-FIX` was outside the
charter's declared vocabulary; from this entry, verdicts use the declared set
and the MUST-FIX count is stated separately — which is what the iteration-8
pin already decoupled.)*

The closing reviewer applied maximum skepticism to the recurring
claim-vs-artifact class, recomputed every recomputable bit-exact (both
engine-hash vintages, all 14 `inputs_hash` values, the gates block, the
selftest count), and **signed off as an independent auditor: internally
consistent, claims-match-artifacts, coordinator-rule-compliant.**

Applied in closing:
1. (SHOULD-FIX) §4.2 still used the retired claim form "hash closure
   completed" in the same file that retired it — one citation away from
   reverting the retirement, the exact weaker-citation risk iteration 3
   flagged for the architecture's halt label. Reworded; cell vintage brought
   current (the PASS is backed by the iteration-9 receipt).
2. (NOTEs) §2's spec-hash row extended with the iteration-9 `inputs_hash`
   stamp (under-claim); charter N4's first clause scoped ("engine modules
   unchanged" → "the closure untouched by the other-plane edits", which was
   the verified statement); the evaluation pass (`evaluate_markout`,
   `el.horizon_rows`) recorded as OUTSIDE every hash — consistent with the
   engine-hash's honest scope, no claim broken today, and the first item for
   any future re-opening of this loop.

**Loop end state.** MUST-FIX trend 12 → 2 → 3 → 0 → 1 → 2 → 2 → 1 → 0 → 0.
What the ten iterations bought, in one paragraph: the DE plane went from two
unreviewed design documents to a corpus where the terminal rule is
enforceable, the halt semantics have two named doors, the capital/size chain
wires through existing contract types, REDUCING-ONLY has one definition with
a definitional cap two false derivations could not shake, a frozen falsifier
was drafted blind / frozen before its answer / and killed the cancellation
family honestly, the harness's acceptance gates state exactly what they test,
every number traces to a receipt, and the record of the record survived four
corrections of its own corrections. **Re-open trigger:** any structural
change to the DE plans, the first non-reference engine in EV-Replay, or the
coordinator's 680-window re-sample ruling reopening the cancel family.
*(Post-close addendum, 2026-08-23: the third trigger is FORECLOSED — R-9's
day series ran instead of the re-sample, DEAD generalised 8/8, and R-11
CLOSED the family. New DE code surfaces (B2 onward) receive the same
fresh-eyes review treatment as dispatch QA, outside this closed loop.)*

*(Second post-close addendum, 2026-08-23, R-44 method note — a hole in THIS
loop's method, recorded where the method lives: ten iterations audited WHAT
EXISTS and never asked WHAT WAS NEVER ATTEMPTED. The capture-vs-adverse-
selection trade-off had three unopened levers — placement depth, quote size,
selective quoting — invisible to every completeness lens here because every
lens walked the corpus, and untested work leaves nothing in the corpus to
walk. Found by the USER asking whether the trade-off had been optimised; not
by this loop, not by the coordinator. Standing correction for any future
review loop: alongside the corpus lenses, one lens per iteration must
enumerate the DECISION VARIABLES the plan exposes and ask, for each, where it
was ever varied — a lever whose every appearance is a single pinned value is
a finding, whatever the documents say. The 5-share pin sat in plain sight in
every receipt for the entire loop.)*
