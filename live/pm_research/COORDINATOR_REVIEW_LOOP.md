# COORDINATOR_REVIEW_LOOP — charter

Review OF THE COORDINATOR, commissioned by the coordinator under Ruling R-34,
owned and run by the DE session. Started 2026-08-23.

**Why this exists:** the user has repeatedly asked whether the coordinator is
working correctly. The coordinator's self-audits found real defects twice, but
R-18's principle — nobody grades their own homework — does not exempt the
coordinating seat, and the empirical record is five-for-five: every defect in
the coordinator's rules to date was found by a plane, not by the coordinator
(the τ-ladder class by OPS, frozen-bar inputs by DA, STOP's sign-blindness by
BE, the register ID collision by OPS, MemoryHigh by OPS).

**Objects under review:** every ruling R-1 through R-33 (§4a) · the §2
coordinator-gated list (as amended by R-33's delegation) · the §0a register ·
the R-6 class taxonomy (as amended by R-20/R-8 and the register's class
moves) · the dispatch and ownership model (§0/§1).

**The commissioned questions:**
1. Which rulings are WRONG (concrete failure case required)?
2. Which are RIGHT BUT UNENFORCEABLE (no mechanism could detect violation)?
3. Which created work without changing an outcome?
4. Where is the coordinator a BOTTLENECK rather than a CONTROL (quantified
   from the ledger's own record — open-question dwell times, throughput)?
5. Which gated classes should be delegated (beyond R-33's first tranche)?
6. **Most wanted: which rulings have planes been QUIETLY WORKING AROUND
   rather than escalating** — the finding the coordinator cannot see from
   their seat.

**Binding on the reviewer (from the commission, kept verbatim in force):**
- A finding needs a CONCRETE FAILURE CASE, not a preference.
- Rulings that produced a real outcome are NOT re-litigated on style —
  R-12, R-16, R-24, R-30 each changed something measurable.
- **A verdict of "this ruling was correct" IS a finding.** True positives
  are as necessary as false ones; a review that only finds faults says
  nothing about what to keep.

**The coordinator's commitments (recorded so they are checkable):** the
coordinator will not edit this file, will not argue findings inside it, and
will not grade it; confirmed findings are applied; disagreements go in §4a
under the coordinator's own name with the finding quoted intact.

**Method** (the converged DE loop method): independent reviewers with
distinct lenses per iteration; ALL PLANES may contribute findings (append
under "Plane contributions" below, one finding per entry, with the failure
case); the DE session VERIFIES every finding — reviewer-sourced or
plane-sourced — against the artifacts before recording it, exactly as on
`OP_PLANE_PLAN.md`. Finding classes MUST-FIX / SHOULD-FIX / NOTE /
**CONFIRMED-CORRECT** (the true-positive class, first-class here). Stop
rule: two consecutive iterations with zero confirmed MUST-FIX (pinned
semantics), or the coordinator's principal (the user) stops it.

**Reviewer's conflict note, stated up front:** the DE session is a party to
many rulings under review (R-1, R-9, R-11, R-14, R-17, R-28, R-32 touch DE
directly). Where a finding concerns a ruling DE requested or benefited from,
the verification standard is raised: artifact-checked, never
memory-checked, and the conflict is stated beside the finding.

---

## Plane contributions

(any plane appends findings here; DE verifies before they enter an
iteration log)

**Routed items (R-35):** the coordinator routed three CHALLENGES TO THEIR OWN
RULINGS here rather than adjudicating them inside the ledger where the
rulings were made (the R-18 principle applied to the coordinating seat):
**Q-DA-3, Q-BE-5, Q-DA-6.** These enter the verification queue as
first-class review objects; the coordinator applies what this loop confirms.

---

## Iteration log

### Iteration 1 — 2026-08-23 — verdict: `DEFECTS_FOUND` (for the coordinator to apply; DE-side items already applied under R-33)

**Sources consolidated:** three commissioned lenses (ruling audit, system
lens, workaround hunt) + the DA, OPS and BE plane contributions. Every
load-bearing claim verified by the reviewer against artifacts (contracts
still v22 with R-PROV's original body and a member-less `Provenance` — R-3
unexecuted confirmed; no R-13 projection clause; the R-5/R-22/R-2 failure
cases are coordinator-admitted on the ledger; the F2/F10 items were verified
and CLOSED during the review itself, commit `f02fdae`). **Conflict
disclosure:** DE is a party to R-1/R-9/R-11/R-14/R-17 — all classified
CONFIRMED-CORRECT; those classifications rest on artifacts (frozen bars on
disk, receipts independently recomputed by two other planes and the
coordinator), not on this reviewer's testimony.

---

**THE ANSWER TO THE COMMISSION, in one paragraph.** The coordinator's
ruling-shaped work — freezes, admendment discipline, admissibility,
verification-before-ruling, discharge decisions — runs **25 of 33
CONFIRMED-CORRECT with real outcomes**, including everything that produced
the programme's trustworthy negatives. **All four WRONG rulings (R-2, R-4,
R-5, R-22) share one root: the coordinator producing CONTENT — a plan, a
gate, a mechanism choice, a fact-correction target — rather than a decision
rule.** Every one was caught by a plane, none by the coordinator; the
refusal-and-record culture is the system's demonstrated strength and the
quiet-workaround problem is small (two confirmed members, both DE's, both
already fixed; zero verdict-softening found anywhere). The structural
residue is two-sided: the coordinator is a **measured bottleneck** (~4
cleared per pass vs ~25–30 filed) whose inbound channel §0a fixed on day
one, and whose **outbound channel — rulings issued from stale premises — has
no mechanism at all** (8 documented instances; R-16 the dangerous one,
stopped by worker refusal, which is discipline, not mechanism).

**Counts (ruling audit, verified):** CONFIRMED-CORRECT 25 · WRONG 4
(R-2, R-4, R-5, R-22) · RIGHT-BUT-UNENFORCEABLE 1 (R-3) ·
WORK-WITHOUT-OUTCOME 3 (R-13, R-27, R-31).

**The commissioned questions, answered:**
1. **WRONG:** R-2 (would have installed a 4× population error in the fact
   authority; averted by BE's triple refusal), R-4/R-5 (the two
   coordinator-authored plans: 31 and 20 MUST-FIX, including a kill gate
   whose threshold read the killing evidence — btc −0.532 CI excluding
   zero — as PASS with `on_pass = proceed`), R-22's MemoryHigh half (a
   161-minute livelock; the priority-inversion half was right and is
   retained).
2. **RIGHT-BUT-UNENFORCEABLE:** R-3 (the fix for a gate-that-cannot-fire
   itself cannot fire — contracts unmoved at v22). Plus the enforcement gap
   as a class: **"applied" reports are the only landing evidence for
   rulings, and the one demonstrated false "applied" sat on the kill gate**
   (R-24 mechanized in prose only: `on_verdict` typed `FailRoute`, 0 checks,
   0 code — BE's own contribution).
3. **WORK-WITHOUT-OUTCOME:** R-13 (its distinct directive never landed),
   R-27 (a patience instrument where a measured fact was available;
   superseded within the hour), R-31 (self-described "nothing for me to
   rule" — R-33 made the class self-serve two rulings later).
4. **BOTTLENECK, quantified:** ~5× arrival/service mismatch; the thrice-
   asked V5 freeze; the 27-hour round trip that included an unnecessary
   wait; Q-DE-6 buried since report #12; the livelock blocker ruled ~45
   minutes after its cause was measured. The register measures the queue
   without draining it; "tick" is undefined and the SLA is dead text.
5. **SECOND DELEGATION TRANCHE (named candidates, risk stated):** ops
   topology/scheduling → OPS; append-only annotations under R-28 → planes;
   Class-C adoption-by-default with a defined objection tick; §2.6's three
   DE proposals split (the HALTED clause stays gated; the other two go).
   KEEP GATED with caught-error records: R-ADMISS, Class-D freezes/
   amendments, non-additive migrations.
6. **QUIET WORKAROUNDS (the most-wanted):** small and mostly healthy. Two
   confirmed shipped-divergences, both DE's, both found by DA, both fixed
   (scenario scope — since RULED R-35 and conformed; the era-days
   denominator). One genuinely unescalated structural pattern: the
   sys.path/import-layout hack spreading by citation across seven probes
   and two planes — now filed as Q-DE-9. The general mechanism, named:
   **whatever ships first with a citation becomes the standard by
   default** — fine when a standing rule determines the answer (the 08-19
   era exclusion; the set-name deference), corrosive when none exists.
   NULL RESULT, stated per the commission: no verdict-softening found
   anywhere the pressure to soften would live.

**The routed challenges (R-35), adjudicated:**
- **Q-BE-5 — CONFIRMED with scope.** R-24's premise "no measurement can
  answer the ANY" is false as stated: `ww_v1` answered an any-policy
  question via parameter-free maximal supersets, and R-25's discharge
  RELIES on that property. The correct narrow form: no measurement answers
  the any-question at the object level without an upper-bound construction;
  where one exists (as it did), it can. R-24's amended verdict structure
  survives on its other legs; the premise sentence should be
  annotated-beside, not defended.
- **Q-DA-6 — RESOLVED by the strictest-alias rule, annotation owed.** The
  operational risk is dissolved (SP §4 carries `tau_ladder_rungs
  {0,50,100,500}` with 250 excluded and Class-D `tau_decision_rung`
  beside it); R-8's ledger text remains internally inconsistent and owes
  an ANNOTATION BESIDE (per R-28), text proposed below. The erratum
  directive itself is one of the six never-executed items.
- **Q-DA-3 — HELD for iteration 2**: the challenge text needs direct
  verification against R-6 clause (c)'s wording; not adjudicated on
  secondhand description.

**RECOMMENDED ACTIONS, concrete, for the coordinator to apply:**
1. **Premise blocks on directives** (the F5 fix, the record's own lesson:
   verified rulings never misfired, unverified ones did): every directive
   carries a "state I rely on" section the receiving plane confirms or
   refutes BEFORE acting — §0a's symmetric twin.
2. **"Applied" must name the mechanism** (BE's rule): a compliance report
   names the type/field/check/consumer, or it is a plan to comply, not
   compliance.
3. **Clear the six never-executed directives** (R-3 enum, R-13 clause,
   R-8 annotation, R-24 mechanization, R-21/R-26's un-dispatched
   algorithmic fix, R-18's unfinished loops) — fold the contract items
   into the held batch explicitly so they stop being invisible.
4. **Annotations owed under R-28** (texts ready): beside R-8 (the 250-rung
   inconsistency, resolved by strictest-alias); beside R-25 ("FIRE_SIDE"
   is h=5-only, computed on a receipt that is not STOP's estimand).
5. **Second delegation tranche** per question 5; **define the tick** and
   the queue discipline (blocking-first is already the informal practice);
   **a read-cursor** on the dispatch ledger.
6. **Amend R-6's taxonomy** with the four missing distinctions the churn
   traced to: a GUARD subclass; class-on-VALUE not row; measured vs
   modelling-choice-with-measured-support; quantifier domains/aliases.
7. **Promote the sealed-receipt pattern** (build-allowed/read-forbidden
   with auditability mandatory) to a standing §2.1 clause.
8. **Adopt the code-vs-frozen-text conformance lens** at freeze time and
   at every probe commit citing a frozen bar — both members of that defect
   class were caught by cross-plane reading, zero by owner selftests.
9. **Sync §2/§1 to the operative state** (the `r=60` authority
   contradiction; the stale live-case; DE's "none built" cell).
10. **Pre-stamp naming rule**: receipt-visible identifiers are
    deploy-equals-freeze; touch the register before first stamp.

**CONFIRMED-CORRECT, preserved with equal weight** (the commission's
requirement): the freeze-first machinery end to end (R-1/R-7/R-8/R-14/
R-19/R-20 — the counterfactual is an untrusted central negative); R-9's
redirect (the seat adding analytic value no plane had posed); R-10 (DA:
"the most productive ruling of the session"); R-12/R-15/R-16-as-executed/
R-17/R-24-as-decision/R-26/R-28/R-29/R-30/R-32/R-33; the three-part Class-D
amendment test, which never churned while everything around it did; and
the routing of these very challenges out of the coordinator's own ledger.

Stop counter: 0 (MUST-FIX-class findings present). Iteration 2 runs after
the coordinator applies; Q-DA-3 adjudication and application-verification
are its first items.

### Iteration 2 — 2026-08-23 — verdict: `DEFECTS_FOUND` (one confirmed challenge; application-verification largely positive)

**1. Q-DA-3 ADJUDICATED: CONFIRMED — Class D's clause (c) is
deterrent-asymmetric, and for refutation-invalidating amendments it is the
objective, not the cost.** Verified against the primary texts (the
three-part test at R-6/§5a; the challenge at §0a; the named route at SP §4's
`primary_horizon` row): clause (c) — "explicitly invalidates every verdict
computed under the old bar" — was designed as a COST that deters casual
amendment. That deterrence only operates when the standing verdict is worth
keeping to the amender. **Every frozen verdict in this programme is a
refutation** (DEAD/DEAD, FIRE_SIDE-negative), so an amender who wants a
refutation gone satisfies (c) *eagerly* — the clause hands them a
legitimated erasure mechanism, and the entire protection collapses onto
clause (b) ("not the result itself"), which is judgment-laden. The concrete
route DA named is real: eth h=15/30 are the two spans-zero cells with no
recorded population-artefact defence — "new information" citing them could
motivate a bar amendment whose (c)-step invalidates the DEAD verdicts, with
every formal box ticked. DA's withdrawn sharpening is noted and does not
touch the structural claim.
**Recommended fix (content offered for the owning seat to ratify, per
R-36's whether/what split): add clause (d) for any amendment whose effect
could convert a refutation toward a pass — the re-run must ALSO be reported
under the OLD bar, permanently, beside the new (dual-bar reporting).** This
converts (c) from erasure into annotation: an amendment can add a second
reading but can never remove the refutation from view. It is R-28's
append-only philosophy and R-10's supersede-never-overwrite applied to
verdicts, it adds no asymmetric bureaucracy, and it makes the directional
attack pointless rather than forbidden-but-arguable.

**2. APPLICATION-VERIFICATION of iteration 1 (R-36 checked against the
ledger, and against its own standard):**
- ADOPTED: rec 1 (premise blocks) = clause 3; rec 2 ("applied names the
  mechanism") = clause 1, STRENGTHENED — the coordinator verifies the
  artifact, not the report, reversing the exactly-backwards trust
  direction the ruling itself names; the PROSE-ONLY class = clause 2; the
  R-18 extension to the full content class (plans, gates, mechanism
  choices, fact-correction targets — "the coordinator rules on WHETHER,
  not on WHAT") adopts the review's one-root finding wholesale.
- FIRST APPLICATIONS, verified present: **R-24 re-opened as PROSE-ONLY**
  (ledger line confirmed) with landing evidence correctly NAMED in the
  re-open itself — a check that FAILS when a sign-blind threshold is
  supplied — which is the new standard applied to the new standard's first
  output. Correctly formed.
- STILL OPEN from iteration 1's recommendations: the R-8 and R-25
  annotations (texts ready, not yet placed); the R-21/R-26 algorithmic-fix
  dispatch (still absent); recs 5 (tranche/tick/read-cursor), 6 (taxonomy
  amendments — Q-DA-3's clause (d) now joins that set), 7 (sealed-receipt
  standing clause), 8 (code-vs-frozen-text lens), 9 (§2/§1 sync), 10
  (pre-stamp naming). R-3/R-13's contract items ride the held batch,
  correctly.
- **R-36's own landing evidence under its own clause 1** (the
  hold-me-to-it duty, first exercise): the R-24 PROSE-ONLY marking exists
  (verified); the clauses are otherwise prospective — their landing
  evidence is every subsequent ruling carrying a premise-artifact citation
  and a landing-evidence field. **Standing DE duty recorded: from R-37
  onward, each ruling is checked for both on arrival; absence is called
  out in the next report, per the coordinator's own instruction.**

Stop counter: 0 (Q-DA-3's confirmation is MUST-FIX-class against R-6's
design; the taxonomy amendment set is the coordinator's to apply).
Iteration 3 verifies the remaining applications and any coordinator
disagreements filed under their name in §4a.

---

## OPS contribution — coordinator review (R-34)

### 1. Rulings OPS worked around or under-escalated — two, both mine to own

**(a) R-27's three-hour bound. I complied and should have pushed.** At 11:42 I
had the decisive fact — `cpu_frac 0.0014`, `stall_frac 0.9799`, state `D`,
measured twice over 90 s and 111 s — and I wrote that "there is no conclusion
coming". Then I **armed a timer for 12:40 and waited**. R-30 later made the
point I should have made myself: once "it is provably not progressing" is a
measured fact, a deadline about patience is irrelevant. **Concrete cost: ~40
minutes of a blocked evaluation lane after the blocking fact was already in the
ledger.** I did not work around the ruling; I under-escalated against it, which
is the quieter failure and the one this question is really asking about.

**(b) The seven asks I filed in prose and never re-raised.** Two sat from early
morning (`STALLED_LANE_POSITION.md` §8) until §0a existed. **I treated an
unanswered question as answered-by-silence and kept working** — the workaround
was continuing without the ruling rather than escalating its absence. The
register fixed the channel; it does not fix the habit, and the habit was mine.

**(c) Not a workaround but worth recording as a live hazard:** at 12:33 I applied
`Q-OPS-2` to `contracts.yaml` under **R-33(4)**, and at 12:36 **R-35** held that
exact question as part of a single batch. I reverted and submitted a delta
instead. **Two rulings can conflict inside three minutes** when delegation and
batching are issued in parallel; a plane acting correctly under one can violate
the other. Worth a precedence rule: the later ruling wins, and a plane that has
already acted reverts rather than arguing.

### 2. Rulings that were CORRECT and load-bearing — keep these

- **R-22's OOM priority split** — batch units `OOMScoreAdjust=1000`, collectors
  untouched and never restarted. The reasoning (a batch is restartable, live
  venue capture is irreplaceable) is the single best-argued call of the day, and
  it held through a 162-minute memory pathology **without a collector losing a
  second of tape**. Note this is separable from the `MemoryHigh` error in the
  same ruling: the priority half was right, the throttle half was wrong.
- **R-30 — "the fact beats the deadline."** Correct, and it generalises beyond
  this incident: whenever a measured predicate is available, a time bound that
  proxies for it is strictly worse.
- **R-8's Class-A/Class-D split of the τ ladder.** It closed a real hole — a
  refutation could have been converted into a pass by extending a "configuration"
  grid — and the generalisation it spawned then caught `refuse_k`, `quote_size_pin`
  and `tau_decision_rung`.
- **R-16's `pid=<n> start=<ticks>` lock format.** Small, exactly right, and it
  made `TIER1_LOCK`'s pid-reuse detection *exact* rather than heuristic.
- **R-26's regime rule.** It stopped me publishing a contaminated exponent.
  Given the corrected inputs would have produced `log(10.5)/log(1.11) ≈ 22`, the
  rule prevented a number that was not merely imprecise but absurd.
- **R-12 — repair the key, do not exclude the data.** Correct, and DA's census
  (499 keys / 264.8 M records) vindicated it.

### 3. Rulings that were WRONG, with the failure case

- **R-22's `MemoryHigh=12G`** — already owned in R-32. Failure case for the
  record: swapless, all-anonymous working set, `file` reclaimed to 0, so
  `memory.high` had nothing to reclaim and degenerated from throttle to stall.
  162 min elapsed, 85 min of it work, 0.065 % of one core at the end,
  `memory.max` never reached so **never killed**. Removing it turned the same
  day into an **8-minute run**.
- **R-21's exponent arithmetic.** Both premises measurably false: 2026-08-20 is a
  **full 24 h day** (288/288 windows; 63.7 % of btc's committed quote rows
  predate the era boundary the ruling treated as the start of collection), and
  the two CPU figures compared **different phases** (Tier-2-only vs
  Tier-1-build + Tier-2). **Failure case: it was about to route an algorithmic
  fix to DA on an implied exponent of 2.44.** Measured like-for-like, it is
  **~1.4**.
- **R-27** — superseded within the hour by R-30; see §1(a).

### 4. Work created without changing an outcome

R-27 (the armed stop was never used), and the Class-C restatements — R-29 said
"measured values are adopted, not chosen", then R-33(5) spent another ruling
saying it again. Both were correct and neither changed what any plane did.

### 5. The thing the coordinator cannot see from the seat

**Every defect in the coordinator's rules today was found by a plane while doing
something else.** I did not find `MemoryHigh` by auditing R-22; I found it
because a livelock happened in front of me while I was measuring an exponent.
The τ-ladder hole came out of a routine class sweep. **This predicts that a
scheduled audit of the rulings will underperform the incidental discovery rate**,
and it argues for keeping planes *executing* rather than reviewing — which cuts
against commissioning too many review loops, this one included.

### 6. What OPS would NOT change

The worker/coordinator split itself. It has cost real latency, and it stopped at
least four values being re-cut after their answers were visible — including one
of mine (`verdict_coins`, where I proposed Class D and SP Rev 7 escalated it
*further* than I had). A plane arguing its own bar into place would not have
survived that scrutiny.

## DA contribution — coordinator review (R-34)

DA filed 12 register entries and has had 4 rulings land directly on its work
(R-16, R-18, R-20, R-32/33). Findings below are what DA can see that the
coordinator cannot.

### 1. Rulings DA WORKED AROUND rather than escalated — three, all mine to own

The question was asked plainly so it gets a plain answer. In each case DA acted
first and flagged second, which is not the same as escalating.

**(a) R-8's internal contradiction — DA resolved it unilaterally in a worker
document.** R-8's adopted text lists `{0,50,100,250,500}` as Class A while
citing R-1's freeze of the 250 rung as its own precedent. Both cannot hold.
DA wrote `SP_PLANE_PLAN` §4 with 250 **removed** from the Class-A set and
stamped the row "R-1, narrowing R-8" — i.e. DA decided which half of a ruling
survived, in a register the coordinator reads as authoritative — and only then
filed Q-DA-6. The escalation was real but it came after the act.

**(b) DA added two entries to a user-ratified set.** `knowledge_lag` and
`primary_horizon` were written into §5's operative block, which is stamped
"user-ratified 2026-08-23, R-6" and which R-6 enumerated as **six** entries.
DA labelled them "recorded, NOT ratified" in the same block — but the labelling
came in a later revision, after they had been sitting inside the ratified block.
`ev_replay.SP_OPERATIVE` still carries R-6's six, so the document and the
deployed set disagreed because of DA.

**(c) DA renamed a published provenance address.** §5 mandated
`sp_operative_v1` while every receipt on disk carried
`SP_PLANE_PLAN_s5_operative_R6`. That is R-10's own defect committed by the
plane that had been citing R-10 at other planes all day. Now withdrawn under
R-33, but DA filed it as a question (Q-DA-10) rather than noticing R-10 already
answered it.

**Pattern:** all three are DA treating "I flagged it in the document" as
equivalent to "I asked". R-33's channel rule and the §0a register both exist
because that equivalence is false, and DA was relying on it before the register
existed.

### 2. Rulings that were CORRECT and are load-bearing — name them so they are kept

**R-10 (an amendment that changes content must change the address) is the most
productive ruling of the session for DA.** It has since caught four distinct
artifacts: the run record, the canary report, the Tier-1 partitions, and — by
DA's own violation — the operative set name. It is also the ruling DA reached
for most often without being told to. Keep it exactly as worded; its generality
is the point.

**R-18 (coordinator does not author module plans) was vindicated harder than
expected.** The review DA ran found 60 findings / 31 MUST-FIX in iteration 1
alone on a single-pass document. That is not a criticism of the draft; it is
evidence that *any* single-pass plan carries that defect density, which is an
argument for the loop, not against the author.

**R-20's substance was right even though DA's framing of it was wrong.** DA told
the coordinator that R-20 "opened a hole"; it did not — R-8 had already
generalised Class D to falsifier-bearing rows and Class D is the most
restrictive class, so R-20 *raised* the price. The snapshot-by-value rule is
sound and should not be narrowed on DA's mistaken framing.

**R-16's refusal was correct to accept.** DA declined to implement the reclaim
mechanism R-16 authorised, on evidence that the lock was never orphaned
(`flock` is fd-held and released on death; the lock was acquirable). The
coordinator accepted the refusal. That exchange is the model: a ruling issued on
a plausible-but-wrong premise, refused with evidence, and not forced.

### 3. Where the gate list was over-broad — corroborating R-33 with DA's data

Of DA's 12 filed questions, **two required no coordinator at all** (Q-DA-7, a
plane-order violation the architecture already settles; Q-DA-10, settled by
R-10) and **two more were confirmations of frozen text** (Q-DA-12, Q-DA-13 —
which the coordinator has already named as the proof). That is **4 of 12, a
third of DA's filings, that a narrower gate would have absorbed**. R-33's five
carve-outs map exactly onto them.

### 4. One structural observation the coordinator cannot see from the seat

**Every plane's escalations are written in that plane's own document, and the
coordinator reads them serially.** DA's §10 grew to fifteen items across seven
revisions; three iterations of review found that §10 was where *most of DA's own
errors lived* — misattributed rulings, an unverifiable quotation, an escalation
naming a hole that did not exist, and a sharpening that a draft-status check
refuted. The escalations were the least-reviewed and highest-stakes text in the
document, precisely because they read as requests rather than claims.

**Suggestion:** an escalation should carry the same evidentiary burden as a
finding — a named authority, a verified quotation, and a stated failure case —
and should be reviewed *before* filing, not after. DA has adopted this for
itself; the register makes it enforceable programme-wide.

### 5. What DA would NOT change

The worker-measures / coordinator-decides split. DA argued against exactly one
ruling on substance (R-16) and was right; DA argued against R-20's framing and
was wrong. That ratio is the argument for keeping the split rather than
loosening it: the plane's judgement is good on facts it has measured and
unreliable on the shape of rules it has not.

---

## DA contract delta — FINAL, for DE's single §2.2 submission (R-35)

Six items. **One item from DA's earlier §7 list is WITHDRAWN**, and the
additive / non-additive split is called out as the ruling requires.

### ADDITIVE — new types and fields, no existing writer changes behaviour

| # | change | why it is needed | consumed by |
|---|---|---|---|
| A1 | **Four `SpecRecord` schemas**: `SP-Venue`, `SP-Instrument`, `SP-Strategy`, `SP-Scenarios` | None exists in v22 — they appear only in `PM_ARCHITECTURE` prose, which that file forbids from introducing a type. Without them `DecisionProblem.spec_snapshot` and `ResolvedContracts.spec_hash` have **no producer** | DE, BE, EV |
| A2 | **An SP spec-resolver port** on `BE-*` / `DE-*`, signature `get(ParamId, ScopeKey, at) -> ParamValue \| Unavailable` | v22 has **no spec/params port at all**, which is why `de_constraints.py` reached into `ev_replay` for the register (Q-DA-13). Until this exists the plane-order fix is a duplicated literal | DE, BE |
| A3 | **A fee family** `(PRODUCT_PQ, {rate, incidence, size_rounding})` | No fee family is defined in v22. `incidence` is required or the measured **taker-only** fact cannot be stored and a maker leg computes a non-zero fee against 744/754 observed zero legs | DE, EV |
| A4 | **`DE-Constraints.consumes: CapitalBudget`** | v22 has `CapitalBudget` **produced by `DE-Allocator`** and absent from `DE-Constraints.consumes`, so the size chain terminates nowhere. `DE_MODULE_PLAN` already lists this change | DE |

### NON-ADDITIVE — migrations; existing data or writers are affected

| # | change | why it is a migration | blast radius |
|---|---|---|---|
| **N1** | **`ParamValue.valid_from` / `valid_to`** | `ParamValue` carries `valid_for: ScopeKey` — a *scope*, not an interval — so **R-VERSION is unsatisfiable for every register row**. Optional fields would leave the rule unenforceable, so to satisfy R-VERSION they must be **required**, which every existing writer must then supply | every `SP-Params` write; `params.at(t)` cannot be built without it, and without `at(t)` a replay resolves *today's* value inside yesterday's run — a look-ahead in parameter space |
| **N2** | **`Provenance` enumerated, with an `OBSERVED` member and de-collided axis names** | R-3 ruled **five** members; DA proposes **six**. `OBSERVED` is needed because `provenance` is carried by 15 non-`ParamValue` types and a wire-read `FlowArrival` fits none of the five — it would be mislabelled `MEASURED` and become decision-gating with no artifact. Separately `Known.t_known_prov` already uses `OBSERVED\|IMPUTED\|ASSUMED` on the *knowledge-time* axis, so reusing two names on the *value* axis leaves a checker unable to tell which fired. **This is a delta on a ruling, not just on the contract** | R-3; every carrier of `provenance` |
| **N3** | **An authority axis** — e.g. `ParamValue.set_by: COORDINATOR \| USER \| MODULE(ModuleId)` | `ParamValue.owner` is a `ModuleId`; the coordinator is not a module and `EV-Gates` gives `STOP-MM-VIABLE` `owner = the user`. So the amended R-PROV clause "nothing but the coordinator may set a CHOSEN value" is keyed on a value the contract **cannot represent**, and passes vacuously — a gate that cannot fire, inside R-PROV's own fix | R-PROV enforcement; every CHOSEN row |

### WITHDRAWN from DA's earlier list

- **`ParamId.namespace`** — DA listed this as a needed contract change. It is
  **already in v22** (`ParamId{namespace: ModuleOrPluginId, name: str}`, both
  non-optional). The real defect is that **no DA register row supplies one**,
  which is a §4 population gap for DA to fix, not a contract change. Caught by
  iteration 4; withdrawn here so the batch is not padded.

### Note for the consolidator

N2 is the only item that changes a **ruling** (R-3) rather than only the
contract, so it may warrant being split out even within the non-additive group.
N1 and N3 are the two that make currently-unenforceable rules enforceable
(R-VERSION and R-PROV respectively) — if the batch is trimmed, those are the two
with teeth.
