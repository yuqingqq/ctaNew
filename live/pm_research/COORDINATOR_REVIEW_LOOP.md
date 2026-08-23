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

(appended per iteration)

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
