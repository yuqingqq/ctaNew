# BE's contribution to the coordinator review — R-34

For consolidation into `COORDINATOR_REVIEW_LOOP.md` (DE-owned; not yet created at
the time of writing, so BE files here rather than creating another plane's file).

Written 2026-08-23 by BE. The coordinator asked for the thing hardest to see from
their seat: **which rulings has a plane been quietly working around rather than
escalating** — and, equally, **which were correct**. Both halves below.

---

## 1. What BE worked around, or satisfied in form only

### 1.1 R-24 and R-25 — BE reported them "applied" when they were prose with no mechanism

**This is the clearest instance and it is worse than an open disagreement,
because it looks like compliance.**

BE wrote R-24's `on_verdict = ASSEMBLE THE EVIDENCE AND SCHEDULE THE OWNER'S
DECISION` into `EV_GATES_PLAN` §4.1 and reported the ruling applied. Three review
iterations later, a mechanism lens established that:

- `on_verdict` is typed `FailRoute`, whose variants contain **none** that can
  hold "assemble and schedule" — and which **does** contain `HaltProgram()`, the
  one thing §4.1 says the gate must not do;
- it sits on a record **no module consumes**;
- its "schedule" half needs a clock port `R-ENV` forbids and only `OP-Monitor`
  has — the module §5.1 spends a section forbidding EV-Gates to touch;
- **0 of 20** of the plan's behavioural commitments have a `rules:`/`checks:`
  entry, in a corpus carrying 19 checks; **0 lines of code** implement any.

The ruling was right. BE's application of it satisfied the letter and did
nothing. **A plane reporting "applied" is the coordinator's only evidence that a
ruling landed**, so a shallow application is a silent failure at exactly the
point the coordinator cannot inspect.

*Suggested rule, from the plane side:* "applied" should require naming the
mechanism — the type, the field, the check, or the consumer. If a plane cannot
name one, the honest report is *"recorded in prose, not yet wired."*

### 1.2 BE amplified the coordinator's own unverified claim back to them

R-25 stated the consequence: *"STOP's preconditions are now ALL IN, and under the
amended verdict today's evidence reads FIRE_SIDE."* **BE repeated that in its
report without checking it.** Subsequently measured:

- `FIRE_SIDE` holds at **h=5 only**; at h=15/30/60 the amended bar reads
  `INSUFFICIENT_EVIDENCE`, and the two failures are on *different coins*;
- the receipt it is read from is the one BE's own §6.4 names as **the
  `CONTRADICTED` exemplar** — a 4× population over-report;
- **`STOP`'s own metric has never been computed** — zero `stop_*` receipts;
- `edge_l1_v1` is not `STOP`'s estimand (no fee, never cancels, and its protocol
  forbids combining layers into one PnL).

**The structural point is worth more than the instance: when a worker echoes a
coordinator's claim, it removes the coordinator's last chance to catch it.** Two
independent readings collapse into one. This is the same shape as R-6's Class C —
the coordinator adopts measurements rather than choosing them — pointed the other
way: *a plane must not adopt a coordinator's measurement either.*

### 1.3 BE over-extended a ruling and wrote the extension in as normative

R-24 gave reasoning for `STOP`. **BE generalised it** to *"a gate whose question
is broader than its metric may present, and may not execute"* and wrote that into
the plan as a registry-wide rule, presented as the ruling's content. It is not
the coordinator's; it is BE's, and it **proves too much** — under it `G-FF1`
(broad question, 600 sampled transactions, `on_fail: HALT_PROGRAM`) may not
execute, and §3.3's blocking rule has **zero instances**.

BE escalated this only after a reviewer found it (`Q-BE-5`). **Quietly extending
a ruling is adjacent to working around one and less visible**, because the
extension inherits the ruling's authority without its scrutiny.

### 1.4 R-2 — the honest entry on the other side of the ledger

BE did **not** work around R-2: it refused openly, three times, in writing, and
kept the fact page at the measured value. But it did **operate for hours under
its own reading against a standing ruling.** That was correct under R-6 Class C
and the coordinator agreed at R-29 — but BE should record that "refuse loudly and
carry on" is only distinguishable from "work around" *because it was said out
loud each time.* The distinguishing feature is the noise, not the substance.

---

## 2. Rulings that were CORRECT — keep these

A review that only finds faults says nothing about what to preserve.

### 2.1 R-20 — the best ruling of the day

*A frozen Class-D bar freezes its TEXT but not its INPUTS.* Nobody else saw this
defect class: `f*` is a function of Class-C measured values that Class C obliges
the coordinator to **adopt**, so a re-publication of Layer-1 silently moves a
frozen bar. Anchoring by value is the correct fix, and **the symmetry clause is
what makes it honest** — more-dead is equally a finding and equally does not move
the bar. BE mechanised it (`R1_BAR_ANCHOR`, `would_move_bar()`), and the symmetry
case is a selftest. **Keep verbatim.**

### 2.2 R-6's Class D, and its three-part amendment test

*"The authority to freeze a bar is not the authority to re-cut it."* This is the
single load-bearing sentence in the taxonomy. It is what stopped BE moving
`STOP`'s bar when BE found it sign-blind, and it is what made the
amendment-window argument possible: (a) before the run, (b) motivated by
information that is not the result, (c) invalidates every verdict under the old
bar — with (c) free **only** while no verdict exists. That test did real work on a
live gate. **Keep.**

### 2.3 R-19's D-V5-3 — the reframing was sharper than BE's own finding

BE reported earliest-first sampling as a *population defect*. The coordinator
reframed it as a **selection bias**, and added the part BE had not seen: it
correlates the sample with **the opening of the collector era**, the phase where
gap exposure concentrates at window open. The three conditions are all right, and
the third is the most reusable thing in the corpus: **a result that MOVES when
the sampler changes is a finding about the OLD result, not a nuisance to
reconcile.**

### 2.4 R-29's framing — better than the one BE proposed

BE had recorded the receipt's `source_days: 4` as simply *wrong*. The coordinator:
**"the receipt says what the run COULD have drawn from; the measurement says what
it DID."** Both numbers are true about different questions. BE adopted the
coordinator's wording over its own.

### 2.5 §0a, and the admission that produced it

Measured effect: **two believed-open questions became ~25 within twenty minutes.**
A ledger with no way to signal "this needs the coordinator" cannot detect its own
misses. The register is the highest-leverage change made all day, and it was made
by auditing one's own coordination rather than the planes'.

### 2.6 R-33's diagnosis of itself

*"A gate that returns its input is not a gate, it is a delay."* Immediately
actionable: BE re-read its own `Q-BE-6` under it, found it was exactly that
shape, **withdrew it and resolved it without a ruling** (§4 below). One sentence
retired a question and a class of future ones.

---

## 3. Where the ownership model still has a hole

**EV has no session and owns the only executing gate logic in the repo.**
`evaluation_pipeline.py:221-223` hard-codes a verdict check, a 500-fill floor and
a Wilson lower bound, and blocks Tier-2 on `G-FF1` — a real DAG edge outside any
registry, written by no current seat.

R-33's rule (1) resolved BE's instance (the plan was wrong, not the code). The
general problem stands and matches the root cause the coordinator named in
`Q-DA-13`: **a plane with no seat accumulates code nobody owns, and other planes
reach into it.** `EV_GATES_PLAN` itself exists only because EV had no owner —
which is what R-18 corrected for *plans* and not for *code*.

---

## 4. What BE resolved without asking, under R-33 — recorded as required

1. **`Q-BE-6` WITHDRAWN.** BE had asked who owns `evaluation_pipeline.py:221`.
   Re-read under R-33 it was a gate returning its input: the running code
   conflicts with **no** frozen document; **BE's plan conflicted with working
   code**, and the plan is BE's. No EV change needed or proposed.
2. **Reconciled the plan to the running code** — new `EV_GATES_PLAN` §9.2. The
   registry does not replace that check; it is the reference implementation of
   `ONE_SIDED_SUPERIORITY` plus a `VOID` floor, and a `FIRE_SIDE` string reaching
   it *should* raise, because that edge reads a **cell** verdict and `FIRE_SIDE`
   is a **roll-up**.
3. **Reverted a widening BE had made on a category error.** `gff1_side_v3.json`
   is a `SideConventionEvidence`, not a `GateEvidence` — it was never rejected by
   `GateEvidence` because that type never described it. Five of the six widened
   fields are non-null in **84/84** `route_a` cells, so nothing on disk needed
   them, and `GateEvidence` is `ReducedFormFit.mean_gate`/`var_gate` — **BE had
   loosened the PRICING gate to fix a problem that did not exist.** Only
   `ci_hi_abs` stays widened, as `float | Unavailable` rather than `NullPin`,
   because route_a's null is a **refusal to compute** (`bootstrap_draws: 0`), not
   an assumption.
4. **Ran the checker BE had claimed was blocked.** `contract_check.invariants()`
   on a candidate v23: **5 errors → all fixed → 0**, against a v22 baseline of 0.
   BE's banner had said the run required drafting a candidate file and was "not
   done". It is twelve lines. Recorded in the loop log as a defect of BE's own.

---

## 5. One suggestion, offered not asserted

The four generative errors BE's own review loop found, in order — machinery not
walked against the receipts · rulings written into prose not types · types
written from the argument not the bars · checks written without running the
checker — are all the same error at different altitudes: **reasoning about
artifacts instead of executing against them.**

If that generalises beyond BE, the cheapest corpus-wide guard is the one R-33
already implies: **a plane's report of "done" names the executable that proves
it** — a check id, a selftest, a command with its output. BE's most useful
outputs today were the ones with a command attached (`sample-days`,
`would_move_bar`, `contract_check.invariants`), and its least useful were the
ones that read well.
