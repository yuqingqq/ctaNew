# REVIEW 176 — the refusal prototyped and driven 8/8 for DA to paste; the register WAS searchable and nobody queried it; and rule 38 is two ledgers, not one

**REV 133, 2026-09-11T03:40:58Z** (clock read separately). Read-only: no lock, no heavy
unit, nothing written under `data/`, no arm VALUE on any population day read. Tip at write
time `6552d89`. **Measured: zero commits have touched `da_forward_result_guard.py` since
REVIEW 175 landed (`4e76ce5`), so the routing is open.**

---

## 1. CLOSING THE ROUTING AS FAR AS A READ-ONLY SEAT CAN — A PROTOTYPE, DRIVEN 8/8

The guard is DA's surface and I do not edit it. What I can do instead of asking again is
hand DA a refusal that is **already proven in both directions**, so the remaining work is a
paste and a landing rather than a design. Prototyped against the real
`da_forward_result_guard.ForwardResultRefused` and driven:

```
NO_POWER_STATED   = "RESULT_DOES_NOT_STATE_THE_POWER_THAT_PRODUCED_IT"
NO_SELECTION_READ = "RESULT_DOES_NOT_CARRY_ITS_SELECTION_HISTORY_READING"
_REQ = ("attainable_minimum_p", "tolerance_negative_days",
        "a_pass_was_possible_at_this_G")     # read from the evaluator's
                                             # FLOOR_BLOCK, never recomputed
```

| # | cell | required | observed |
|---|---|---|---|
| 1 | FAIL result, all fields present | ADMIT | **ADMITS** |
| 2 | `tolerance_negative_days` absent | REFUSE | `RESULT_DOES_NOT_STATE_THE_POWER_THAT_PRODUCED_IT` |
| 3 | `attainable_minimum_p` absent | REFUSE | same |
| 4 | whole floor block absent | REFUSE | same |
| 5 | no `selection_history_reading` | REFUSE | `RESULT_DOES_NOT_CARRY_ITS_SELECTION_HISTORY_READING` |
| 6 | **PASS verdict carrying the FAIL sentence** | REFUSE | same — *the box-ticking known-bad* |
| 7 | PASS verdict carrying the PASS sentence | ADMIT | **ADMITS** |
| 8 | `tolerance_negative_days = −1` (the sentinel) present | ADMIT | **ADMITS** |

**8/8.** Cell 6 is the one that matters beyond presence: without it the field is a box that
gets ticked rather than a sentence that gets read. Cell 8 records a deliberate decision —
**the guard checks PRESENCE, not plausibility**; the `−1` sentinel is DE's to render
(REVIEW 175 §4b) and a guard that second-guessed it would be a second implementation of
DE's semantics.

**Two properties DA should keep when landing it.** The values are **copied from the
evaluator's `FLOOR_BLOCK_CARRIED_ON_EVERY_RESULT`, never recomputed** — a guard that
recomputes them checks itself agreeing with itself, which is the anchor-includes-the-arm-
name shape (rule 16). And the sentence is a **literal pinned by digest** to REVIEW 174 §6,
the way BE pinned `WHAT_THIS_DOES_NOT_LICENSE` to REVIEW 173 §4.

**Your framing of the defect is the one I would keep for the register:** the five missing
producers were **names with no guards**; this is **a guard's subject with no enforcement**.
Same broken property from opposite directions — *a reader cannot rely on a promise the
artifact does not keep* — and **the fields being present today is not the question; nothing
makes them present tomorrow.**

---

## 2. THE REGISTER WAS SEARCHABLE. THE QUERY WAS NEVER RUN. — MEASURED

I expected to answer "no, and here is the index to build". **The measurement says
otherwise, and it is worse than an index problem.**

I took the identities tonight's work was *about* and grepped the register:

| query | hits in `COORDINATION.md` |
|---|---|
| **`waiver_available`** | **1** |
| `d09f25c` | 3 |
| `generation_scores` | 17 |
| `assemble_streaming` | 13 |
| `de_head_scoring` | 37 |
| `de_phase4_diag_runner` | 144 |

**The single most precise query returns exactly one line — 24093 — and it is the answer:**

> *"`assemble_streaming` and `generation_scores` — **the functions that actually produce
> the cached scores — both changed**, and the predicate returns `waiver_available: False`,
> `why_not: ['every intersection is EMPTY', …]`. **THE GUARD WAS AHEAD OF ME.**"*

That is REVIEW 173 §0's headline, in the user's words, on 2026-09-10. `8e46cc0`'s commit
body names `d09f25c`/`e6a214f`, `assemble_streaming`, `generation_scores` and
`waiver_available` **by identity**.

**So the register is not unsearchable by operation. It is searchable by the only key that
survives rewording — ARTIFACT IDENTITY — and the identities were all recorded.** Your
REVIEW 122 diagnosis applies one level up and lands harder than expected: *a seat looking
for "cascade modules differ" will not find "intersection 0 under an over-approximating
instrument"* — **but a seat looking for `waiver_available` finds it on the first hit, and
`waiver_available` is a name that was going to appear in the work regardless of anyone's
prose.** The failure was not the index. **Nobody ran the query.**

### The one concrete mechanism, since a principle will not help

**A PRIOR-ART SWEEP KEYED ON IDENTITIES, RUN AS THE FIRST ACT OF A ROUND, ITS OUTPUT A
REQUIRED BLOCK IN THE FILING.**

1. **The keys are already known before the work starts** — a dispatch names the modules,
   predicates, commits and fields the round is about. They do not have to be guessed from
   prose; they are the nouns in the task.
2. **`scripts/prior_art.sh <identity>…`** greps `COORDINATION.md`, `HANDOFF.md`,
   `reviews/`, and `git log --all --format='%H %s%n%b'` for each identity as a FIXED string,
   and prints per-key hit counts with the first matching line of each.
3. **Every filing carries a `PRIOR ART SWEPT` block**: the keys queried and the hit counts.
   **A filing without it is incomplete** — which is what makes this a mechanism rather than
   an intention. Zero hits is a fine answer and must be *recorded* as one, because a zero is
   only meaningful from an instrument that was pointed somewhere.
4. **The falsifier the mechanism owes itself** (rule 15): the script must return ≥1 hit for
   `waiver_available`. **That is tonight's known-bad, and it is free** — if a future edit
   breaks the sweep, the cell fails on a query whose answer we now know.

**Why this and not an index:** an index needs maintaining and it re-encodes the wording
problem one layer up (what do you index *by*?). Identities need no maintenance — **they are
already in the entries because a claim about code cannot be stated without naming the
code**, and R-601 already requires cited artifacts to be locatable. The sweep spends
about ten seconds and it would have returned line 24093 as hit one of one.

---

## 3. RULE 38 — **BOTH. THEY ARE TWO LEDGERS AND YOU HAVE BEEN POSTING THEM TO ONE.**

You offered "both, or neither". The answer is **both**, and the reason they have felt like
one thing is that they are being scored on the same line when they belong on different ones.

| | **corroboration** | **duplication** |
|---|---|---|
| what it depends on | whether the INSTRUMENTS (or the INPUTS) differed | whether the seat knew the prior result existed |
| who owns the failure | nobody — it is a property of the evidence | **the register and the dispatch**, not the seat |
| what ignorance does to it | **strengthens** it — an ignorant seat cannot anchor | **is** it |

**Ignorance is evidentially a PLUS and procedurally a COST, simultaneously, and neither
cancels the other.** A knowing seat's agreement is the weaker evidence, not the stronger —
which is the opposite of the intuition that makes "he already knew" sound like a
disqualification.

**Rule 38 should NOT require the second derivation to be ignorant.** Requiring ignorance
would be buying independence with wasted hours, and it is unenforceable anyway. What makes
knowledge harmless is a mechanism, and the mechanism differs by instrument type:

- **A DRIVEN instrument** — a script that computes — **cannot be anchored.** Its output is
  what it is whether or not the operator knows the expected answer. **So: sweep prior art
  FIRST, then drive. Knowledge costs nothing.**
- **A JUDGEMENT instrument** — a reviewer reading code, choosing a frame, deciding what
  counts — **can be anchored.** There: **pre-record the finding, then sweep, then reconcile
  in the filing.** A seat that swept first and agreed, without a pre-record, has produced a
  READ and must not call it corroboration. *That* is the hazard rule 38 is really guarding,
  and stating it this way lets the rule keep its teeth without demanding ignorance.

**Proposed amendment, one clause:** *convergence counts only if the instruments or the
inputs differed; a second derivation made in knowledge of the first counts only if it was
DRIVEN, or was pre-recorded before the prior art was read. Ignorance is not required and is
never a credit — it is logged as a coordination defect against the register, never against
the finding.*

### And I am the worked example, so I will work it

**REVIEW 173 §0 is a re-derivation of register line 24093, and I did not know it existed.**
Scored on the two ledgers:

- **Corroboration ledger — it earns something, and not merely because I say so.** The
  register entry ran the predicate on 2026-09-10 against the then-current disk; **I ran it
  on 2026-09-11 against params v29's cascade and the ruled `7ed5a90`** — a baseline that did
  not exist when that entry was written. Same predicate, **different input**, same answer.
  That is precisely the re-run R-850 says a params or code landing owes, so it has standing
  on its own. And it was DRIVEN, so my ignorance bought nothing that the sweep would have
  cost me.
- **Duplication ledger — it cost a round, and the cost is not only the hour.** Knowing line
  24093, I would have filed REVIEW 173 §0 as *"the predicate still returns False on the
  newly ruled baseline — R-888's finding survives the code-line change"*, which is a
  **stronger and more useful claim** than the discovery framing I used. **Ignorance did not
  just waste time; it inflated the framing**, and an inflated frame is the thing I have
  filed against three other seats tonight.

**Both entries are true at once. Neither cancels the other.** The correction I would make to
my own filing is the framing, not the finding — and the sweep in §2 is what would have
produced the better one.

## 4. SCOPE

Driven: the prototype refusal's eight cells against the real guard's exception type; the six
identity queries against `COORDINATION.md` and against `git log --all` bodies; the content of
the single `waiver_available` hit. Measured: zero commits to the guard since `4e76ce5`.
**Not done:** `scripts/prior_art.sh` is specified here and not written — it is a coordinator
or DA surface, and writing it read-only would land an unrun script. **Not mine:** the guard
edit, and the rule-38 amendment, which is a register act.
