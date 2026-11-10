# REVIEW 275 — rank the fifteen by cost, and cull them to ten

REV round 239. Filed 2026-09-12T03:02:55Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

**Your correction (1) verified at the fetched ref:** the claim-checker now
counts **1** at `origin/mm-research` and `RETIRED 2026-09-12` greps **1** there.
My `??` was correct against the ref I read at 02:37:58Z. Timing artifact,
recorded as such.

**And correction (2) is the more instructive half, as you say:** a push that
exits zero while the three-way merge keeps the pre-retirement side is **a
successful command with an unsuccessful effect** — the shape §7k already
addresses (*landed = a count at the fetched ref, never a local commit*) and
which you closed the right way, by grepping the fetched ref for the content
rather than trusting the exit code.

---

## 1. RANKED BY COST INCURRED TONIGHT

Cost measured as *decisions moved* > *claims retracted* > *rounds spent* >
*wall-clock*. Only what I can point at.

| rank | rule | what it actually cost |
|---|---|---|
| **1** | **#3 population boundary goes where the CONDITION changed** | I violated it **twice in three rounds**, and it moved the planning rate 0.875 → **0.700** and P(`INSUFFICIENT_EVIDENCE`) 0.023 → **0.42** — a decision-grade number already given to the user |
| **2** | **#7 state the CRITERION as well as the population** | my 90.9% survived a round because the criterion was unstated; DE caught it; directly upstream of #3 |
| **3** | **#8 deliver a claim in the form the consuming instrument reads** | **three rounds** of REV↔coordinator on a ruling already delivered — the largest pure coordination cost I can measure |
| **4** | **#1 liveness from a heartbeat, not an anomaly stream** | blocked a build, produced **two wrong diagnoses from you inside an hour**, with the right signal sitting unread in `collector_health.jsonl` |
| **5** | **#4 a prose field inside a measured artifact is still an assertion** | the `limits` block caused a **ruling that had to be reversed**, and `UNPOPULATED_WS_ZERO` took out my REVIEW 251 premise |
| **6** | **#13 a stated adoption is not an adoption** | five instances (three yours, BE's, and the merge in your correction 2); cost is rework rather than wrong conclusions |
| **7** | **#9 a record written before its subject completes** | the 6.6-minute placeholder produced my "09-11 failed" mis-reading and a wrong denominator |
| **8** | **#6 numerator and denominator from ONE source at ONE ref** | my 344/336 gap, plus a **wrong causal claim** ("eight failed `ast.parse`") that I had to retract |
| **9** | **#2 a test declares a minimum EFFECT, not only a minimum sample** | **cost nothing — found before the clock started.** Highest future value, zero incurred cost, and that is the argument for writing it |
| **10** | **#14 declare the acceptance criterion before the result is visible** | cost nothing; BE practised it and it worked |

## 2. THE CULL: fifteen → ten. Five do not survive as separate rules.

**Drop outright — 2**

- **#12, "P(fewer than k), not E[·] − k".** This is a **statistics correction,
  not a programme rule.** It is true, and it is one instance of "use the right
  statistic", which cannot be enumerated. A runbook that starts collecting
  individual statistical errors reaches fifty entries and none is findable.
  **The next session's statistical error will be a different one.** It belongs
  in the declaration where the band risk is computed, as a field with the
  formula beside it.
- **#15, the empty-shell / same-day-median check.** A **detector, not a rule** —
  specific to per-window files with a size distribution. With different
  artifacts it does not apply. It belongs in the lane's preflight. Its durable
  content is already **#9 plus #4**: a 1.2 KB shell is a record whose
  incompleteness was not distinguishable from completeness.

**Fold — 3**

- **#5 (status name must not assert what its value cannot carry) into #4.**
  Both say *the label is not the measurement*. Two rules saying one thing is
  worse than one rule with two worked examples — keep both examples
  (`UNPOPULATED_WS_ZERO`; `ESTABLISHED_ZERO` beside `NOT_ESTABLISHABLE`).
- **#11 (entity-level vs event-level denominator) into #7.** It is CLAUDE.md
  rule 8 with a worked example — 6 of 218 accounts, not 10 of 1,056 legs. The
  generalisable half is already written; the example is what is missing.
- **#10 ("not determinable" is a usable result) into §7k.2.** It is a
  **permission, not a rule**, and on its own it licenses giving up early.
  "Not determinable" is usable **only** with what was searched and excluded —
  which is §7k.2, already written. One clause there, not an entry.

**That leaves ten: #1, #2, #3, #4(+5), #6, #7(+11), #8, #9, #13, #14.**

## 3. ONE OF MINE I WOULD NARROW BEFORE YOU WRITE IT

**#8, "deliver a claim in the form the consuming instrument reads."** As stated
it is too broad — with no automated gate it degenerates into "tell the right
person", which is not a rule. The durable core is narrower:

> **When an automated gate consumes your output, the gate's input format IS the
> delivery requirement.** A ruling delivered in prose to a predicate that reads
> a declaration has not been delivered.

You have already had one of yours narrowed tonight; this is mine, and I would
rather narrow it now than have it retracted in a second session.

## 4. WHICH OF THE TEN I AM LEAST SURE OF

**#6.** "Numerator and denominator from one source at one ref" is at risk of
reading as a duplicate of #7 and of rule 8, and a reader who cannot tell three
adjacent rules apart applies none of them. It is genuinely distinct — #7 is
*state the criterion*, #6 is *do not mix sources* — but if you write it,
**write it as a clause of #7 with its own worked example** (344 from `ls-tree`
at one commit, 336 from `glob` at another) rather than as an eleventh entry.

**#14** is the second least certain: it is close enough to CLAUDE.md rule 6
("declare the null before the result") that it should be written **as an
extension of rule 6** — *the rule is not only for statistical nulls; any
acceptance decision declares its criterion before the result is visible* —
rather than as a new rule, or it will be read as a duplicate and skipped.

## 5. What I excluded

The ranking is by cost I can **point at in this session**; a rule that cost
nothing tonight may be the one that matters most in another, which is why #2
sits ninth on cost and first on value — do not read the ranking as a priority
order for writing. And the cull is my judgement about durability, not a
measurement: I have no second session to test it against, which is exactly the
thing being predicted.
