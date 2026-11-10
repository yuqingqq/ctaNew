# REVIEW 249 — the fee trigger is identifiable, and the sensitivity is 5x to 250x light

REV round 212. Filed 2026-09-11T23:53:22Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 23:50Z.

---

# PART 2 FIRST — THE FEE RULING

I attacked it and it does not survive in its current form, but **not** where I
expected, and one of my prior objections is withdrawn by the data. Everything
below is read at
`data/pm_5min/derived/p003_da_onchain_fee_audit__20260905T155346Z.json`.

## 2.1 The trigger is NOT unidentified. It partitions by ACCOUNT, totally.

The audit's own two fields settle it:

| maker address | charged legs | that address's TOTAL maker legs |
|---|---|---|
| `0x0fd0ebb1ba53…` | 1 | 1 |
| `0x18b0b7105413…` | 1 | 1 |
| `0x2277c18fb73d…` | 3 | 3 |
| `0x8d009282a757…` | 1 | 1 |
| `0xb3b0780f2877…` | 2 | 2 |
| `0xbdf221228d7c…` | 2 | 2 |

**Six addresses, each charged on 100% of its maker legs. Addresses with both a
charged and a zero maker leg: zero.** Charged-ness is not a per-fill trigger at
all — it is an account attribute, all-or-nothing. That is why 25 maker BUYs at
the same price in the same block buckets paid nothing: they belong to accounts
in the other class.

So the residual is not "0.95% unexplained". It is **"six accounts out of the
corpus are in a charged class, and every fill they make is charged."** That is a
rule, and it is in the data DA already holds.

**The one question that converts the hope into a rule** — and DA can answer it
tonight from this same file: **is our own maker address in this corpus, and
which class is it in?** If it appears with zero-fee maker legs, the qualified
zero is supported *for us specifically* and the declaration can say so. If it
does not appear at all, then 1,046 zeros are other people's accounts and we have
no observation of our own treatment — which is a materially different receipt.

## 2.2 The sensitivity does not bound the exposure. Two reasons, both measured.

**(a) There are two rate tiers, and DE's 10% is the lower one.** All ten charged
legs, with the implied rate computed as `fee_per_share / min(p, 1-p)`:

| price | size | fee_usdc | cents/share | implied rate | maker |
|---|---|---|---|---|---|
| 0.99 | 31.73 | 0.03141 | 0.09899 | **0.0990** | `0xb3b0780f` |
| 0.99 | 128.0 | 0.12672 | 0.09900 | **0.0990** | `0x2277c18f` |
| 0.99 | 128.0 | 0.12672 | 0.09900 | **0.0990** | `0x2277c18f` |
| 0.99 | 9.87 | 0.00977 | 0.09899 | **0.0990** | `0x2277c18f` |
| 0.99 | 10.22 | 0.01011 | 0.09892 | **0.0989** | `0x8d009282` |
| 0.99 | 77.21 | 0.07643 | 0.09899 | **0.0990** | `0xb3b0780f` |
| 0.99 | 3.70 | 0.00366 | 0.09892 | **0.0989** | `0x18b0b710` |
| 0.99 | 19.99 | 0.09895 | 0.49500 | **0.4950** | `0xbdf22122` |
| 0.99 | 100.0 | 0.49500 | 0.49500 | **0.4950** | `0xbdf22122` |
| 0.99 | 14.00 | 0.06930 | 0.49500 | **0.4950** | `0x0fd0ebb1` |

Seven legs at **9.90%**, three at **49.50%** — and the tier splits by address
too. A worst-case that charges 10% is **five times below the worst rate already
observed**. "10%" is not the cap; it is the modal tier.

**(b) Every observation sits at the cheapest point of the schedule.** All ten are
at price **0.9900**, where `min(p, 1-p) = 0.01` is at its minimum. The same rate
at the prices this strategy actually quotes:

| price | 9.9% rate | vs the observed 0.099 c/share |
|---|---|---|
| 0.99 | 0.099 c/share | 1× |
| 0.90 | 0.990 c/share | 10× |
| 0.75 | 2.475 c/share | 25× |
| **0.50** | **4.950 c/share** | **50×** |

At the 49.5% tier and p = 0.50 that is **24.75 cents/share — 250× the observed
magnitude.** So "charge every maker fill 10%" bounds nothing unless it means
*10% of `min(p, 1-p)` evaluated at each fill's own price*, and even then it is
5× short of the observed worst tier. Whether the sensitivity is a bound or a
rounding error depends entirely on which of those DE implements, and the phrase
as routed does not say.

**(c) The formula used to extrapolate is not validated where fees are certain.**
Taker legs are 901/901 charged — the one place the fee schedule is fully
observable — and the audit reports `taker_formula_match_share = 0.1221` with a
maximum residual of **0.51 USDC**. The model reproduces 12% of the fees it can
check against. Extrapolating maker exposure with it is extrapolating with an
instrument that fails its own positive control.

## 2.3 One objection of mine, withdrawn

Before reading the data I expected the worst case to be **asymmetric** — an
unidentified trigger correlating with something the two legs do differently,
so that charging both legs equally would understate the *difference*, which is
the estimand. **The account partition refutes that.** Both legs are the same
account, so an account-level fee applies to both identically and the symmetric
all-fills shape is correct. That part of the safeguard is right, and I would
have been wrong to insist otherwise.

## 2.4 Does it let a freeze go effective on an assumption?

On the narrow rule-12 question, **DA's reasoning is sound**: rule 12 requires
the builder committed, the pipeline in-repo, the nulls declared and the
multiplicity recorded. A fee rule declared as "zero, with these ten exceptions
recorded and a sensitivity attached" is a recorded decision, not an absence, and
§7 wants it written down rather than left open. Holding for a publication that
may never come is the worse failure.

**But the decision does not have to be made tonight at all**, and this is the
part I would put in front of the user. `fee_rule` is an **economic-only** gap —
REVIEW 245 §2, unchanged: of the fifteen original gaps, thirteen including
`fee_rule` and `pnl` bear only on §9. **§8's log loss never touches a fee.** So:

- a *predictive* freeze can go effective today with `fee_rule` unresolved, and
  no assumption enters §8;
- the fee ruling gates §9, which is ten evaluable days away;
- resolving §2.1's one question — our own address's class — costs a query
  against a file that already exists, not a publication.

**My recommendation: do not adopt the qualified zero tonight.** Answer the
address question first; it is cheap, it is available, and it turns the
declaration from "a rule with ten unexplained counterexamples" into "a rule with
a named class and our position in it". If the answer is unfavourable or our
address is absent, the sensitivity has to be rebuilt at 49.5% of `min(p, 1-p)`
per fill anyway, and better now than after ten days are consumed.

---

# PART 1 — THE SUPERSET CLAIM AND §7k

## 1.1 The superset holds, and it holds harder than you claimed

Tested as sets, at `origin/mm-research` (404 review files, state round **401** —
one higher than your read, it moved again — `STATUS.yml` 60,251 lines, 919
R-entries):

    worktree − origin = 0      backup − origin = 0
    union of all three = 404 = origin          ✓

**And I tested the axis your set test does not cover: content.** For every
review file present in both the worktree and `origin/mm-research`, I compared
git blob hashes: **0 files differ.** So it is a byte-level superset, not merely
a name-level one. I could not defeat the claim.

## 1.2 Where I can defeat it: scope

The canonical ref is canonical for the **narrative** half and not for the
**evidentiary** half.

| | `origin/mm-research` | `origin/de-freeze-chain-v2` |
|---|---|---|
| declarations under `live/pm_research/declarations/` | **204** | **216** |

**Fourteen declarations exist on the chain and not at the canonical ref**,
including all seven `da_fair_value_progress_ledger_v1..v7.json`. A reader sent
to the canonical location cannot open the progress-ledger declarations that the
six-of-six ruling is written against, nor the freeze declaration's supersession
chain. So "a canonical place now exists" is true for reviews, state and the
register, and false for the artifacts those documents cite as evidence.

That is the two-rule split from REVIEW 248 §1.2 with numbers on it, and it is
the thing §7k should say: **one canonical location for the record, the executing
refs for the evidence, and a statement of which a given claim was read against.**

## 1.3 One improvement to the rule itself

§7k's set test is right and is the reason I could not defeat the claim. But a
set of **names** passes over a same-name content change — a force-push that
edits a landed review keeps the name set identical. Make it a set of
**(name, blob)** pairs, which is the test I ran in §1.1 and which costs one
extra field in `git ls-tree`.

# PART 3 — the cited-but-absent reviews

Checked at the canonical ref rather than assumed. Of the 18 numbers
`COORDINATION.md` cites without a file beside it in the shared tree, **all ten
of mine — 206, 208, 213, 221, 229, 230, 232, 235, 236, 246 — are present at
`origin/mm-research`, 10 of 10.** They were never missing from the canonical
ref; they were missing from the shared tree, and naming `origin/mm-research`
canonical resolves every one of them without my doing anything.

The remaining eight — 86, 155, 158, 162, 163, 164, 166, 167 — are **absent at
the canonical ref too**, count 0 each. They are not mine and they are cited by a
register that cannot resolve them. Someone should establish whether those
reviews were ever files.

## 4. Owed

- REVIEW 247 Part 1 still awaits a resolver; REVIEW 246 and 245 items remain
  open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs, now at the canonical
  ref and stated as such.
