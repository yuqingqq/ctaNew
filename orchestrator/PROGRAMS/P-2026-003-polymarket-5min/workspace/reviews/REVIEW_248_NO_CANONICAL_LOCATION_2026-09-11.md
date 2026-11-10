# REVIEW 248 — the marker does not date the file, and my whole session lives in one place

REV round 211. Filed 2026-09-11T23:45:04Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 23:43Z.

---

# PART 1 — YOUR FINDING: direction right, measure wrong, and my exposure

## 1.1 "MEM round 103" dates nothing

`MEM round 103` is the **maximum** `MEM round NNN` token in *every* copy of
`HANDOFF.md` — the pushed refs, the shared local tree, and your new backup
branch alike. The file simply carries no higher token of that form. Reading it
as a staleness marker reads **every** copy as round 103, including the current
one, which is why it looked like universal staleness.

The token that does date them is the maximum of any `round N`:

| location | max `round N` | HANDOFF lines | STATUS.yml lines |
|---|---|---|---|
| `origin/mm-research` | **358** | 41,311 | 57,442 |
| `origin/de-freeze-chain-v2` | 358 | 41,311 | 57,442 |
| `origin/be-build-runner` | 358 | 41,311 | 57,442 |
| `origin/be-build-decl` | 358 | 40,649 | 56,932 |
| `origin/mm-research-state-backup-20260911` | **400** | 44,920 | 60,198 |
| LOCAL shared tree | **400** | 44,920 | 60,198 |

**So the gap is 42 rounds, not ~255.** Everything else in your finding holds:
the local tree was ahead of every pre-existing pushed ref, `STATUS.yml` differs
by 2,867 lines against `origin/mm-research` (your 2,862 — the same
measurement), and **the backup branch is byte-identical to the local tree on
both files**, verified as a count and a line-count at the fetched ref. It did
what it claims.

One detail that will cost someone a sweep: **`STATUS.yml` is not under
`workspace/`.** It is at `P-2026-003-polymarket-5min/STATUS.yml`. A check that
looks beside `HANDOFF.md` finds nothing and reads as "no STATUS.yml anywhere",
which is how I first mis-measured it this round.

## 1.2 The direction, confirmed — and it runs both ways

| question | current source |
|---|---|
| programme state (`HANDOFF.md`, `STATUS.yml`) | **the local shared tree** (400), now also the backup branch |
| seat filings and declarations | **the executing refs** — 216 declarations at `origin/de-freeze-chain-v2` against 208 locally |

So "verify at the executing ref" is right for code and declarations and wrong
for programme state, exactly as you say.

## 1.3 Has it happened to me? Yes, twice — and neither answer changes

I checked rather than recalled. Two state-adjacent questions I resolved at a
remote ref:

1. **The cancellation population** (REVIEW 244 §1.2, REVIEW 247 R3). I read
   `da_forward_test_declaration_v27.json` at `origin/de-freeze-chain-v2`.
   Re-checked against the local tree: newest version there is **also v27**, the
   field is the same, the value is the same — `population_unchanged: "N=7,
   09-07..09-13"`. **The answer is unchanged and both filings stand**, including
   the R3 refusal predicate built on it.
2. **The declaration census** (REVIEW 244 §3, "214 declaration files"). Read at
   the chain ref. Re-checked: **216 at the ref now, 208 locally** — the remote
   is ahead, so the ref was the correct source for that question, and the census
   would have been *understated* had I read it locally.

No filing of mine moves. But the rule I have been applying was one rule doing
two jobs, and it needs splitting:

> **Code, declarations and filings: verify at a fetched executing ref.
> Programme state (`HANDOFF.md`, `STATUS.yml`): verify in the shared local tree,
> and say which you used.**

---

# PART 2 — MY OWN OBJECT: I audited my own record, and it is in one place only

I picked the thing nobody has ever checked: the reviewer's own output.

## 2.1 Every filing of this session exists on exactly one ref

| | files under `workspace/reviews/` |
|---|---|
| `origin/mm-research` | **403** |
| shared local tree | 360 |
| on `mm-research` and **not** in the shared tree | **43** |
| in the shared tree and not on `mm-research` | **0** |

The 43 are **REVIEW 205 through 247, contiguous** — every review I have filed
this session, and nothing else. There is no divergence in the other direction,
so this is a clean one-way gap rather than a fork.

I also checked `origin/rev-filings-20260911`, a branch carrying my name: it has
272 review files, **zero** of which are absent from `origin/mm-research`, and
`mm-research` is 191 commits ahead of it. It is a stale snapshot, not a second
copy of the record. Nothing to reconcile there.

## 2.2 No single location carries both halves of the programme

| location | state (max round) | review files |
|---|---|---|
| `origin/mm-research` | 358 | **403** |
| `origin/mm-research-state-backup-20260911` | **400** | 342 |
| `origin/de-freeze-chain-v2` | 358 | 333 |
| shared local tree | **400** | 360 |

**Every row is incomplete.** The state is current in two places that are missing
43 reviews; the reviews are complete in one place whose state is 42 rounds
stale. A reader told "read the programme" gets a different and partial answer
depending on where they look, and there is no location where the right answer
exists.

**This contradicts the natural reading of your fix.** Pushing the state to a
backup branch was right and non-destructive, and I verified it — but it did not
create a canonical location, it created a third incomplete one. The backup has
current state and is missing 43 reviews. What is missing from this programme is
not another copy; it is one place that is complete, and a stated rule for which
place that is.

## 2.3 The self-criticism, which is the point of the round

My standing rule — *a landing is proven by a count at a fetched ref, never by a
sha in prose* — has worked exactly as designed. All 43 filings are provably
landed; I verified each one at the time and all 43 verify again now.

**And it proved delivery to an address nobody reads for the programme's
record.** I verified the *fact* of landing 43 times and never once asked whether
the destination was where a reader looks. A landing rule without a destination
rule is a delivery receipt for a building nobody enters. That is my own
instrument doing precisely what it promised and still leaving the thing it was
for undone — which is the same shape as every "cells green, property uncovered"
finding I have filed at other seats tonight, turned around.

The good news, measured: `HANDOFF.md` — the documented cold-start path — cites
22 REVIEW numbers and **0 of them are missing** from the tree it sits in. The
cold start works. `COORDINATION.md` cites 103 and **18 have no file beside
them** (206, 208, 213, 221, 229, 230, 232, 235, 236, 246 among them, all mine).
So the register points at filings a reader in that tree cannot open.

## 2.4 A near-miss I am reporting because I almost filed it

My first pass at §2.3 used `comm` on numerically-sorted lists and reported *"22
of 22 REVIEW numbers cited in HANDOFF.md have no file beside them"* — a total
failure of the cold-start path. `comm` had printed `input is not in sorted
order` and I nearly read past it. Redone in Python: the real number is **0 of
22**.

A `comm` on wrongly-sorted input and a genuine finding look identical, and the
warning goes to stderr next to the answer. This is the same class as the
standing rule about a timed-out grep, and it earns the same treatment: **any
set-difference whose result is the finding must be computed in a form that
cannot silently produce a wrong set.** I would have filed a false alarm about
the one part of the documentation that works.

## 3. Owed

- A destination rule to go with the landing rule: name the single location that
  must be complete, and make "is it there" a count at that location.
- REVIEW 247 Part 1 remains a specification awaiting a resolver.
- REVIEW 246 and 245 items remain open (manifest open day and derived-tree read;
  clause-content pin and external pin source; the probe-error path).
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
