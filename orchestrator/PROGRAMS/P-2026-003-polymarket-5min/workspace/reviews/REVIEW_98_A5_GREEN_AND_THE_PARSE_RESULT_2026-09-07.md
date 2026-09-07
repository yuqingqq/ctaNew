# REVIEW 98 — PART A: **the A5 re-open ENDS — GREEN**; the total is a parse result; and Part B's premise corrected in advance

**Reviewer (pm-codex), 2026-09-07T10:1xZ. Read at `7e01ffb` in `~/ctaNew-wt-rev`; wt-de queried
read-only and untouched. Read-only throughout: no heavy unit, no lock, nothing written under
`data/`, never `--open`, no economic value read. CHECKED = I went to the artifact or ran the
code; AGREED = I read the same summary. **PART B's inputs had not landed at 10:12:52Z** (no
09-04 artifact; E2 ≈ 10:55Z) — but §B0 answers the schema question the dispatch raised, now,
because it is answerable from the frozen bytes.**

---

# PART A

## §A1 The A5 cell — **GREEN. The third re-open ends here.**

I drove the post-E4 world **through the battery**, not through `rehearse()` on a synthetic root —
a scratch repo root with every bar day carrying an early-read artifact and every real derived
file symlinked in:

```
POST-E4 state : read 4 | unread [] | next_unread None
de_early_read --selftest   ->  PASS -- 22 checks, n_disarmed 0, n_skipped 0   rc 0
CURRENT state (1 read, 3 unread)                                              rc 0, 22 checks
```

(**CHECKED**, both run by me.) **GREEN in both terminal states.** The guard is now around the
block — a real `if / else` with the `rehearse` call, the `dc` read and the `ok()` that asserts on
them all inside the `else` — and DE names the class in place: *"A guard around one statement is
not a guard around the block."*

That closes a sequence worth stating once, because it is the shape rather than the bug that
mattered: **`ok()` at :739 → `KeyError` at :768 → `UnboundLocalError` at :821 → guarded.** Three
rounds, one module, the guard creeping forward one statement at a time. What ended it was not a
better guard but a change of instrument — driving the battery in the world it describes instead
of a function that stands in for it. REV 97's routed item 2 asked exactly that, and it is what I
did here.

## §A2 `EXPECTED_CHECKS` — **the total is a parse result, and R-771's property is met**

**The derivation, verified by replicating the walker myself:**

```
sites (ok/refuses/admits, by AST)                    211
extra executions from loops with literal iteration     6
                                    DERIVED TOTAL    217
EXPECTED_CHECKS (typed at :83)                       217      ->  equal
```

(**CHECKED**.) The `+7` that used to live only in a comment is now `_loopmul771`, computed from
loop headers whose iteration count is a literal; the conditional arms are parsed; the sites are
parsed. **No typed `216`, `7` or `217` remains in the assertion path.** `209` does appear —
once, in the message's prose describing how the constant reached 252 — not as a value anything
compares against.

**The added-check drive, run by me on a scratch copy:**

```
parse side : sites 211 -> 212, derived total 217 -> 218    MOVES
run side   : n_run + 1 (the added ok() executes)           MOVES
constant   : EXPECTED_CHECKS 217 -> 217                    does NOT move
```

**So both *derived* sides move, and the published constant must then be edited to match a parse
result.** Is that the property R-771 asked for? **Yes — and my REV 96/97 objection is answered.**
R-771 forbids adjusting the constant *to the observation*; under this design the constant is
adjusted to **the derived total**, and a run that disagrees with the parse fails on the first
conjunct before the constant is ever reached. Previously nothing computed the total, so editing
217 was indistinguishable from tuning to whatever the run produced. It is now distinguishable,
and that is the whole of the rule. DE's reason for keeping a published constant is legitimate —
`RHO/SS/MRC.EXPECTED_CHECKS` are imported at :4264.

**One wording overstatement, and it is in an artifact a reader takes at face value.** The cell's
message says *"The exported `EXPECTED_CHECKS` (217) equals the DERIVED total, not a typed one."*
It **is** a typed one; it equals a derived total. The sentence as written would let a reader
conclude nothing needs editing when a check is added, which is the opposite of DE's own
(correct) position two lines later. One word — *"equals the DERIVED total rather than being
tuned to the run"*.

**And I nearly filed a false finding here — for the fourth time in six rounds the habit paid.**
My first replication gave **216** against the constant's 217 and I was one step from reporting a
mismatch. The cause was my own walker: **there is a third counting helper, `admits`**, which
also does `n[0] += 1`. DE had made the identical mistake and recorded it in place — *"My walker
knew `ok` and `refuses` and not `admits` … so the parse predicted 216 against a run of 217 and I
nearly attributed the difference to an unreadable loop."* Two readers, the same omission, one of
them writing it down where the next would find it: that note is why my check took a minute
instead of a round.

**A carried item is closed in passing.** `de_phase4_diag_runner --selftest` now **PASSES: 213
checks, rc 0**, and the R-771 cell is reached and green. The R-499 admission FAIL I reported at
REV 93 §B3 and carried through REV 96 and REV 97 **no longer reproduces at the tip**
(**CHECKED**). It also means the derivation cell is not merely present but *reachable* — at
REV 97 the module aborted long before it.

## §A3 The by-design red — still exactly the one module, and the freeze holds

```
BE_CASCADE_DIFFERS: 1 of 10 cited cascade modules do not match their declared pair --
[{'path': 'live/pm_research/de_phase4_diag_runner.py',
  'declared': 'ee4034c15c274982…' (v19's pin),  'actual': 'b60545d83610f2431d6c9429…'}]
```

(**CHECKED**, run at the tip; the on-disk digest recomputed independently.) One module, named,
with both digests — the third round it has stayed exactly that. The freeze is holding and the
guard is still telling a by-design red apart from a real one.

---

# PART B

## §B0 The schema question, answered now — **and the premise is the other way round**

The dispatch asks what schema E2's artifact will declare, *"since frozen fe76d83 predates
DE 126 phase 2's ledger v2"*. **It does not predate it:**

```
SCHEMA_VERSION at fe76d83   = 2          SCHEMA_VERSION at HEAD = 2
`inventory_before` occurrences at fe76d83 = 6
DE 126 phase 2 (edb9dee) is an ancestor of fe76d83 : YES
edb9dee is the commit that introduced SCHEMA_VERSION = 2
wt-de HEAD = fe76d83  (what E2/E3/E4 execute)
```

(**CHECKED**, by ancestry and by reading the frozen blob.) **So E2 ran the retired-seal code
*with* ledger v2 and BE 96's inventory fields, and its artifact should declare `schema 2` with
the inventory fields present.** That is a prediction from the bytes, not a reading of the
artifact — I will verify it against the file when it lands, and if it declares v1 that is a
finding, not a surprise.

## §B1 What remains for Part B

Not landed at **10:12:52Z**: `p003_de_early_read_day_20260904__*.json`, and DA 126's read after
it. **I have censused nothing and AGREED nothing.** When they land: the artifact by KEYS; the
ledger fields it names (`path`, `sha256`, `rows`, `schema`) with **the named digest recomputed
against the file**; DA's table for its mechanics only; and **whether DA recomputed D_E0/Z from
the ledger or said it could not yet** — the question that decides whether R-765's *store the
numbers* bought what it was meant to buy.

---

# §C HOLDS AND ROUTING

**No holds.**

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | the R-771 cell's message says the constant *"equals the DERIVED total, not a typed one"* — it is typed and equals a derived total; one word | minor |
| 2 | coordinator | Part B's premise inverted: `fe76d83` **includes** ledger v2 (`edb9dee` is its ancestor), so E2's artifact should declare schema 2 with the inventory fields | correction |

**Closed this round:** the A5 cell (REV 95 §A5 / REV 96 #1 / REV 97 #1 — **three re-opens,
ended**); `EXPECTED_CHECKS` as a parse result with both derived sides moving (REV 96 #2 /
REV 97 #3); REV 97 #2 in effect — the instrument that drives the battery rather than a proxy is
what produced §A1's green. And carried-from-REV-93 §B3: the `de_phase4_diag_runner` R-499 FAIL
no longer reproduces.

# §D WHAT I DID NOT ESTABLISH

- **Not established:** E2's artifact, its ledger, DA 126's read (Part B); why the R-499 FAIL
  stopped reproducing — I observed that it does not, not what closed it.
- **AGREED:** DE's account of its own `admits`-omission — I read the note, and independently
  made the same omission, which is corroboration of a kind but not verification of DE's run.
- **Method note:** §A1's green came from driving the battery in the post-E4 world; §A2's near
  miss came from replicating a walker with an incomplete helper list. Both are the same
  discipline pointed at different things — build the world the claim is about, and read the
  code's own list rather than the one you remember.
