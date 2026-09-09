# REVIEW 146 — DA 168's proof that the rebuild is unnecessary

**REV, 2026-09-09T15:44Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

**VERDICT: THE PROOF REPRODUCES INDEPENDENTLY AND HOLDS ON ALL FOUR. I derived the reachable
set myself and got `{ruled_day_set}`; I enumerated what moved BY OPERATION rather than from
the commit list and got DA's 8 functions and 11 constants exactly, with an EMPTY intersection;
`ruled_day_set` is `41988ef4272c73a9` in both versions, DA's own digit. AND ONE FACT DA'S
FRAMING UNDERSELLS: `de_phase4_diag_runner.py` — the module the entry points LIVE IN — IS
BYTE-IDENTICAL between the book and disk. The scoring module has not moved at all. The
payload DA proposes computes in 0.37 s. ONE THING THE WAIVER MUST CARRY EXPLICITLY: the
book-code predicate REFUSES on this book today, by name, and a waiver is a decision to
override a check that is firing CORRECTLY.**

---

## (1) IS THE PATH REALLY ONE FUNCTION? — DERIVED, NOT READ

My own AST walk from the two scoring entry points (`assemble_streaming`, `build_tape_index`),
following calls and attribute reads, closing transitively:

```
de_phase4_diag_runner functions reachable from the two entry points : 18
attributes of de_multiday_gate1_runner reached                      : ['ruled_day_set']
transitive closure INSIDE that module                               : {'ruled_day_set'}
module constants read on that path                                  : ['PARAMS_REL']
non-function attributes reached                                     : none
```

**And the alias assumption my method rests on is verified at the source**, which is where it
could have failed: in all 16k lines of `de_phase4_diag_runner` there is **exactly one import
of that module — line 7519, LAZY, function-local, aliased `_G` — with exactly one use at line
7520, `_G.ruled_day_set()`, and no `getattr` on it** (every `getattr` hit in the file is
unrelated AST-walking code).

**A fact that strengthens the waiver and is not in DA's headline:**

```
de_phase4_diag_runner.py   book cb97b94dbd3fc6ca   disk cb97b94dbd3fc6ca   IDENTICAL
de_multiday_gate1_runner.py book b4532d00c9ac6c19   disk a76e81c3391e0779   MOVED
```

**The module that owns the scoring entry points has not moved.** The whole question is what,
inside the one module that did move, is reachable — and the answer is one 8-line function.

## (2) IS THE MOVED SET COMPLETE? — ENUMERATED BY OPERATION

I did not take DA's commit list. I found **the book's own recorded version** by scanning
history for the receipt's sha256 (commit `69123fddb`) and hashed **every top-level
`def`/`class`/assignment** in both versions:

```
MOVED: 8 functions, 11 constants, 0 classes        <- DA's count, reproduced
  functions: _main_day, aggregate, day_decision_population, main,
             ruled_lazy_exemption, run_day, selftest, test_statistic_from
  constants: DAY_SUPERSEDES_NOTHING, DAY_SUPERSEDES_NOT_THE_HEAD,
             DAY_SUPERSEDES_TARGET_ABSENT, EXPECTED_CHECKS, LAZY_ONLY_EXEMPTION,
             RULED_LAZY_EXEMPT, RULED_LAZY_SOURCE, STATISTIC_BLOCKS,
             THETA_DRIFT, VERDICT_ENDPOINT, _ARM_DAY_VALUED_STATUSES

THE SCORING PATH   : {def ruled_day_set, const PARAMS_REL}
INTERSECTION       : EMPTY
ruled_day_set      : 41988ef4272c73a9 -> 41988ef4272c73a9   (8 lines)
PARAMS_REL         : 9f0266f5a6175372 -> 9f0266f5a6175372
```

**This is the spelling-versus-operation hazard closed**: the set comes from hashing every unit
of both files, so a change no commit message mentioned would still appear.

## (3) THE TWO THAT LOOK ALARMING — DRIVEN

**`day_decision_population`: 7 lines added, 0 removed.** Six are comment lines; one is
`"theta_drift": THETA_DRIFT.get(arm, …)`. `decisions`, `by_side` and every computed quantity
are untouched.

**And the "appended field that something iterates" risk, swept at the consumers:**

- `de_multiday_gate1_runner:16613` — `for k, v in payload["decision_populations"].items()`
  iterates **arms**, not the inner fields.
- `da_gate1_day_verdict:5130` — iterates `pops.items()` (**arms**) and then reads **named
  fields**, `blk.get("head")` and `blk.get("theta")`. **It does not iterate the inner dict, so
  an appended field is inert there.**

**The multi-day repoint** is `aggregate`, which the empty intersection already places off the
single-day scoring path — cross-day, as DA said.

## (4) IS THE PAYLOAD COMPUTABLE IN SECONDS? — MEASURED

```
reachable-set derivation                                    0.37 s   80 MB
moved-set derivation, INCLUDING scanning 40 commits for
the book's recorded blob                                    0.40 s   70 MB
```

**Yes, with margin.** Both halves of DA's proposed refusal payload — the reachable closure
intersected with what changed, and the byte-identity of the functions on the path — are
sub-second on a 16,205-line module. **"Do not narrow the condition, widen the payload" is
cheap enough to be the right answer**, and it makes the rebuild-versus-supersede decision
readable from the refusal instead of requiring a round like this one.

## THE THING THE WAIVER MUST CARRY EXPLICITLY

```
assert_book_scoring_code on the EV21 receipt, right now
  -> REFUSED BOOK_BUILT_BY_DIFFERENT_SCORING_CODE
     naming: ['de_multiday_gate1_runner.py']
```

**The predicate is firing, correctly, on exactly the module this proof exonerates.** A waiver
is therefore not "the check agrees" — it is a decision to **override a correct refusal on the
strength of a finer-grained argument the check does not make**. That is legitimate and it is
the user's call, but it must be recorded as an override with this proof attached, not as a
pass. **This is precisely why DA's widened payload matters: with it, the refusal would carry
its own exoneration and no override would be needed.**

## SCOPE

My reachability follows **calls and attribute reads** through the AST. It cannot see a
dynamic `getattr(_G, name)`, a module reached through a variable, or an `importlib`
indirection — **I checked the first two at the source for this module pair and there are
none.** The claim is about **this book's SCORES**; it says nothing about the receipt's
provenance being accurate, which is DA's own "buys provenance, not correctness".

## ROUTED

1. **Coordinator / user — the proof holds.** Independently derived, and stronger than stated:
   the scoring module itself is byte-identical.
2. **Record the waiver as an OVERRIDE of a firing refusal**, with this proof attached — not
   as a pass.
3. **DA — the widened payload is worth building: 0.37 s.** It would have made this round
   unnecessary.
