# Review — DE round 17 (DE15-R1..R4 closed: the declared limit says what the predicate does)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `a8093a5`** (Q-DE-35 row at `451f7fc`).
**Request of record:** `REQUEST_DE_ROUND_17_2026-09-02.md`.
**Composed 2026-09-02T13:37:16Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach a8093a5`. Read-only under `data/`; every mutant
applied to the worktree copy and restored; `COORDINATION.md` never written. No timer, no
service, no launcher. **DE round 18 (`db039a3`) was not fetched, read or executed** — this
review is at its own tip only.

Scope confirmed: `de_admissible_windows.py` **+160/−27** and the one declared exception in
`de_ratification_check.py` **+21/−7** (= +181/−34); the other ten DE-family files are
**byte-identical** to `829910e` by blob hash. Suites at this tip: admissible **75**,
ratification **132**, seam **69**, rc 0 under both launchers.

---

## Verdict

### RELEASED. Every mutant I could think of dies by name, including the one that left 104 green two rounds ago.

Two findings, both LOW, and both about *association* rather than existence: the entry→assertion
map is keyed by list position and nothing binds a position to its entry, so **reordering the
list leaves the suite green at 75** while the map claims the wrong assertions cover the wrong
entries; and the OVER-CAUGHT paragraph is the one statement in the block that still has no check
behind it.

---

## 1. DE15-R1 closed — membership is asserted, and the sentence says only what is evaluated

| mutant | suite |
|---|---|
| a **fifth entry** with no assertion key | **FAIL** — *"THE LIMIT IS DECLARED AND ITS MEMBERSHIP IS ASSERTED: every one of the **5** entries is keyed…"* |
| a fifth entry **with** a key but no loop row | **FAIL**, same check |
| the `builtins.eval` row **deleted from the loop** | **FAIL** — *"THE MAP IS ASSERTED AGAINST THE LOOP THAT RAN … the **5** expected-blind rows executed above"* |

The message interpolates the list itself — `[x.split(' (')[0] for x in DECLARED_BLIND_SHAPES]`
— so what is printed is what was read. Nothing is paraphrased.

**The docstring, read as someone who has never seen the review.** The block at `:138-201` now
runs REFUSED → DECLARED BLIND → OVER-CAUGHT → NOT BLIND → the closing statement, and each
section says what the code does:

- `:143` *"exec / eval / compile … a BARE-NAME call, argument unread"* — the false "any use at
  all" is gone, and the attribute form is sent to the blind list in the same breath.
- the blind entries each carry their reason, and entry 3 carries **NOT ASSERTABLE IN-PROCESS**
  as a stated decision.
- the NOT-BLIND paragraph sits **below** the entries (`:183`), where it cannot read as one.
- the closing paragraph no longer says testing is impossible; it says the statement is checked,
  names the both-directions property, and carries the disposition ("delete the entry and its
  assertion together … the blindness is never restored to keep the suite green").

Grepped for the four stale claims: `does NOT see` **0**, `no way to TEST` **0**, `any use at
all` **0**. `builtins.__import__` survives five times and every one is in the CAUGHT sense or as
recorded history. **I found no sentence in the block that a check does not evaluate** — except
the OVER-CAUGHT paragraph, DE17-R2.

## 2. DE15-R2 closed — three rows for the three shapes of entry 1

The loop is six rows (`:1000-1018`): `runpy.run_path`, `runpy.run_module`, `builtins.exec`,
**`builtins.eval`**, **`builtins.compile`**, `getattr`-reached `import_module`. Both directions,
driven:

| direction | result |
|---|---|
| `builtins.eval` **starts catching** (`eval` added to `DYNAMIC_IMPORT_CALLS`) | **FAIL** — *"EXPECTED-BLIND (builtins.eval (attribute form)): the predicate still sees exactly ['builtins'] …"* |
| `builtins.compile` **starts refusing** (an injected raise) | **red, and NAMED**: *"EXPECTED-BLIND (builtins.compile (attribute form)) started REFUSING: MUTANT…"* |

The refusing direction now names the row — the `_label` re-raise at `:1027`. That was the nuance
I filed in round 15 as a one-line closure; it is closed exactly there.

Entry 3 is annotated NOT ASSERTABLE IN-PROCESS at the list and checked at `:1109`
(`"C extensions" in DECLARED_BLIND_SHAPES[next(iter(_unasserted))]`), so the empty row is a
decision rather than an omission — and, as it happens, that is the only check in the map that is
position-aware (DE17-R1).

## 3. DE15-R3 closed — the over-catch is declared, checked, and the direction claim holds

The OVER-CAUGHT paragraph (`:171-181`) states the reach with measured examples, and two checks
carry it: a user class whose method is named `__import__` contributes `{'not_a_module'}`, and
`os.environ.__import__('da_forward_day_verify')` is **CAUGHT** (`reads_no_verdict` → False).

**The direction question, answered: yes, it can refuse a file that imports no verdict producer
— and that is the safe side.** `obj.__import__('da_forward_day_verify')` is exactly such a file:
the literal is a method argument, nothing is imported, and the predicate answers False. What
would the seam do with it? The only consumer of `reads_no_verdict` outside this module is
`ev_replay_seam.py:1484`, a selftest assertion that the seam's own source reads no verdict
producer. So the consequence of a false catch is **a suite going red with a named message** — a
human reads it and finds the answer written at the list. A false *admission* would instead let a
verdict producer through a supplier that must not decide (rule 14). The asymmetry is the right
way round, and the module says so at the list rather than in a review.

## 4. DE15-R4 closed — the binding phrase, and no check traded away

`check.__doc__` and the emission's `stamp_fields` are each asserted as **one string** binding the
field to its meaning (`de_ratification_check.py:1151` and `:1158`), not two tokens. The swap
mutant — `stamped_at_raw` declared the CANONICAL PARSE, both tokens still present — **dies**:
*"the emission's stamp fields are DOCUMENTED in check()'s docstring as BINDING PHRASES"*. The
same mutant left the suite green at 104 when I ran it at `0ca510e`.

**Nothing was removed to keep the count at 132.** AST call-site census at both tips:
`829910e` = `{ok: 67, refuses: 39, refuses_nv: 4}` with tuple-loop lengths
`[2,2,2,3,3,3,3,3,3,3,6]`; `a8093a5` = **identical on every figure**. The +21/−7 strengthened two
existing checks and added none.

## 5. The judgement clause — structural, not only a message

Both halves are enforced, and I drove both: an entry added **without** its assertion goes red
(mutant 1a), and an assertion row deleted **without** its entry goes red (mutant 2, at the
map-vs-loop check). So "the entry and its check live and die together" is a property of the
code, and the clause in the message tells a maintainer what the red means rather than standing
in for the check. **The caveat is DE17-R1:** the binding is by position, so it enforces existence
in both directions and association in neither.

## 6. Deltas, two nets, and nothing moved

69 → 75 is six checks: two new loop rows (eval, compile), the two over-catch checks, and the two
membership checks. Emptying the whole expected-blind loop fires the **map** assertion first
(*"the 0 expected-blind rows executed above"*), not the count — they are two detectors of
different things, and only the earlier one is observable in a given run. That the count is a
genuinely independent second net I confirmed separately: deleting a check **outside** the loop
gives `FAIL: check count asserted at run time: 74 == 75` with the map still green.

| | |
|---|---|
| R-419 on 09-01 | `verified_for_new_run True`, `unverifiable []` |
| audit | 25 cases / 19 sites, survivors `[]`, `coverage_matches_expected True` |
| seam | **1,875** specs; `daw is de_admissible_windows` **True** |
| closure self-predicate | all three files `reads_no_verdict True` |
| `decides` | *"nothing -- this reports; admission is the coordinator's act…"* |

## 7. Rule 10 / rule 14, and the citations

Of the **5** `ok()`/`refuses()` call sites the round added, **4** interpolate what they saw; the
one that does not (`:1064`) states the direction claim whose values are in the assertion but not
printed — a reporting nicety, not a rule-10 breach, and a large improvement on the 7-of-20 I
measured last round.

**Citations spot-checked against `a8093a5`, three from the Q-DE-35 row and five from the
request** — every one exact: `:183` (the NOT-BLIND paragraph, below the entries), `:1027` (the
`_label` re-raise), `de_ratification_check.py:1151`/`:1158` (the binding phrases), and
`:203-210` / `:223` / `:1101` / `:1117` / `:143`. DE's Q-DE-33 filing rule — anchors asserted,
file re-read, line numbers from the commit — holds at every point I tested it.

---

## Findings

### DE17-R1 — LOW — the entry→assertion map is keyed by position, so a REORDER is invisible

`BLIND_ENTRY_ASSERTIONS` (`:223`) is `{0: (...), 1: (...), 2: (...), 3: ()}` keyed by index into
`DECLARED_BLIND_SHAPES`, and the three membership checks test only *shape*: every index is
keyed, `_covered == {0,1,2}`, `_unasserted == {3}`, and the label set equals the loop's. None of
them looks at **which entry** an index holds.

Reproduced: I swapped entries 0 and 2 (`runpy…` ↔ `getattr(importlib, "import_module")(...)`)
and left the map untouched, so it now claims the runpy assertions cover the getattr entry and
the getattr assertion covers runpy. The suite reported **`selftest OK -- 75 checks`, rc 0**.

The comment at `:212-222` states the intent — *"Keyed by the entry's INDEX … so an entry and the
assertions that cover it live and die together"* — and that holds for existence (both mutants
above go red) but not for association, which is what "cover" means. The one entry that *is*
pinned is entry 3, by `"C extensions" in DECLARED_BLIND_SHAPES[3]` (`:1109`) — which is also the
shape of the fix.

**Closure:** key the map by a stable token from the entry (`"runpy"`, `"builtins.exec"`,
`"getattr"`, `"C extensions"`) rather than by index, or extend `:1109`'s idiom to all four —
one substring assertion per index, so a reorder moves a check off its entry and goes red.

### DE17-R2 — LOW — the OVER-CAUGHT paragraph is the one statement in the block with no check behind it

The round's own thesis is that a statement about the predicate must be checked, and the blind
entries now are. The OVER-CAUGHT paragraph (`:171-181`) is not: nothing asserts it exists, and
its checks (`:1054`, `:1064`) never read it. Its disposition even instructs a maintainer to
*"delete the check and the OVER-CAUGHT paragraph together"* (`:1062`) — the same "together" the
blind list now enforces structurally and this one asks for on trust.

Delete the paragraph and the suite stays green at 75; delete the checks and only the count
notices, without naming what was lost. That is precisely the position the blind list was in
before this round.

**Closure:** the idiom the sibling module used for DE15-R4 — assert the binding phrase in the
module's own docstring text (`"ANY object's `.__import__('literal')` contributes that literal"
in the source), so the declaration and its checks fail together.

---

## Executed evidence

At `a8093a5`, 2026-09-02T13:33–13:37Z, `~/ctaNew-wt-rev`, both launchers:

| check | result |
|---|---|
| scope | +160/−27 and +21/−7; ten other DE files byte-identical to `829910e` |
| suites | admissible **75**, ratification **132**, seam **69**, rc 0 each way |
| fifth entry, no key / with key | **red** at the membership check, message naming "5 entries" |
| eval row deleted from the loop | **red** at the map-vs-loop check ("5 rows executed") |
| eval starts catching / compile starts refusing | **red**, each **naming the row** |
| doc meanings swapped | **red** at "BINDING PHRASES" (green at 104 when run at `0ca510e`) |
| **entries 0 and 2 swapped** | **green, 75** — DE17-R1 |
| OVER-CAUGHT paragraph deleted | green — DE17-R2 |
| check deleted outside the loop | `74 == 75`, map still green (the count is a real second net) |
| loop emptied | the **map** fires first, not the count |
| stale claims | `does NOT see` 0, `no way to TEST` 0, `any use at all` 0 |
| over-catch direction | `os.environ.__import__('da_forward_day_verify')` → CAUGHT; the only outside consumer is `ev_replay_seam:1484`, a selftest — a false catch reddens a suite, never admits |
| ratification call-site census | identical at both tips (67/39/4, same loop lengths) — no check removed |
| new messages | 4 of 5 interpolate |
| R-419 / audit / seam / `daw` | True / 25-19, no survivors / **1,875** / True |
| citations | 8 spot-checked (3 from the Q-DE-35 row, 5 from the request) — all exact |
| worktree | clean at `a8093a5` after every mutant restored |

---

## Disposition

- **RELEASED:** DE round 17. DE15-R1, R2, R3 and R4 all close, both of the nuances I filed last
  round are taken (the `_label` re-raise, the disposition clause), and the swap mutant that
  passed two rounds ago now dies by name. **No hold.**
- **FILED:** **DE17-R1** (LOW — positional keying makes a reorder invisible; extend `:1109`'s
  idiom to all four entries) and **DE17-R2** (LOW — the OVER-CAUGHT paragraph has no check
  behind it).
- Both are one edit in the same neighbourhood; neither blocks anything.
