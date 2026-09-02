# Review — DE round 13 (DE11-R1 and CO-6/DE12-R1 closed)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `f04c06a`** (Q-DE-31).
**Request of record:** `REQUEST_DE_ROUND_13_2026-09-02.md`.
**Composed 2026-09-02T12:13:40Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach f04c06a`. Read-only under `data/`; register
fixtures on copies in temp dirs — `COORDINATION.md` never written. No timer, no service,
no launcher. DE12-R2 and the audit-count wording are out of scope and are not re-filed.

---

## Verdict

### RELEASED. Both findings close, and the reach of each closure is wider than it was asked to be.

Two findings: **DE13-R1** — CO-7 confirmed, and I can state it more strongly than the
observation did: reverting the fix leaves the suite **green at 84**. **DE13-R2** —
`stamped_at_raw` is undocumented. One judgement given on item 3, and I disagree with DE's
"a limit cannot be tested".

---

## 1. DE11-R1 closed as filed

| shape | result |
|---|---|
| `exec('import X')` | **REFUSES**, naming the shape (`<opaque-exec>`) |
| `eval("__import__('X')")` | **REFUSES** |
| bare `compile(...)` | **REFUSES** |
| rebound `__import__` | **REFUSES** (`<rebound-import>`) |

The controls hold: literal `import X`, `importlib.import_module('X')`,
`from importlib import import_module`, aliased `importlib`, `__import__('X')` with a
literal — all resolve and are **caught** as verdict imports; a non-literal argument
refuses. The refusal is raised from `reads_no_verdict` on the assembled set, so no partial
answer is given.

## 2. The false positive, and no residue in this closure

`re.compile('x')` → `['re']`, not refused: the narrowing to **bare names** works.

**Does the predicate now refuse a file it should read?** I checked the closure rather than
reasoning about it — `grep '\bcompile('` finds no bare `compile` anywhere outside
`re.compile`/`.compile(`, and the only bare `exec`/`eval` tokens in the tree are this
module's own **string fixtures and comments**, which the AST does not see as calls. And
the decisive test:

| file | `reads_no_verdict` on its own source |
|---|---|
| `de_admissible_windows` | **True** |
| `de_ratification_check` | **True** |
| `ev_replay_seam` | **True** |
| `be_forward_day`, `da_blackout_mask` | False — they legitimately import verdict producers, which is the predicate working, not failing |

The self-refusal DE caught is gone, and no file in the closure is refused that should be
read. That is a property of the closure as it stands today, not a guarantee — a future
module with a legitimate bare `exec` would be refused, and the declared-blind list is
where that would have to be resolved.

The attribute forms behave as declared: `builtins.exec(...)`,
`getattr(builtins,'exec')(...)` and an aliased `b.eval(...)` all yield `['builtins']` and
**pass silently** — and `DECLARED_BLIND_SHAPES` names them, with the reason attached
(*"matching the bare name is what keeps `re.compile` from reading as an opaque exec"*).
The trade is stated where a reader meets it.

## 3. The declared limit — I disagree with "it cannot be tested"

DE's position is that there is no way to test for a shape one cannot see, so the assertion
is that the list exists and names them. **The premise is true about detection and false
about documentation.** You cannot test that the predicate *sees* `runpy.run_path`; you can
test the observable consequence of not seeing it:

```
imported_modules("import runpy; runpy.run_path('x')") == {'runpy'}   # expected-blind
```

That assertion fails in **both** directions — if a later change starts catching the shape
(the set grows) or starts refusing it (an exception) — which is exactly the property a
declared limit needs, because *a declared limit that silently stops being true is the
failure mode of an untested list*. The codebase already has the idiom: the audits'
`refuses_on_the_control: false` is the same construction pointed the other way.

I verified all five declared shapes behave as declared (the three `builtins` forms,
`runpy.run_path`, and `getattr(importlib, "import_module")`), so writing the five
expected-blind assertions is mechanical. **Recommend it**; it is not a defect today.

## 4. The rebound-`__import__` reach — wider than the list

Every shape I tried **refuses**, including three the request did not name:

| shape | result |
|---|---|
| `f = __import__; f('X')` | REFUSES |
| chained `f = __import__; g = f` | REFUSES |
| dict value `{'i': __import__}` | REFUSES |
| list element `[__import__][0]('X')` | REFUSES |
| default argument `def h(i=__import__)` | REFUSES |
| keyword argument `k(imp=__import__)` | REFUSES |
| **tuple unpack** `a, b = 1, __import__` | REFUSES |
| **attribute assignment** `o.i = __import__` | REFUSES |

And the legitimate forms are not swept up: `__import__('X')` with a literal resolves and
is **caught** as a verdict import; `__import__` inside a **string** or a **comment**
passes. The rule is evidently "a reference to the bare name that is not a literal call",
which is the right one — nothing joins the declared-blind list from this axis.

## 5. CO-6 / DE12-R1 closed at entry

`stamped_dt = parse_instant(stamped_at, "stamped_at")` sits at `check()+34`, **before**
the `if stamped_at is None` branch at `+68` — parsed at entry, not on a path.

| ref | `stamped_at` | result |
|---|---|---|
| **R-419** (not superseded) | `'not-a-time'`, `123`, `''` | **all REFUSE by field and value** — including the non-string widening I filed |
| R-419 | `'2026-09-02T10:30:00Z'` | verified; emission carries the **parsed** value plus `stamped_at_raw` |
| R-419 | `None` | verified — still "no receipt" |
| R-418 | 10:30Z / 11:30Z / `'not-a-time'` | `provenance True` / ALREADY-superseded refusal / refuses |

DE12-R1 is closed as filed, and the `unparsable_stamped_at` audit path now reaches the
entry parse.

### DE13-R1 — LOW-MEDIUM — CO-7 confirmed, and the fix has no falsifier at all

The coordinator observed that the checker's diff adds no selftest line (84 → 84). I
confirm it — `git diff 9dbaa5a..f04c06a` on the checker contains **zero new `ok()` lines**
— and I can put it more sharply than an absence:

**I restored the exact pre-fix shape** (parse only inside the superseded branch, emission
echoing the raw value) and ran the suite: **`selftest OK — 84 checks`**, and
`check(sup, "R-419", stamped_at='not-a-time')` returned `verified True` again. **The
defect can be reinstated in full with the suite still green.**

(A cruder mutant — deleting the parse outright — does go red, but with a `TypeError` from
the superseded path comparing a datetime to `None`. That is an incidental crash from a
shared variable, not a control on the behaviour the fix exists for.)

Severity: the coordinator filed LOW and the behaviour is correct today, so it is not
urgent. I place it **LOW-MEDIUM** because this module's entire discipline is that every
guard ships a falsifier, and this is the one fix that would regress invisibly — in the
field that carries provenance. **One `ok()` line closes it**: assert that a non-superseded
ref with an unparsable `stamped_at` refuses.

### DE13-R2 — LOW — `stamped_at_raw` is undocumented

The field appears exactly **once** in the module, at the emission site (`:652`), with no
comment, no docstring mention, and no note in the emission itself. A consumer reading a
result cannot tell that `stamped_at` is the canonicalised parse and `stamped_at_raw` is
the string as supplied — which is the whole reason both exist. **Closure:** one line of
docstring or an emission note, in the idiom `refusal_scope` and `decides` already use.

## 6. Counts and shapes

`de_admissible_windows` 53 → **62** with `EXPECTED_CHECKS = 62` asserted at run time; I
emptied a selftest loop and the suite **failed** — the count backstop holds here as it
does in the checker. Ratification **84 / 84** under both launchers, seam **69**.

## 7. Nothing under review moved

R-419 → `verified_for_new_run True`; R-418 @10:30Z → `provenance True`; the seam emits
**1,875** specs on the real 09-01 supply; `ev_replay_seam.daw is de_admissible_windows` →
**True**; `mutation_audit` 19 paths, `survivors []`.

## 8. Rule 10 / rule 14

Refusal messages compute what they print — each names the shape or the field, the value
received, and (for instants) the formats accepted. The emission still carries
`decides: "nothing -- this reports; admission is the coordinator's act and accrual is
decided elsewhere"`.

---

## Executed evidence

At `f04c06a`, 2026-09-02T12:10–12:13Z:

| check | result |
|---|---|
| scope | the two named modules + the register |
| suites, both launchers | admissible **62**, ratification **84**, seam **69**, rc 0 |
| four opaque shapes | **all REFUSE**, each naming its shape |
| seven controls | resolve and are caught; non-literal refuses |
| closure files vs their own predicate | 3 of 3 pass; the two that return False import verdict producers legitimately |
| bare `compile`/`exec` in the closure | **none** outside this module's own fixtures |
| attribute forms | pass silently and are **named** in `DECLARED_BLIND_SHAPES` |
| eight rebinding shapes | **all refuse**; literal call / string / comment not swept up |
| `stamped_at` parse position | `check()+34`, before the `None` branch at `+68` |
| R-419 `'not-a-time'` / `123` / `''` | **all refuse** |
| **pre-fix shape restored** | **suite green at 84**, defect returns — DE13-R1 |
| new `ok()` lines in the checker diff | **0** |
| `stamped_at_raw` occurrences | **1**, undocumented — DE13-R2 |
| admissible loop emptied | suite **fails** (count backstop) |
| R-419 / R-418@10:30Z / seam / `daw` | True / True / **1,875** / True |
| register file | never written; worktree clean |

---

## Disposition

- **RELEASED:** DE round 13. DE11-R1 is closed with a reach wider than filed, and
  CO-6/DE12-R1 is closed at entry including the non-string widening. **No hold.**
- **FILED:** **DE13-R1** (CO-7 — the pre-fix shape passes the suite; one `ok()` closes it;
  I read it LOW-MEDIUM rather than LOW, for the regression risk in a provenance field)
  and **DE13-R2** (`stamped_at_raw` undocumented).
- **Judgement, item 3:** the declared-blind list **can** be tested — not for the shape,
  but for its observable consequence — and an expected-blind assertion per entry fails in
  both directions. Recommended, not required.
