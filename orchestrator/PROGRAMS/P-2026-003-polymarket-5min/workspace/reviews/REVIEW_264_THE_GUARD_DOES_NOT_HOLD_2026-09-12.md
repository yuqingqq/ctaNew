# REVIEW 264 — the anti-amendment guard does not hold: five attacks, five succeed

REV round 228. Filed 2026-09-12T01:27:53Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Driven at
`origin/de-freeze-chain-v2` = `00ce098`, module `de_band_hazard.py`.

## THE VERDICT: IT DOES NOT HOLD

**The guard refuses correctly on honest inputs and is bypassed by anyone who
supplies different ones.** Its four inputs — `lever`, `declared_utc`,
`clock_start_utc`, `outcome_is_known` — are **all caller-supplied arguments, and
not one is read from an artifact.** A guard whose entire evidence base is what
the constrained party types is a statement of intent, not a constraint.

You asked for this to be adversarial. It is, and I am not softening it: this is
the most self-serving thing in the lane and it does not currently stop the
thing it was built to stop.

## 1. The controls fire — on honest inputs

| control | result |
|---|---|
| no timestamps | REFUSED `AMENDMENT_CHOSEN_AFTER_THE_CLOCK_STARTED` |
| declared after the clock | REFUSED, same |
| `outcome_is_known=True` | REFUSED, same |
| unknown lever | REFUSED `UNKNOWN_LEVER` |

All four correct, and the module's own falsifier is green: **37/37, rc 0**,
including *"refuses: declared after the clock"*. **Renaming a lever is caught.**

## 2. Five attacks, five succeed

| # | attack | result |
|---|---|---|
| 1 | **Backdate**: `declared_utc = "2020-01-01T00:00:00Z"` | **ADMITTED** |
| 2 | **Amender supplies the clock**: `clock_start_utc = "2027-01-01T00:00:00Z"` against a declaration dated 09-20 | **ADMITTED** |
| 3 | **Outcome known, not declared**: omit `outcome_is_known` (defaults `False`) | **ADMITTED** |
| 4 | **Timezone offset**: `declared_utc = "2026-09-13T23:00:00-05:00"` — which is **09-14T04:00Z, four hours AFTER** a clock starting `2026-09-14T00:00:00Z` | **ADMITTED** |
| 5 | **Add a lever** (your bet, in its real form) | **ADMITTED, and worse — see §3** |

**(1) and (2) are the same defect twice**: `declared_utc` and
`clock_start_utc` are parameters. Nothing verifies the declaration's date
against a commit, and nothing verifies the clock start against the validation
window. The amender writes both sides of the inequality.

**(3) is the strongest clause reduced to opt-in self-incrimination.**
`outcome_is_known: bool = False`. The guard's best refusal — *"the shortfall is
the information being used"* — fires only when the party using the information
says so.

**(4) needs no bad faith at all.** The comparison is
`declared_utc >= clock_start_utc` on **strings**, so it is lexicographic. A
declaration rendered with a negative UTC offset sorts before a `Z`-rendered
clock start whose instant is earlier. `+00:00` against `Z` at the *identical*
instant also admits, because `+` (0x2B) sorts below `Z` (0x5A). This one fires
by accident, and a correct timestamp in a different rendering is all it takes.

## 3. Your bet was right, and the mechanism is the opposite of renaming

**Renaming is caught** — `UNKNOWN_LEVER` fires. **Adding is not.** The guard
reads `spec.get("must_be_declared_before_the_clock")`, so a lever registered
**without that key** is falsy and takes the early return:

    vi_extend_the_observation_window   (key omitted)        ADMITTED
    vii_relax_required_days            (key set False)      ADMITTED

and both were admitted **with `outcome_is_known=True`**, because the
flag check sits **above** the outcome check and returns first:

    if not spec.get("must_be_declared_before_the_clock"):
        return {"admissible": True, ...}       # <- returns BEFORE
    if outcome_is_known:                       #    the strongest clause
        raise ...

**So the weakest check gates the strongest clause.** A lever added tomorrow
that does exactly what `iii_longer_band` does, under a new name, with one key
forgotten, is admissible after the band has fallen short and after the outcome
is known. That is precisely the situation the guard exists for, and the default
is permissive — the class this programme has been removing all night, here in
the instrument built to stop us.

## 4. And it has no production call site

`amendment_is_admissible` appears **5 times lane-wide, all inside its own
module** — the definition, one prose reference, and three calls from its own
falsifier. **Zero occurrences in any other file** (positive control: the same
query finds `levers(` in its caller). So nothing in the lane consults it before
an amendment lands. It is a guard with cells and no site.

## 5. What it would take to make it hold

Every fix is to bind an input to an artifact instead of an argument:

1. **`clock_start_utc` ← the validation-window declaration**
   (`da_validation_window_v1.json`, REVIEW 247 R4), read at a fetched ref. Not
   a parameter.
2. **`declared_utc` ← the committer timestamp of the commit that lands the
   amendment declaration**, present on both executing refs — the same R1
   construction used for `T_eff`. A date cannot then be typed.
3. **`outcome_is_known` ← computed**, not asserted: does any band-day score,
   evaluable count or verdict artifact exist at the ref? If the band has
   produced output, the outcome is known whether or not anyone admits it.
4. **`spec["must_be_declared_before_the_clock"]`, not `.get(...)`** — a lever
   missing the key must raise, not pass. And pin the lever set the way DA's
   freeze checker now pins its enumerations, so a lever cannot be added without
   the pin refusing.
5. **Parse both timestamps to aware datetimes and compare instants.** Never
   lexicographic.
6. **Reorder**: check `outcome_is_known` **before** the lever's flag, so the
   strongest clause is not gated by the weakest.
7. **Give it a call site**, or it constrains nothing.

## 6. What I am not claiming

The guard's *design intent* is right and its refusal vocabulary is good —
`AMENDMENT_CHOSEN_AFTER_THE_CLOCK_STARTED` names the failure exactly, and
`UNKNOWN_LEVER` closes the rename. The falsifier is green and its cells test
real properties. **The defect is not the idea; it is that every input is
supplied by the party being constrained.** Fixes 1–3 alone would move it from
a statement of intent to a guard.

## 7. What I excluded

Driven: `amendment_is_admissible` and `levers()` at the ref, the module's
falsifier, and a monkey-patched `levers()` for attack 5. Grepped lane-wide for
call sites with a positive control. **Not read**: whether any *process* outside
the lane consults the guard, the runbook's procedural rules around amendments,
or whether a human review step is intended to supply what the code does not. If
such a step exists, it is not in the code and a reader of the code cannot find
it.
