# REVIEW 99 — PART A: DE 131's absolutes. The verification method is sound; one guard has never been watched fire

**Reviewer (pm-codex), 2026-09-07T10:5xZ. Read at `717f634` in `~/ctaNew-wt-rev`; wt-de queried
read-only and untouched. Read-only throughout: no heavy unit, no lock, nothing written under
`data/`, never `--open`. The early-read family is unsealed by the USER, so I censused
mechanics; I interpreted nothing. CHECKED = I went to the artifact or ran the code; AGREED =
I read the same summary. **PART B's inputs had not landed at 10:50:17Z.**

---

# PART A

## §A1 The verification method — **sound, and for a better reason than "it was the only way"**

DE could not run the runner's battery in the shared tree at all: it is red by design under the
freeze. It temporarily restored `de_phase4_diag_runner.py` to the bytes v19 pins, ran the
verification, and restored. **Three things make that sound, and I checked each:**

**1. The bytes it verified against are not an arbitrary choice — they are the bytes the runs
execute.** `ee4034c1…` is v19's pin *and* wt-de's frozen content, so DE verified the new runner
against the cascade E2/E3/E4 actually use. That is the right state to verify in, not a
convenient one.

**2. The leak did not happen — the hazard this method carries, checked at four commits:**

```
7e01ffb  de_phase4_diag_runner.py = b60545d83610f2431d6c9429
286335f  (DE 131's landing)       = 309b98c7a1045d1d28a52fc3
717f634  (the tip)                = 309b98c7a1045d1d28a52fc3
```

(**CHECKED**.) **Nothing landed at `ee4034c1…`**, so the temporary restore stayed temporary.
`309b98c7…` arrived from `f91ad2c` — DE 130, the wording fix REV 98 routed — and DE 131's own
diff touches only the runner and the ledger. *(This is the check that caught an uncommitted-byte
pin at REV 94; it is the reason to keep running it.)*

**3. The residual — DE verified one cascade and a post-freeze GO #9 would run another — is
measurable, and it is nothing.** Between the verified `ee4034c1…` and the tip's `309b98c7…`:

```
top-level defs/assigns CHANGED: ['assign:EXPECTED_CHECKS', 'def:selftest']
```

(**CHECKED**, AST comparison.) **Battery-only; nothing on the draw path.** So the combination
DE verified and the combination production would run differ in no way that reaches a number.

**One thing I could not do, and it bounds the rest of this section:** I could not run the
runner's battery myself to see the two R-782 cells pass — the shared tree is red by design. So
**that they execute green is AGREED** (DE ran it with the pin temporarily fresh); **what they
compute I verified directly** (§A2).

## §A2 The absolutes — landed, and I drove the arithmetic myself

**The landed bytes carry it** (**CHECKED**): `absolute_legs()` at `:4559`, the reconciliation at
`:6119–6145` in the arm-day block, and the ledger's per-day summary carrying
`"absolute": a.get("absolute")` — so *"a reader with the ledger and no receipt can answer what
did 0-cancel make and not only how much better."*

**The design is right in the two places it could have gone wrong.**

*Same replay, not a second pass:* the legs are computed from the fills already in memory —
*"nothing here is a second pass over the day and nothing can disagree with the number it is the
absolute of."* An absolute recomputed separately could drift from its own difference; this one
cannot.

*An unrecorded leg is not a zero one:* the inventory leg is `None` with a reason where BE 96's
position fields are absent. I drove it:

```
arm total  8.0 cents over 4 fills | baseline 4.0 over 2 | difference 4.0   (= D_E0)
inventory leg where BE 96's fields are ABSENT:  None      (not 0)
```

(**CHECKED**, my own call on hand-built fills at (100.2−100)×10 = 2 cents each.) Every number is
hand-computable, which is what a fixture for a new quantity should be. And `HSP_BUY_SIDE()`
reads the side from the policy module *"never the literal 'B'"* — the literal-tracking discipline
applied inside a fixture, which is where it usually is not.

**The landed receipts are untouched** — all five digests still what earlier rounds recorded
(09-03 `b4f11590…`, 09-04 `6c74928f…`, 09-05 `5f0241fc…`, 09-06 `1a2dd10f…`, E1 `5c8a58f5…`)
(**CHECKED**). Batteries runnable at the tip: ledger **8**, design declaration **134**, phase4
**213**, all rc 0. The runner's red still names exactly one module
(`declared ee4034c1…`, `actual 309b98c7…`).

## §A3 The one finding: **the reconciliation refusal has never been watched fire**

`ABSOLUTES_DO_NOT_RECONCILE` is the guard that stops the absolutes being published when they
disagree with `D_E0`. It appears **exactly twice** in the file — the `raise` at `:6125` and its
own name inside the reconciliation block at `:6143` — **and nothing drives it**:

```
R-782 cells in the battery:  10847-10883, two cells, BOTH GREEN
   - the hand-computed legs and their difference
   - the inventory leg is None where BE 96's fields are absent
nothing perturbs _abs_arm / _abs_base / observed to make the identity fail
```

(**CHECKED**, every mention and every call site enumerated.) **This is the class DE itself named
two rounds ago, in its own phase4 cell: *"a guard nobody has watched fire is a comment."*** The
green control is excellent — hand-computable, unit named, both legs — and the red half of the
pair is missing.

It matters more than a missing falsifier usually does, because of *what* the guard protects. The
whole argument for these numbers is that the absolute and the excess come from one replay and
therefore cannot disagree. If they ever do, something deep is wrong — and that is precisely the
moment the refusal has to work, having never been exercised.

**Closure: one cell.** Call the reconciliation arithmetic with a perturbed `observed` (or a
fills list whose difference is not `observed`) and assert `RunnerRefused` carrying the name.
**Not a hold** — DE 131 lands for **GO #9 onward**, and E2/E3/E4 and tonight's GO #8 all run
frozen bytes that predate it. **It should close before GO #9.**

---

# PART B — pending

Not landed at **10:50:17Z**: `p003_de_early_read_day_20260904__*.json`, and DA 126's read after
it. **I censused nothing and AGREED nothing about either.**

**One prediction from REV 98 §B0 stands, and one caution beside it.** `fe76d83` includes ledger
v2 (`edb9dee` is its ancestor and is the commit that introduced `SCHEMA_VERSION = 2`), so if E2
writes a ledger it should declare **schema 2 with the inventory fields**. **But MEM round 266
reports that on E2's own path no decision ledger is written at all** — which, if so, is the more
important fact and would make the schema question moot for this artifact. **I have verified
neither**; both are for Part B, and the artifact settles it. If no ledger is written, the
question worth asking is not what schema it declares but **why R-765's *store the numbers* did
not reach the early-read path**, since that path is the one producing the numbers the user asked
to see.

When they land: the artifact by KEYS; the ledger fields it names (`path`, `sha256`, `rows`,
`schema`) with **the named digest recomputed against the file**; DA's table for its mechanics,
including **the new baseline line DA derives from the ledger and the reconciliation
arm − baseline = D_E0 shown**; and **whether DA recomputed D_E0/Z from the ledger or said it
could not yet**.

---

# §C HOLDS AND ROUTING

**No holds.**

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | `ABSOLUTES_DO_NOT_RECONCILE` is never driven — one cell, before GO #9 | routed (§A3) |

**Closed this round, observed in passing:** REV 98's wording item (DE 130, `f91ad2c` — the cell
now says what it establishes); the landed receipts confirmed untouched across DE 130 and DE 131.

# §D WHAT I DID NOT ESTABLISH

- **AGREED, not established:** that the two R-782 cells execute green inside the runner's
  battery — the shared tree is red by design, so I verified what they compute (by driving
  `absolute_legs` myself) and not that the battery reaches them.
- **Not established:** E2's artifact, its ledger or its absence, DA 126's read (Part B); MEM's
  report that E2's path writes no ledger.
- **Method note:** the leak check in §A1 exists because REV 94 found a pin taken over
  uncommitted bytes. Running it here cost one command and confirmed a clean restore — which is
  the argument for keeping cheap checks that have fired once.
