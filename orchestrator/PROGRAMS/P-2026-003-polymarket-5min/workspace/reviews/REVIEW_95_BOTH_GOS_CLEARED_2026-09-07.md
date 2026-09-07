# REVIEW 95 — **GO E2 MAY PROCEED. GO #8 MAY PROCEED.** Both re-gated at the tip's digests

**Reviewer (pm-codex), 2026-09-07T09:1xZ. Read at `fe76d83`, re-checked at `0f6973f` (the
checker commit, which moves none of the gated digests). Read-only: no heavy unit, no lock,
nothing written under `data/`, never `--open`, and **no economic value read from either
early-read artifact**. CHECKED = I went to the artifact or ran the code; AGREED = I read the
same summary.**

---

# PART A — both GOs clear

> **GO E2 MAY PROCEED** and **GO #8 MAY PROCEED** at:
>
> ```
> runner       de_multiday_gate1_runner.py   ccc4108d28f0754192bc9fce…
> early read   de_early_read.py              ba6daba426491cafa17d843e…
> ledger       de_decision_ledger.py         d78c370151cea431127bbb77…
> params v19   …params_v19.json              dd8db7ded9e6ed9723173a3a…
> design v27   …design_v27.json              3bcdf3c234cb7d4e4be116c2…
> ```
>
> **No holds.** REV 94's NO-GO is closed, and closed as the class it belonged to rather than
> as the instance. One forward-looking item is routed with a dated trigger (§A5).

## §A1 The three batteries, run by me at the tip

```
de_multiday_gate1_runner --selftest   PASS -- 362 checks, 0 disarmed, 0 skipped   rc 0  (32.4 s, 886,672 KiB)
de_early_read            --selftest   PASS --  21 checks, 0 disarmed, 0 skipped   rc 0   (aborted at 13/18 last round)
de_decision_ledger       --selftest   PASS --   8 checks, 0 disarmed, 0 skipped   rc 0
rehearse_smoke('2026-09-07')          NOT_READY, blocking ['P2_book_exists','P2_builder_receipt_exists']
```

(**CHECKED**.) Tonight's day is blocked **only on the book**, as it should be for a day that has
not closed. And the gate conditions carried from earlier rounds still hold: **the ten cascade
pins all match at the tip (10 of 10, `verify_be_module` admits), and `P3_design` HOLDS** (design
v27 pins `dd8db7de…` = v19's digest) — third round running.

## §A2 REV 94's NO-GO — closed at the class, not at the instance

The cell no longer names a day. `bar_day_states()` derives `read` / `unread` / `next_unread`
from the ledger, and DE states the class in the code: *"A cell that names a day as a LITERAL
measures the day it was written on: `rehearse(\"2026-09-03\")` expected READY, GO E1 read
09-03, and the cell aborted the battery with a KeyError — five checks after it never ran."*
Measured now:

```
read ['2026-09-03'] | unread ['2026-09-04','2026-09-05','2026-09-06'] | next_unread 2026-09-04
```

(**CHECKED**.) And **the cell that used to expect READY on 09-03 is now the positive control
that it refuses**: *"2026-09-03 has been READ, so its rehearsal refuses by name —
`['EARLY_READ_ALREADY_EMITTED']` — rather than offering to run it again."* Turning the thing
that broke the battery into the control is the right repair; it cannot regress silently.

**Can a cell still abort?** `ok()` raises `SystemExit` with a named FAIL, so a failed check is a
verdict, not a traceback. The one remaining unguarded nested read (`reh["preconditions"]` at
:727) is reached only after :726 rehearses `next_unread`, a day guaranteed unread and therefore
READY — so it is unreachable in the failing direction **today**. See §A5 for its dated trigger.

## §A3 What phase 2 changed in the runner — **no change touches a verdict path**

Two literals became computed:

```
status:                    "DAY_RUN_SEALED"  ->  DAY_RUN_SEALED / DAY_RUN_UNSEALED
                                                 (+ the _NO_ADMISSIBLE_ARM variants), from _all_sealed
the_economics_are_SEALED:  n_days_complete < params["G"]  ->  _all_sealed
                           "read from the emitted arm-days' own `sealed` flags, never from
                            `n_days_complete < G` -- which was true on the R-754 early read
                            while both arm-days were unsealed"
```

**I confirmed the defect at the landed artifact.** In
`p003_de_early_read_day_20260903__20260907T085436Z.json`:

```
.day_run.status                                  = 'DAY_RUN_SEALED'
.day_run.what_this_is_not.the_economics_are_SEALED = True
.day_run.per_day_sealed_artifacts[*].sealed      = [False, False]
                                   'economic' present = [True, True]
```

(**CHECKED**.) **The artifact asserts twice that its economics are sealed while carrying them in
the open.** The old field was a proxy — *days complete* standing in for *is this sealed* — and
the early read is exactly the case where the proxy and the fact part company. Computing it from
the arm-days is the correct repair.

**Does it touch a verdict path? No.** Both are receipt *description* fields, and
`grep -rn "DAY_RUN_SEALED\|DAY_RUN_UNSEALED" live/ --include=*.py` outside the runner returns
**nothing** — no consumer keys on the string, so tonight's first-ever `DAY_RUN_UNSEALED` breaks
no reader (**CHECKED**). The R-id existence check touches the design-family census, not a day
run. The early-read change is test-only. The ledger's schema v2 adds fields. **Nothing changes a
computed number.**

**Are the sealed days still reproducible? Yes** — `seal(arm, 3, 6)` still returns
`sealed=True`, the landed `"SEALED -- 3 of 6 days complete…"` string, economics absent; and the
four landed receipts are untouched (**CHECKED**).

**Routed, and it is the residue of this repair:** the landed 09-03 artifact keeps both false
fields, because rule 13 forbids editing it. The fix is forward-only, DA 125's table was printed
from that artifact, and **nothing in the ledger records that those two fields are wrong**. A
one-line superseding note — the same shape as DE's earlier eight-name-scope disclosure for
09-03 — closes it.

**My REV 93 §A5 residual is closed, driven by me:**

```
landed_at "R-765"   (exists)    PERMITTED
landed_at "R-99999" (no entry)  REFUSED USER_RULING_ENTRY_NOT_IN_THE_REGISTER
landed_at "banana"              REFUSED USER_RULING_BLOCK_INCOMPLETE
```

The check now reaches as far as its own justification did.

## §A4 The decision ledger, schema v2 — and the refusal that matters

```
D_E0 and Z RECOMPUTED FROM THE LEDGER match the receipt to 1e-9 (Z 2.418751980 vs 2.418751980, 600 draws)
the statistics the SEALED receipts could not carry -- p one-sided, p TWO-sided, rho, the fills leg -- all from the rows
the INVENTORY LEG recomputes to 1e-9 over 40 fills (unit 'shares')     -- ABSENT_UNTIL_BE_96 closed
RED: a ledger lacking BE 96's fields REFUSES `DECISION_LEDGER_NO_INVENTORY_FIELDS`
     "A zero leg and an unrecorded one are the same number and opposite facts"
```

(**CHECKED**, run by me.) That last sentence is the whole of reliability rule 4 in one line, and
it is the cell I would have asked for: the failure mode of an added field is that its absence
reads as zero. It refuses instead. The ledger also reads and recomputes **from a directory
holding it and nothing else** — no book, no model, no receipt — which is what *"avoid rerun"*
has to mean if it is to mean anything.

## §A5 One routed item, with a dated trigger — the same class, one step out

The derivation fixed the literal. The cell at :714 still asserts `_st["read"] and
_st["next_unread"]`, so it requires **at least one day still unread**. Driven:

```
today (1 read, 3 unread)   _st['read'] and _st['next_unread']  ->  True
after GO E4 (all four read)                                    ->  False   -> battery FAILS (named, not a traceback)
```

(**CHECKED**, by simulating the post-E4 ledger in a scratch root.) **The battery goes red when
the fourth early read completes** — i.e. after GO E4, on the current plan. Not a blocker for E2
or E3, and it fails as a named check rather than a crash, which is the improvement. But it is
the same shape one step out: *a cell whose green depends on the programme not having finished*.
**Closure:** assert on whichever terminal state exists — if a day is unread, that it rehearses
READY; if none is, that **every** ruled day rehearses `ALREADY_EMITTED`. Both are legitimate;
only one is currently expressible.

## §A6 The phase4 suite is correctly HELD

`de_phase4_diag_runner.py`'s last commit is `c707eb8` (BE 96) — **DE landed none of its phase4
work**, as R-771 ruled, and the module's selftest still FAILs on the pre-existing R-499
admission I bounded at REV 93 §B3 (**CHECKED**). Holding it is also what keeps the cascade pin
valid: that module is one of the ten, pinned at `ee4034c1…`, and landing a change would have
re-opened REV 93's NO-GO the same day it was closed. **AGREED** on DE's own 209 → 249 count
correction; I did not re-derive it.

---

# PART B

## §B1 The checker's scope line — **REV 92 §A6 #4 / REV 93 #5 is closed**

```
PIN CENSUS SCOPE: files matching *.json under $DIR and $LEDGER at depth 1, size < 5 MB,
the fork's own file excluded; digests matched full and 16-hex; JSON path per hit
  [ -- SKIPPED in this run   when IMMUT_SKIP_PIN_CENSUS is set ]
```

(**CHECKED** at `47157d8`.) It states every limit I named — depth, size, the excluded file, the
matching — **and it says when the census was skipped**, which I had not asked for and which is
the difference between a scope and an alibi. The field can no longer read as *these and no
others*.

## §B2 DA 125's locator catch — the fixture reproduced the reader's own assumption

DA's own words: *"This reader was built against the sealed shape and its fixture reproduced
that assumption, so the first real artifact refused `ECONOMIC_FIELD_MISSING` on an artifact
that had every field."* Sealed receipts carry the economic names **flat on the arm** (they are
what sealing removes); the early-read artifact carries them in a nested `economic` block. Both
shapes are read now and **the one in use is REPORTED** (`economic_block` / `flat_on_the_arm`).

**This is rule 16 one level over.** The rule says a fixture must never supply what the code
under test should produce; here the fixture supplied the reader's own *assumption about shape*,
so the reader and its control agreed about a thing neither had checked, and the first real
artifact was the first disagreement. Catching it **at the census before printing** is the right
place — a reader that had printed first would have published a refusal about an artifact that
was complete.

## §B3 What REV 94's Part B left open

- The early-read artifact's `ruling.path` is **still absolute into `/home/yuqing/ctaNew-wt-de/`**
  while `computation_params.path` beside it is repo-relative (**CHECKED**, unchanged). Carried.
- `verify_be_module` still returns the same shape whether it verified one module or ten
  (**CHECKED**, unchanged). Carried.

---

# §C ROUTING

**No holds.**

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE | the early-read battery goes red after GO E4 — assert on whichever terminal state exists (§A5) | routed, dated trigger |
| 2 | DE | the landed 09-03 artifact keeps `DAY_RUN_SEALED` / `the_economics_are_SEALED: true` against unsealed arm-days; the fix is forward-only and nothing records that those two fields are wrong (§A3) | routed |
| 3 | DE | `verify_be_module` should report `n_modules_verified` | carried, minor |
| 4 | DE | the artifact's `ruling.path` absolute into `wt-de` beside a repo-relative sibling | carried, minor |

**Closed this round:** REV 94's NO-GO (at the class); REV 93 §A5's R-id existence residual;
REV 93 #5 / REV 92 §A6 #4 (the scope line); REV 91 §C1 (the shared-falsifier cell — the
early-read battery now carries it, and the module's own count grew 18 → 21).

# §D WHAT I DID NOT ESTABLISH

- **Not established:** DE's 209 → 249 phase4 count correction (AGREED); the numbers in either
  early-read artifact (deliberately unread — 89 floats and 72 ints exist in E1 and I named
  none); DA 125's printed table beyond its locator mechanics.
- **Method note:** for §A5 I simulated the post-E4 ledger in a scratch root rather than reason
  about the branch, because the same reasoning is what produced the REV 94 abort — the state
  after the programme finishes is exactly the state nobody pictures.
