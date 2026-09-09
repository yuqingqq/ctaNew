# REVIEW 125 — an audit of my own greens: which were checking the wrong thing?

**REV, 2026-09-09T09:01Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`.

**THREE OF MY GREENS WERE TRUE OF THE ARTIFACT THEY EXAMINED AND FALSE OF THE CLAIM THEY
WERE CITED FOR. Two more I had already caught. The rest stand, and I state for each what it
actually examined.**

The worst is **REVIEW 110 §(4)**, because the coordinator carried it into the user's decision
brief as the reason the cancel is the action. **I own that one first.**

---

## THE THREE THAT WERE CHECKING THE WRONG THING

### 1. REVIEW 110 §(4) — THE WRONG ID SPACE. Confirmed at the code.

**What I wrote:** *"the unit is the ACTION. `_cancels` keys by `(slug, side, gen)` — one entry
per generation — and the engine's own invariant is one cancel per generation. So the 39.7 % of
rows collapse to one cancel by construction."*

**What the check actually examines** — `harmful_stateful_policy`, verbatim:

```python
issued: dict = {}          # (slug, side, policy_gen) -> issued event
...
out["one_cancel_per_generation"] = all(n <= 1 for n in issue_counts.values())
```

**It is keyed on `policy_gen`.** The trajectory schema carries **`ref_gen` AND `policy_gen`
side by side** in every event. A repost makes a second POLICY generation (`7.r1`) from one
REFERENCE generation (`7`), so the invariant passes while the reference generation has been
cancelled twice.

**The claim it was cited for was about REFERENCE generations** — that the cancel is the
action, one per generation of the reference. **True of the id space it checks, false of the
claim. That is the class, and it is mine.**

**Second-order, and worth having:** `de_cancel_count_delta._cancels` keys `by_gen` on
`int(c["ref_gen"])` while `cancels_issued` counts every issue, so **the check I asked for in
REVIEW 118 (`len(by_gen) != cancels_issued`) WILL FIRE on a repost-enabled day** — the
instrument refuses rather than silently under-counting, which is the right failure, but it
means it cannot run there at all. And `de_matched_cancel_control.draw_one` draws one row per
**reference** generation, so the control's action count cannot match an arm that cancels one
reference generation twice. **The matching premise is false in the same place the invariant
is.**

### 2. REVIEW 123 §(2) — I examined the SET and never the PREDICATE. Re-driven now.

I established that `SCORING_PATH_MODULES` is a hand-typed 5 of a recorded 49. **I never asked
what the predicate does with a receipt that carries only some of the five.** Driven today:

```
all five modules present     -> BOOK_SCORING_CODE_MATCHES  n_checked=5
ONE of the five present      -> BOOK_SCORING_CODE_MATCHES  n_checked=1
two of the five              -> BOOK_SCORING_CODE_MATCHES  n_checked=2
none at all (empty closure)  -> REFUSED BOOK_SCORING_CODE_NOT_RECORDED
```

**The predicate means "every module the receipt happens to name matches", not "the scoring
path is the declared one".** With my REVIEW 123 finding it is doubly weak: a sample of the
closure, and any non-empty subset of that sample suffices. **This is the user's `n_checked: 1`,
and my green missed it because I audited the membership of the list and not the behaviour of
the check.**

### 3. REVIEW 113 — my corpus tested the guard's STRINGS and was silent about its FIELD SET

Every case I drove passed `{"economic": {"Z": v}}`. So I established what the guard does with
**text**, and established nothing about **which quantities it guards** — which is the hole I
found myself two rounds later (REVIEW 123: `ECONOMIC_FIELDS` does not name `D_E_settle`), and
which the user has now confirmed from the other side with a planted settlement result.
**REVIEW 113's green was true of the field it examined.**

## THE TWO I HAD ALREADY CAUGHT — named so the ledger is complete

- **REVIEW 105 §3** — I passed a `Path` where the function wanted the parsed receipt, so the
  walk could not have found an L if one had been there. *The conclusion survived and the
  evidence did not.*
- **REVIEW 120** — the consumer enumeration was one spelling, not the operation; five, not
  three (REVIEW 122).

## THE GREENS THAT STAND, WITH WHAT EACH ACTUALLY EXAMINED

| review | what the check examined | cited for | same thing? |
|---|---|---|---|
| 110 §(1) | the battery going RED under two mutations of the module | "the falsifiers can fail" | **yes** |
| 110 §(2) | the pre-fix code's stream vs `old_stream`, **on a 2-generation fixture** | "the old aggregation is faithful" | **on that case** — never on a day's population; the limit stands |
| 115 | a fixture run's ledger + the landed 09-07 receipt, with a cross-derivation control that flips a sign | "the statistics re-derive" | **yes**; the settlement half is fixture-only, as I said then |
| 116 | the 09-07 receipt + its digest-verified ledger, with a one-draw tamper control | "the published contract works" | **yes**; `D_E0` is a read-back, as I said then |
| 117 | 2,880 real boundary reads, against a deliberately different reader | staleness bounded; one verdict flips | **yes** — **but btc/S60 only; I did not test the other coins or S30** |
| 118 (1) | DE's `_old163` against the verbatim old rule, plus my corpus | "the reimplementation is the old rule" | **yes** |
| 118 (2) | the refusal, the `TypeError`, and a 24-hour-old cache | site 4's argument | **yes** |
| 118 (3) | the four books' parsed receipts under the armed guard | "arming costs nothing" — **refuted** | **yes**, and its scope was corrected in 120 |
| 121 | BE 114's three, each with a green control | "the three fixes hold" | **yes**, after I supplied the green control I had first omitted |
| 124 `_GEN_REQUIRED` | the constructor's field set + a sweep of seven consumers | "the set is the object's fields" | **yes** |
| 124 `fit_code_files` | greps for `be_rule22`/`import_closure` in three fit modules | "no fitting closure is recorded" | **grep-based, and I named it as not driven** |
| 124 exclusion vocabulary | the typed triple and the identity that consumes it, **read not run** | "loud at the window level" | **REASONED, NOT DRIVEN — I am naming it as an un-driven green** |

## WHAT I WOULD RE-DRIVE, AND WHAT I DID

- **Re-driven this round:** REVIEW 123 §(2)'s predicate (above), and REVIEW 110 §(4)'s id
  space (above, at the code).
- **Would re-drive, cannot without a corrected book:** REVIEW 110 §(2) on a real day's
  population rather than a two-generation fixture.
- **Would re-drive, cheap, did not this round:** REVIEW 117 across the other coins and S30 —
  my staleness bound is a btc/S60 bound and I quoted it without that qualifier in the
  synthesis; and REVIEW 124's window-level exclusion identity, which I reasoned about and
  never ran.

## THE PATTERN IN MY OWN FAILURES, SINCE THAT IS THE USEFUL PART

All three are the same shape and it is not carelessness about the code — **it is examining
the thing in front of me and reporting on the thing it is named after.** The invariant is
*called* `one_cancel_per_generation` and I did not ask **which generation**. The predicate is
*called* `assert_book_scoring_code` and I asked whether its list was right, not whether it
checked the list. The guard is *called* the sealed-value guard and I asked what it does with
strings, not what it considers sealed. **The question that would have caught all three is the
same one: what does this check REFUSE, and is that the set of things the claim says it
refuses?**

## ROUTED

1. **DE / coordinator — REVIEW 110 §(4) is WITHDRAWN as stated.** The cancel is not
   one-per-reference-generation by construction; the invariant that says so is keyed on
   `policy_gen`. The matched control samples reference generations. **Everything resting on
   "the cancel is the action" needs the reference-generation invariant stated and checked,
   not the policy one.**
2. **DE — `assert_book_scoring_code` must require the full set**, not accept any non-empty
   subset; `n_checked` should be compared to `len(SCORING_PATH_MODULES)` and the set itself
   derived from the recorded closure (REVIEW 123).
3. **Me, next round if there is one:** REVIEW 117 across coins and S30, and REVIEW 124's
   window identity driven rather than read.
