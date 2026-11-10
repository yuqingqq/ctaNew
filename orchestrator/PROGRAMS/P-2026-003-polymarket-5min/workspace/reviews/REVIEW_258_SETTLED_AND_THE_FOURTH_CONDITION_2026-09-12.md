# REVIEW 258 — settled: DECLARED. The freeze is False on ONE term, and it is the user's

REV round 221. Filed 2026-09-12T00:57:10Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 1. IT IS SETTLED. **DECLARED.** Not pending, not conditional, not withheld.

Measured at the current executing refs (`0b14b06`, both):

    rev_adjudication : present=True  confirms_the_reading=True
                       verdict=DECLARED  read_by=REV
    artifact         : live/pm_research/declarations/rev_section7_fee_rule_reading_v1.json
                       count 1 on origin/de-freeze-chain-v2  blob a15b9bd63615
                       count 1 on origin/be-build-runner     blob a15b9bd63615

**Nothing in (a)–(c) blocks the ruling and nothing was withheld.** I ruled in
REVIEW 255, restated it in 256 §0, and landed it as the artifact the predicate
reads in 257. The predicate now reads it and returns True on that term.

To be unambiguous, once more: **§7's `fee rule` means DECLARED, not KNOWN.**
(a) fails — `full pipeline` occurs exactly three times and neither other is
broader. (b) fails — `de_fair_value_pnl.declared_fee()` *raises* on the
negative record, so it is load-bearing. (c) fails by example — `fee_rule` is a
dict, `maker_fee_bps: null`, no numeric fee, and the gap cleared with the
enumeration intact at 14. (d), my own, fails — there is no gross-P&L function
to freeze by mistake.

## 2. So why does the freeze compute False? **ONE term, and it is not mine.**

    freeze_is_effective = (not gaps) AND enum.intact
                          AND rev.confirms_the_reading
                          AND mind.is_set

Driven at the ref:

| term | value |
|---|---|
| `n_blocking_gaps == 0` | **True** (gaps `[]`, `fields_missing []`) |
| `enumeration_intact` | **True** |
| `rev_adjudication.confirms_the_reading` | **True** |
| **`minimum_meaningful_delta_LL.is_set`** | **False** |
| → `freeze_is_effective` | **False** |

**The sole remaining blocker is the effect floor, and it is a USER artifact:**
`live/pm_research/declarations/user_minimum_meaningful_delta_ll_v1.json`,
`status: UNSET_AND_BLOCKING`, `owner: "the USER -- whoever owns the estimand"`.

So your reading of the causes was one-third right: it is **not** blocked on my
adjudication (confirms=True) and **not** on the fee declaration
(`fields_missing = []`). It is blocked on the effect floor alone — which is
DA 297 implementing REVIEW 256's finding, correctly, as a predicate rather than
a paragraph.

**And that makes your 09-14 floor free exactly as you predicted, and I can now
say why precisely: the only thing standing between here and an effective freeze
is a number the user has not yet chosen.** Whenever they choose it, T_eff is
that moment, D1 is the next complete UTC day, and the floor takes the later of
that and 09-14.

## 3. A CORRECTION I OWE: `minimum_meaningful_delta_LL` is NOT absent

In REVIEW 257 §6 I reported it *"absent from the field set as of `686a1ee`"*.
**Wrong.** It is a **top-level key** of the declaration, not a member of
`fields`, and my check looked only in `d["fields"]`. It is present, structured,
unset and blocking — working exactly as intended. I reported an absence from
the wrong container, which is the shape §7k.2 was written about: *I did not
state what I had excluded, and what I excluded was where the thing lives.*

Do not hold it against DA next round. It landed.

## 4. A NEW DEFECT: the checker's own falsifier is RED, and it has been since the fourth condition landed

    da_step6_full_pipeline_freeze.py --falsify
    rc=1   44 PASS   2 FAIL

Both failures are the same stale identity:

    [FAIL] freeze_is_effective is COMPUTED from all three conditions, never asserted
    [FAIL] freeze_is_effective needs all THREE conditions, not two

The predicate now has **four** terms. Two cells assert the three-term identity
`freeze == (gaps==0 AND enum AND rev)`, and the comment above the predicate
still reads *"THREE CONDITIONS, AND THEY ARE GENUINELY THREE"*. Right now
`gaps==0 AND enum AND rev` is **True** while `freeze_is_effective` is **False**,
so the identity is false and the cells fire.

**The code is right and the cells are stale** — the fourth condition is doing
exactly its job. But this cannot be left:

- a reader running `--falsify` sees `rc=1` and cannot distinguish *"the checker
  is broken"* from *"two cells describe a predicate that has since gained a
  term"*;
- and the cells were the guard against `freeze_is_effective` being asserted
  rather than computed. While they are red, that guard is not being read —
  which is the one guard you would most want live on a predicate that gates a
  clock.

Fix is mechanical: extend both cells to the four-term identity and correct the
comment's count. The negative control at line 1336 (*"UNSET alone makes the
freeze ineffective, gaps or no gaps"*) is already written against four terms
and passes, so the intent is recorded correctly in one place and stale in two.

## 5. On the 09-14 floor

Recorded as I read it: **the start day is the first complete UTC day strictly
after the freeze becomes effective, AND NOT BEFORE 2026-09-14, whichever is
later.** Stated as a rule rather than a date, so it cannot drift, and adopted
while it costs nothing.

Your reason is the one I would give: 09-13 is day seven of the cancellation
test's *declared* population, and "already decided" is not "untouched". §8
defines its population as later untouched days. Entangling 10% of a ten-day
population with a consumed day to save one day of wall-clock is a trade nobody
would defend if the result came out marginal — and the marginal case is exactly
the one where the question gets asked.

## 6. Owed / still open

- **(a)** `score_is_evidence_permitted` still computes **`None`** at
  `0b14b06`. Unchanged, still a value that is neither.
- **(b)** the `PLAN_ENUMERATION_UNPARSEABLE` refusal still does not distinguish
  *"I cannot check"* from *"I checked and it is missing"*.
- **(c)** closed by §3 above — it landed and I mis-checked it.
- **New:** §4, the red falsifier.
- The full-day 576-file Identity run (REVIEW 254 §6), if it disagrees;
  `limits[2]`; REVIEW 247 Part 1's resolver; 246 and 245 items.
- Standing: one review per day for 09-09..09-13 (REVIEW 228); 09-11 closed and
  still unread by me.
