# REVIEW 200 — exactly ONE field differs, it is NOT provenance and NOT the four commits: the live settlement oracle grew between the two runs. The combine is sound, and the waiver must not say "provenance".

**REV 160, 2026-09-11T10:56Z** (clock read separately). Read-only, no lock. Predicate read at
`~/ctaNew-wt-deval/live/pm_research/de_forward_evaluator.py:439–466`; both cells read from
`data/pm_5min/derived/fwd_v2/`.

## (1) THE PREDICATE COMPARES ELEVEN FIELDS. **ONE DIFFERS.**

Canonicalised as sorted JSON per arm and compared for set-size 1:
`book_sha256`, `winner_source`, `zero_model_cancel_baseline_total_cents`,
`robustness_leg.zero_model_cancel_baseline_total_cents`, `book_receipt`, `params_pin`,
`input_verification`, `score_neutrality_certifications`, `score_delta_max_certified`,
`forward_book_margin_guard`, `producer`.

```
SAME   book_sha256                 SAME   book_receipt          SAME   params_pin
SAME   zero_model_cancel_baseline_total_cents                   SAME   input_verification
SAME   robustness_leg.zero_model_cancel_baseline_total_cents    SAME   producer
SAME   score_neutrality_certifications   SAME   score_delta_max_certified
SAME   forward_book_margin_guard
*DIFF  winner_source
```

**`winner_source`, and nothing else:**

| | CONDVALUE (10:12:15Z) | HAZARD (10:52:53Z) |
|---|---|---|
| path | `data/pm_5min/resolutions.jsonl` | same |
| **sha256** | **1e9500d0daed5a35…** | **4f0fc39a7ff6f978…** |
| n_records | **45,877** | **45,932** |
| n_closed_records | 45,852 | 45,907 |
| n_slugs | 45,852 | 45,907 |
| method / settle_up_cents | identical | identical |

## (2) **NO. IT IS NOT PROVENANCE — AND IT IS NOT THE FOUR COMMITS.**

**Tree HEAD, closure listing and dirty flag are not among the eleven fields.** DE's four
commits (`68e7d23 → … → 577dad2`) are **irrelevant to this refusal**; the predicate never
looks at the tree. **The cause is outside the worktree entirely.**

`winner_source` is **the settlement oracle** — the file that decides which side won every
window, the direct input to `settled_total`. **The two arms were valued against two different
byte-states of it**, because `resolutions.jsonl` is **live and append-only** and gained **55
records in the 40 minutes between the two runs.**

**BUT THE DIFFERENCE IS INERT FOR THIS DAY, AND THAT IS MEASURED, NOT ASSUMED:**

```
resolutions.jsonl now : 45,989 records (still growing — 57 more since the HAZARD run)
last 120 records      : 2026-09-11 windows, ALL of them
records for 2026-09-07: 2,016  =  288 windows x 7 coins  -- COMPLETE
```

**The file grows only at today's head.** 09-07 closed on 2026-09-08 and its 2,016 records
have been complete and immutable since. *(One inference, labelled: I can read only the
current file, so "the 55 added records are all 09-11" is measured on the tail, and "09-07's
slice is identical in both versions" follows from the additions being 09-11-only plus
append-only growth — it is not a byte comparison of the two historical versions, which no
longer exist.)*

**So: the two arms consumed an identical settlement slice for 09-07, from two different whole-file digests.**

## (3) THE COMBINE IS SOUND — WITH THREE CONDITIONS, AND THE FIRST IS ABOUT THE WORDING

**The design does NOT require both arms from one tree state.** The predicate compares
**inputs**, not tree state; no tree field is in it. A superseding in-band combine over the two
existing cells is sound **provided:**

1. **THE WAIVER MUST NOT SAY "PROVENANCE-ONLY". It is false.** `winner_source` is a
   settlement input. The waiver must name the measured fact:
   > *the two cells' `winner_source` digests differ (`1e9500d0…` / `4f0fc39a…`) because
   > `resolutions.jsonl` is append-only and gained 55 records between 10:12:15Z and
   > 10:52:53Z; every record added is a 2026-09-11 window; 09-07 holds 2,016 records
   > (288 × 7, complete) in both versions, so the slice this day consumes is identical.*
   **A waiver that names the field without the slice measurement waives a settlement input on
   a category error.**
2. **The per-09-07 winner identity should be asserted, not argued.** The cheapest closure:
   both cells already carry `winner_source.n_slugs`; add — or compute once for the waiver —
   **the digest of the 09-07 SLICE** (the 2,016 records), and show it equal. That turns §2's
   one inference into a measurement and costs a single pass over a text file.
3. **AND THE AUTHOR'S MESSAGE IS WRONG IN THIS INSTANCE.** The refusal reads *"{day} arm
   cells disagree on their **book, zero-cancel baselines or guard inputs**"* — and on all
   three of those the cells **agree**. `winner_source` is in the compared set but in none of
   the three named classes. **A reader resolving the cause from the message gets a false
   one** (rule 30's shape). The message should enumerate the differing field, which the code
   already knows.

### THE READING I TAKE FROM THE AUTHOR'S OWN WORDS

The docstring is thin — *"Load only cells that reconcile to the V2 runner checkpoints"* — so
the refusal string is the intent statement, and it enumerates **book / baselines / guard
inputs**: a *same-inputs* predicate, not a *same-tree* one. **`winner_source` was swept in
correctly and described incorrectly.** On the author's enumerated intent, these two cells
pass; on the author's implemented set, they fail; **and the implemented set is the better
one**, because a moving settlement oracle is exactly the kind of input that should stop a
combine until someone looks.

## THE THING THAT OUTLASTS THIS DAY, AND I WOULD FIX IT BEFORE DAY TWO

**`resolutions.jsonl` is live. Any two arms valued more than a few minutes apart will differ
on `winner_source` — every remaining day, forever.** Waiving per-day signs the programme up
to six more waivers on a settlement input, each needing the §3.2 slice check to be honest.

**Pin it instead:** value both arms against a **frozen snapshot** of the oracle (or record and
compare the **per-day slice digest** rather than the whole-file digest). **That converts a
recurring waiver into a non-event**, and it is a change to the runner's inputs rather than to
any pinned computing module.
