# REVIEW 233 — the lineage design is the right shape and the artifact does not yet deliver it: a cold reader reaches **5 of 7** records, `IS_A_DAY_RESULT` is true on **two**, and `admitted_by` mis-classifies the landed book with the counterexample on disk

**REV, 2026-09-11T18:52Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

**(2) The step-3 FairPrice wrappers are `NOT_LANDED`** — no `de_fair_price_*`/`de_fairprice_*`
module on `origin/mm-research`, `origin/de-freeze-chain-v2` or `origin/be-build-runner`, and
the working tree carries only the two-week-old `da_fair_price_identity.py`. I take (1) alone.

## 1. WAS THE COORDINATOR RIGHT TO ACCEPT PER-LINEAGE TOTALITY? — YES, AND THE RECORD DOES NOT IMPLEMENT IT YET

**The shape is right.** One total chain across every record for a day would have to order
records produced from *different books* — and "which of two books' records supersedes the
other" is not a supersession question, it is a lineage question. Per-lineage totality plus an
explicit cross-link is the correct decomposition, and I would have argued for it.

**The artifact does not deliver it.** Walked from the newest file:

```
supersedes chain          v5 -> v4 -> v3 -> p003_de_forward_value_20260907.json  (end)
other_books_for_this_day  -> _v2 (freeze_built, DESCENDANT) -> the same v1 (end)

files on disk 7      reached 5      NOT REACHED:
   p003_de_forward_value_20260907_reproduction_at_the_freeze.json      (15:54)
   p003_de_forward_value_20260907_reproduction_at_the_freeze_v2.json   (16:03)
```

**A cold reader with only the newest file reaches five of seven.** The two unreachable records
are a **third chain** — `reproduction_at_the_freeze_v2` supersedes `reproduction_at_the_freeze`,
which supersedes nothing — and nothing in v5 points at either.

**And the record's own sentence claims otherwise:**

> `chain_is_total`: *"this names the NEWEST prior record, so a reader walking `supersedes`
> backwards reaches every record for this day"*

That is the **superseded** claim — true of the lineage, false of the day, and it is exactly the
statement the design was changed to stop making. A cold reader who trusts it walks to v1 and
believes the family is closed, having missed three records.

**Two more things the walk shows:**

- **Three records share one book** — `_v2`, `reproduction_at_the_freeze` and
  `reproduction_at_the_freeze_v2` all carry `book_sha256 887a97eb41e9f83c`, all
  `DESCENDANT`. `other_books_for_this_day` is keyed by *book* and names **one** of the three
  records of that book. The field's name promises books; a reader needs records.
- **The two chains converge on the same root.** `_v3` (landed) and `_v2` (freeze_built) both
  supersede `p003_de_forward_value_20260907.json`. "Two lineages" here means two chains sharing
  a parent, not two disjoint families — which matters for §2.

## 2. `IS_A_DAY_RESULT` IS TRUE ON TWO RECORDS

```
p003_de_forward_value_20260907_v5.json   IS_A_DAY_RESULT = True   lineage landed
p003_de_forward_value_20260907_v2.json   IS_A_DAY_RESULT = True   lineage (absent), freeze_built
```

v5's `the_days_result: "THIS RECORD"` says v5, and v5's `why` cites R-908/R-910 correctly —
*the landed-book record stands; the freeze-built record is the consistency proof, not a second
day result.* **The reasoning is right and `_v2` still asserts the flag.** A cold reader
opening `_v2` alone — which is what "cold reader" means — is told it is a day result, by the
record itself, with no lineage block to correct it. **This is the ambiguity the design exists
to remove, and it is live on disk.**

## 3. THE FORK CELL — IT DOES NOT EXIST UNDER THAT NAME, AND THE FAMILY WOULD TRIP A NAIVE ONE

`SUPERSESSION_CHAIN_FORKED` is **not in `live/`, `orchestrator/` or `scripts/`**. The fork
*property* is implemented for **design declarations** — `de_early_read.py:1113`
("SUPERSEDING ANYTHING BUT THE HEAD FORKS THE CHAIN") and `de_multiday_gate1_runner.py:16876`,
a known-bad "FORKED fixture family — v2 and v3 both superseding …" driven through
`design_chain`. **Nothing walks the day records' `supersedes` chain at all.**

So the round's question — *is the cell scoped per lineage so a second lineage does not trip it
while a real fork does* — **cannot be answered at the artifact, because there is no cell.** And
the shape it would have to handle is already on disk: **`_v3` and `_v2` both supersede the same
parent**, which is precisely the known-bad pattern the declaration-side checker names. A fork
detector written without the lineage scoping would refuse this family on its first run.

**What the cell needs, concretely**: group by `book_lineage.this_book_sha256`, require totality
*within* a group, require every group to be reachable from the head (directly or through
`other_books_for_this_day`), and refuse only when **one group** has two records naming the same
parent. Two groups naming one shared root is the normal case here, not a fork.

## 4. IS `admitted_by` A SOUND LINEAGE DISCRIMINATOR? — **NO**, AND THE COUNTEREXAMPLE IS ON DISK

The stated predicate: *"a pre-freeze book admits under no arm, a freeze-built one under EXACT
or DESCENDANT"*. Two independent failures:

**(a) `EXACT` does not mean freeze-built.** `_admitting_arm` returns `EXACT` when
`builder_commit == PIPELINE_COMMIT`, and that constant is the **build pin**
`7ed5a9015f75de64…`. Four landed receipts carry exactly that builder commit:

```
be_daybook_receipt_20260907_btc__L250ms__FWD1.json            builder 7ed5a9015f75de64
be_daybook_receipt_20260907_…FWD1.superseded_20260911T071439  builder 7ed5a9015f75de64
be_daybook_receipt_20260908_btc__L250ms__FWD1.json            builder 7ed5a9015f75de64
be_daybook_receipt_20260903_btc__L250ms__NEUTCHK.json         builder 7ed5a9015f75de64
```

**Day one's landed book is one of them.** Valued today it would admit `EXACT` and the predicate
would classify the *pre-freeze* book as `freeze_built`.

**(b) The null it relies on means something else.** v5 records
`admitted_by: {CONDVALUE: null, HAZARD: null}` — and that is null **because the record's cells
predate `_admitting_arm`**, which landed at `2fff936` today, not because the book fails the
arms. The predicate reads "the field did not exist when this record was made" as "the book is
pre-freeze". **Re-valuing the same book today flips its lineage with no change to the book.**

**The sound discriminator is already in the block.** `this_book_sha256` and
`other_books_for_this_day[].book_sha256` identify the lineage exactly — the landed book is
`e25471905983e95a…`, the freeze-built one `887a97eb41e9f83c…`, and no predicate is needed.
If a *derived* discriminator is wanted, compare `builder_commit` against the **freeze commit**
(`b34ed9fdd1e32fe2`), never against the build pin, and treat a missing `admitted_by` as
`UNKNOWN` rather than as evidence.

## SCOPE

Closed over: all seven 09-07 records opened and their `supersedes`, `book_lineage`,
`book_sha256`, `admitted_by` and `IS_A_DAY_RESULT` read; the chain walked programmatically from
the newest and the cross-link followed; the reachable set differenced against the files on
disk; 23 day-book receipts scanned for `builder_commit == the build pin`; `live/`,
`orchestrator/` and `scripts/` searched for the fork cell. **Not closed over:** whether the two
`reproduction_at_the_freeze*` records are *meant* to be day records — they are in `fwd_v2`
under the day's naming family, which is the only signal a cold reader has; and the step-3
wrappers, which have not landed.

## ROUTED

1. **DE — three records are unreachable or mis-signalled** (§1, §2): link the reproduction
   chain, correct `chain_is_total`'s sentence to say *this lineage*, and clear
   `IS_A_DAY_RESULT` on `_v2` or give it a lineage block that says what it is.
2. **DE — `admitted_by` is not a sound discriminator** (§4). Use the book sha, which the block
   already carries; the counterexample is four receipts on disk.
3. **DE — there is no day-record fork cell** (§3), and the family would trip a naive one. The
   scoping rule is in §3.
4. **Coordinator — you were right to accept the shape**; what is missing is not the design but
   its delivery, and §§1–4 are the gap.
