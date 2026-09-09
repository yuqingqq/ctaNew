# REVIEW 120 — if a 09-03 build started now, what would be wrong with the book?

**REV, 2026-09-09T08:23Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. Everything below is driven at the tip unless it says otherwise.

**THE SHORT ANSWER, IN THE THREE CATEGORIES.**

- **(a) BAKED IN AND INVISIBLE — I find NOTHING.** The two defects that could have been
  baked in are closed and I verified both at the code today: the era now travels (so the
  gaps travel), and the scoring is causal with the assembly per-row. **This is a "nothing I
  can find, and here is what I checked" answer, and I say so plainly rather than dress it as
  a list.**
- **(b) WOULD REFUSE — and a build is what CLEARS it, not what trips on it.** My REVIEW 118
  refutation needs one correction of scope: `require_book_declares_L` as armed refuses the
  four books that ALREADY EXIST, because they predate BE 101's builder. A book built now
  declares L — 09-07's rebuilt receipt already carries
  `.placement_latency.placement_latency_ms` and passes. **So arming is a reason to build, not
  a reason to wait.**
- **(c) WRONG BUT VISIBLE — three, all priceable.**

**AND THE ANSWER TO YOUR FIRST QUESTION IS YES: there is a second consumer, and it is
three of them.**

---

## 1. WHAT IS CLOSED, VERIFIED AT THE CODE TODAY, NOT TAKEN FROM THE LIST

**The era.** `be_daybook_build` now reads
`era = HER._era_or_refuse(fi, era_res["era"], "be_daybook_build")` — the day's resolved era,
not `None`. Driven end to end on the days I can resolve:

```
20260901: builder 'clob_v4_1'  DA ['clob_v4_1']  AGREE | 265 windows | 159 now carrying gaps
20260902: builder 'clob_v4_1'  DA ['clob_v4_1']  AGREE | 248 windows | 145 now carrying gaps
20260906: builder 'clob_v4_1'  DA ['clob_v4_1']  AGREE | 288 windows |  14 now carrying gaps
```

09-06's fourteen are the same fourteen I named in REVIEW 112 as lost. **REVIEW 112 is closed,
and REVIEW 114's two-channel collapse closes with it** — the windows DA deliberately left for
the gap channel will now arrive carrying their gaps.

**The scoring**: causal, first-crossing, faithful to the pre-fix aggregation (REVIEW 110,
driven on the engine and against the code at `c501824^`). **The null**: both endpoints'
draws persisted and the statistics re-derive (REVIEW 115). **The ledger**: its published
contract reproduces a real day's numbers (REVIEW 116).

## 2. (a) BAKED IN AND INVISIBLE — NOTHING I CAN FIND

What I checked, so the negative is worth something: the era and therefore the gaps (§1); the
scoring path's causality and the old/new aggregation equivalence; the cancel unit (one per
generation, now asserted); the null's draws and moments; the ledger's write-and-read
contract; the mask's own soundness and its interaction with the gap channel; the boundary
reader's staleness distribution. **Each of those was a candidate for a silent bake-in and
each is now either fixed and driven, or measured and bounded.**

**The one thing I cannot rule out is not in the book.** DE's claim that a generation's first
row starts exactly at its `t0` (measured on 09-04) has never been verified by anyone — I
flagged that in REVIEW 105 and it is still open. **If it fails, the BOOK is still correct**;
what breaks is three consumers, in §4. So it is not a bake-in.

## 3. (b) WOULD REFUSE — one, and it is cleared BY building

`require_book_declares_L` is armed in params v25. REVIEW 118 measured that 09-03..09-06 all
refuse `SETTLEMENT_BOOK_DECLARES_NO_PLACEMENT_LATENCY` and only 09-07 passes. **The scope
correction that matters for tonight: those four are the EXISTING books.** 09-07 passes
because it was rebuilt and its receipt carries the key. A 09-03 book built now, by the same
builder, would declare L and pass. **The armed guard refuses yesterday's books; it does not
block today's build. It is a reason to rebuild, not a reason to wait.**

## 4. YOUR FIRST QUESTION — YES: THREE CONSUMERS NOBODY HAS ENUMERATED

The shape change is in the assembly's **KEYS**, not only its values, and that is what nobody
swept. The old assembly was keyed `(slug, side, t0)` — one key per generation. The corrected
one is keyed `(slug, side, t_start)` — **one key per row**. Three DA modules test membership
with the OLD key shape and have **zero** shape-awareness:

| module | line | test |
|---|---|---|
| `da_elementwise.py` | 32 | `(s_, sd, float(g["t0"])) in gs` |
| `da_elem_grid.py` | 33 | same |
| `da_elementwise_hz.py` | 32 | same |

**Driven.** A fixture with two generations — one whose first row is at its `t0`, one whose
first row starts 3 s late:

```
corrected assembly keys: [100.0, 103.0, 503.0, 506.0]
generation 0 (t0 100.0): key present True   -> counts as SCORED
generation 1 (t0 500.0): key present False  -> counts as UNSCORED, though it has 2 scored rows
the assembly itself reports SCORED 2, ROWS_SCORED 4
```

**The assembly is right and the consumer reads it wrong**, silently, as a smaller population.
**BE already handled its own** — `be_generation_count_derivation` carries three shape-aware
mentions. **The DA three do not**, and their exposure is exactly the unverified 09-04 claim
in §2: if every generation's first row is at its `t0`, zero generations are affected; if not,
up to the 27 % of generations DE measured as having rows after their start.

**I could not measure the real rate**: every book on disk predates the causal scoring
(newest Sep 8; `c501824` landed 09-09T04:02), so no corrected assembly exists yet to count
against. **The first corrected book is the artifact that settles it** — which is an argument
for building, provided the three consumers are not run against it until they are swept.

## 5. (c) WRONG BUT VISIBLE — three, priceable

1. **09-03's settlement winner remains a DISAGREE** on `btc-updown-5m-1788469500`. The
   artifact will say so — `VERIFICATION_DID_NOT_AGREE`, `is_final_for_quotation: false` — and
   REVIEW 117 now explains it: a 2,000 ms stale boundary sample on a −14 ppm move. Visible,
   explained, and it makes 09-03 not quotable as final.
2. **The mask's exclusion will not travel into the receipt** (REVIEW 114): `n_slugs = 247`
   with nothing saying 40 windows were masked or by which artifact. Visible as a number,
   illegible as a population.
3. **If the run is a point estimate**, the settlement figures carry
   `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN` and have no control at all (REVIEW 115 §5). Honestly
   labelled; it means "the point estimate stands", nothing more.

## 6. YOUR SECOND QUESTION — WOULD ANYTHING MAKE A CORRECTED BOOK'S CANCELS UNTRUSTWORTHY?

**Nothing I can find, if gate item 4 lands under ruling (b) and is verified.** What I
checked, each driven rather than read:

- the decision is causal and cancels at the FIRST crossing — three cases on the policy
  engine, including the one where the old code cancels and the new does not (REVIEW 110);
- the old aggregation the delta compares against is the pre-fix stream event-for-event, from
  the code at `c501824^` (REVIEW 110);
- the unit is the ACTION — one cancel per generation, and since REVIEW 118 that invariant is
  asserted rather than inherited;
- the matched control samples CANCELS under ruling B, with the drawn sets persisted because
  the draw is data-dependent (REVIEW 113);
- the null's draws are persisted for both endpoints and its moments re-derive (REVIEW 115);
- gate item 4's abutting boundary is the one open hole, and ruling (b) — route by `gen` —
  removes the identity ambiguity rather than moving a threshold by one character.

**The residual is not in the cancels. It is in who reads them** — §4's three consumers — and
in the unverified `t0` claim they rest on.

## 7. SO: BUILD, WITH TWO CONDITIONS

Nothing I can find would be baked into a 09-03 book built now and invisible. The one guard
that refuses refuses *yesterday's* books. **Two conditions, both cheap:**

1. **Do not run `da_elementwise`, `da_elem_grid` or `da_elementwise_hz` against a corrected
   book until their membership test is swept** to the per-row key shape. One line each, and
   the fixture in §4 is their cell.
2. **Record the mask identity and `n_masked` in the receipt** (REVIEW 114), so the first
   corrected book's population is legible without re-deriving it.

And the first corrected book is what finally measures the `t0` claim that three consumers
and one of my own open items rest on — **which is itself a reason to build rather than to
wait.**
