# Fair-value predictor — plan v1

**Status: DRAFT for the user's ruling. Nothing in it has been run.**
Written 2026-09-11 by the coordinator, while the descriptive valuations of
09-09..09-13 run. It does not alter the forward test, whose verdict is fixed
(R-910/R-911/R-914: both arms futile at day two).

---

## 1. Why this exists: the estimand was blind to fair value

Every candidate this programme has raced — the 69 screened, the two that
survived to the forward test — lives in the **cancellation layer**. The quoting
layer was held fixed: fair value = the book's own best bid/ask, plus a skew.

That is not an accident of the runs, it is built into the measurement. The
statistic is

    D = (arm's settled total) − (vanilla no-cancel book's settled total)

and **both legs quote off the same fair value**. Any error in that fair value
is common to the two legs and cancels out of D. The design could therefore
never see fair-value quality, only whether cancelling beat not cancelling.
It answered that question: no, on two days, not distinguishably from random.

The code carries the unused slot. `be_trajectory_export.py` takes a
`fairprice_estimator` argument that every call site passes as `None`, and names
an arm `CONDVALUE_X_SKEW_X_FAIRPRICE` that was never populated.

**One correction to the framing that prompted this plan.** "Vanilla performs
best" is true only *among what was tested*. The no-cancel book lost 4,438c on
09-07 and made 86,227c on 09-08. It beat both arms on both days; nothing
establishes that quoting mid-plus-skew is profitable.

## 2. What is already known about the settled event

This matters more than anything else here: you cannot price a binary whose
settlement rule you have not pinned down.

- **Venue: Chainlink, not Binance.** R-253, verified at the artifact: of 17,727
  market records, 17,727 mention Chainlink and 0 mention Binance.
- **Statistic: `S60(T) vs S60(t0)`, ties UP** — the 60-second TWAP at expiry
  against the 60-second TWAP at the window's open (amendment A2, Q-DA-142).
  The full-window mean reading (`meanS60[t0,T]`) is **refuted** at 86.9%.
- **Reproduction rate**: 99.8% on the original population against a
  pre-registered gate of ≥99.0%, and **99.85% pooled / 99.9% on |margin|>0.5bp**
  on a pre-registered fresh split (2026-08-24..27, n=8,022 windows, full
  coverage, disjoint from the original) — Q-DA-146. The equivalence control that
  licensed this was found decorative and was made executable (Q-DA-148); the
  numbers survived and strengthened.

**Consequence that shapes the whole design: `S60(t0)` is fully realized before
the window opens.** The strike is *known* at decision time. The unknown is only
`S60(T)`, and in the final 60 seconds even that is partially observed.

**Two things to settle before leaning on this (step 0 below):** the residual
~0.15% of windows the convention does not reproduce has never been characterised
(ties? boundary reads? oracle round timing?), and `CLAUDE.md` still carries the
older line that the settlement statistic is "contested and no form is asserted",
which is now stale against R-253/Q-DA-146 and should be reconciled or defended.

## 3. The data already exists, including the settlement inputs, live

- `data/pm_5min/prices/crypto_prices`, **`crypto_prices_twap_sixty`**,
  `crypto_prices_twap_thirty` — the settlement statistic's own inputs, collected
  in real time. This is the fact that makes a model-free fair value possible.
- `markets.jsonl` (46,538 records) — strikes, windows, rules text.
- `resolutions.jsonl` (46,535 records) — realized winners.
- The PM CLOB tape — the book's own price at every instant.
- Binance fine features already in the feature set: `bnf_midbps` at 10/25/50/100/250 ms,
  `bnf_imb_now`. Sub-second-reliable only from **2026-08-24 13:48:54 UTC**.

## 4. Hypotheses, ranked by how cheaply they die

**H1 — the book misprices the mechanics of the settlement statistic.**
The strike is known at `t0`; the terminal statistic is a 60-second *average*,
not a point; in the last minute part of that average is already realized. A fair
value computed from those three facts is **model-free** — no fitting, no
training window. If the book's mid differs from it systematically, that is an
edge with an explicit mechanism. This is the hypothesis to test first because it
requires no predictor at all.

**H2 — Chainlink's feed is stale relative to spot in a knowable way.**
Settlement runs on a feed that updates on deviation/heartbeat, not continuously.
When the oracle has not printed and Binance has moved, the direction of the next
oracle print is partly known. Observable, and the staleness is measurable from
the collected feed.

**H3 — a fitted predictor beats the mid incremental to it.**
Only worth reaching if H1 and H2 leave residual structure. This is the one that
needs a training window, a freeze, and multiplicity discipline.

## 5. Step 0 — reconcile the settlement claim (hours, no lock)

Deliverable: a declaration naming the settlement rule, its reproduction rate on
a fresh window, and the **characterisation of the residual** — for every window
the convention fails to reproduce, which of {tie, boundary read, oracle round
timing, data gap} explains it, counted, with exclusions as statuses.
Falsifier: recompute settlement for a fresh, declared window and match
`resolutions.jsonl` winner-for-winner; any unexplained mismatch class over a
declared threshold **stops the programme here**.
Also: reconcile or defend the stale `CLAUDE.md` line.

## 6. Step 1 — the ceiling, measured without a model (a day, no lock)

**Population**: a declared window strictly **before 09-07** — 08-26..09-06 is the
natural choice (08-20..08-25 are consumed for the harmful-fill line; 09-07..09-13
are being consumed now by the descriptive valuations). Declare it before looking.

**Unit**: one row per decision instant per market (the same decision-time unit the
programme already uses), carrying: the book's mid, the model-free fair value from
H1, the Chainlink staleness state from H2, time remaining, and the realized winner.

**The statistic**: log-loss (and Brier) of the book mid against settlement, and the
**incremental** improvement from H1's fair value and H2's staleness state. Rule 9:
skill is reported incremental to the book's own price, never against a base rate.
Clustered by UTC day; point estimate only below five complete days, and say so.

**Declared before the data is read**: the minimum incremental improvement that
would justify step 2, and the arithmetic translating a probability edge of that
size into expected value per contract **at the quoted spread and fees** — because
an edge smaller than the spread is not an edge, which is how the Amihud line died
in the CTA programme.

**Most likely outcome, stated in advance: no room.** The mid is the crowd's price
of a 5-minute event; it should be close to right. That is a result, cheaply got.

## 7. Step 2 — the predictor, only if step 1 clears

- **Form first, not features**: the theoretical price from H1 blended with the
  mid, two parameters. A big model is the wrong first candidate because its
  failure teaches nothing.
- **Point-in-time**: every input read at knowledge time (`recv_ns`); era purity
  (sub-second data only after 2026-08-24 13:48:54Z).
- **Frozen as a commit** (rule 12): builder file committed, hash and commit ref
  in the receipt, declared nulls inside the receipt, **and the count of
  candidates in the race recorded at freeze time**.
- **Candidate count is a design variable, not an afterthought** — see step 4.

## 8. Step 3 — evaluate at book level, never through the replay first

Calibration and log-loss improvement over the mid on held-out days. Only after
that does the economic question arise, and then as a **quoting change on the
reference path** (fair value moved, skew unchanged).

**Note the cost**: changing fair value changes the reference path, so tonight's D
numbers do not transfer. A new baseline must be built and declared. The old
vanilla book stays as provenance, not as a comparator.

## 9. Step 4 — the forward test, with the multiplicity fixed in advance

This is the design correction that tonight's result paid for, and it is the most
valuable line in this plan.

The forward test that just concluded could not have passed even with a perfect
run: seven unanimous days attain p = 0.015625; times the Holm m=2 that was
declared, 0.031 — a pass; times the **69 candidates actually screened**, 1.078 —
impossible. The design's minimum attainable p exceeded 1/69 before a single day
was valued.

So, before any fair-value forward test begins, state and satisfy:

    (minimum attainable p under the design) × (number of candidates screened) < α

which means either **few candidates** (H1 is one candidate, which is the point of
starting there) or **more power** (more days, or a within-day unit that admits a
finer statistic than a day-cluster sign test). Declare which, with the arithmetic,
before the first day is valued.

## 10. Kill criteria

- Step 0's residual is unexplained above the declared threshold → **stop**.
- Step 1's incremental improvement is below the declared minimum → **stop**, and
  the finding is "the book's mid is not improvable on free data at this horizon",
  which is worth writing down.
- The edge exists but is smaller than the spread plus fees → **stop**; report it
  as real-but-unharvestable, the honest form the CTA programme had to learn.
- Any edge that requires reacting faster than the measured latency **L** is valued
  only after `t + L` (rule 7), or it is not valued at all.

## 11. What this plan does not do

It does not revisit the cancellation layer, it does not alter the forward test's
verdict, and it does not touch any frozen module inside the current population.
Everything in steps 0 and 1 is read-only against existing data and needs no
heavy lock, so it can run alongside the remaining descriptive valuations.
