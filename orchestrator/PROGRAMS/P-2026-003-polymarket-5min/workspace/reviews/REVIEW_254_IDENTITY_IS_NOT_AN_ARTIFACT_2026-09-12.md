# REVIEW 254 — Identity is a real price; C1 is Identity ± half a tick

REV round 217. Filed 2026-09-12T00:35:42Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## THE HEADLINE: the validity threat I named is NOT present

**My hypothesis is refuted by the data, and that is the useful outcome.** The
artifact share at decision-capable states is **0.17% (BTC) / 0.58% (ETH)**, the
quoted spread is **one tick at the median and at the 90th percentile**, and
there were **zero** one-sided and **zero** crossed states in 9.4 million
observations. `Identity` is a tight, real price essentially always. §8's
comparison is not a comparison against a coin flip the book never offered.

This clears the threat **before** a validation day is consumed, which is what
you asked for.

## 1. Method, stated before the numbers

**Source rule followed.** Read from `price_change.best_bid`/`best_ask`, never
from `book` snapshots — CLAUDE.md's own rule, snapshots are p90 6.2 s stale.
Using snapshots would have measured staleness and called it spread.

**Pre-stated parameters**, fixed before any result was seen:

    EPS    = 0.01   near-0.5 band (one tick at the coarse observed tick size)
    S_MAX  = 0.10   wide-spread threshold
    ARTIFACT := one-sided (missing best_bid or best_ask) OR spread >= S_MAX
    Identity  = 0.5 * (best_bid + best_ask)

**Near-0.5 is reported, never used to define the class.** A 5-minute up/down
crypto market is *genuinely* near 0.5 much of the time; treating that as
evidence of an artifact would have manufactured the finding I was looking for.
The artifact test is about whether the book can *support* the price, not about
where the price is.

**Positive controls, required to pass before the run** (§7k.2):

    (0.62, 0.63) -> CLEAN,    Identity 0.625   ✓
    (0.01, 0.99) -> ARTIFACT, Identity 0.500   ✓
    (0.49, 0.50) -> CLEAN                      ✓

A tight book at 0.49/0.50 must read CLEAN even though Identity is 0.495 — the
control that would catch me conflating "near 0.5" with "artifact".

**Excluded, stated:** every coin but BTC and ETH (`SCOPE_COINS`); all
non-`price_change` events; entries with an unparseable best bid or ask (counted
as ONE_SIDED, not dropped); markets whose slug carries no window-start integer.
Day 2026-09-09. Samples of 40 BTC and 25 BTC/ETH files; the full 576-file
BTC+ETH run is still going as I file and I will report it if it disagrees.

## 2. The measurement

| | BTC (40 files) | ETH (25 files) |
|---|---|---|
| `price_change` entries | **7,860,202** | **1,560,672** |
| CLEAN | 7,846,675 — **99.83%** | 1,551,650 — **99.42%** |
| **ARTIFACT** | 13,527 — **0.172%** | 9,022 — **0.578%** |
| ONE_SIDED | **0** | **0** |
| CROSSED | **0** | **0** |
| spread p50 | **0.0100** | **0.0100** |
| spread p90 | **0.0100** | 0.0300 |
| spread p99 | 0.0400 | 0.0800 |
| spread max | 0.70 | 0.69 |
| near-0.5 **and** artifact | 502 — **0.0064%** | 156 — **0.0100%** |
| near-0.5 and CLEAN | 282,230 — 3.59% | 74,650 — 4.78% |

**At the decision instant** (last state at or before T−60, the regime boundary
in the estimand): **130 of 130 (slug, token) pairs CLEAN**, zero artifacts, on
both coins.

## 3. The partition you asked for, and why I cannot compute it

You asked me to split the population into artifact-suspect and clean and report
the challengers' advantage in each. **I cannot, and the reason is the finding:
the artifact partition at decision instants is empty in this sample.** There is
nothing to compare against.

**And I will not overstate that.** 130 decision instants has low power: at the
measured all-states rate of 0.17–0.58%, the expected number of artifact
decision instants in 130 is **0.2 to 0.75**, so observing zero is exactly what
that rate predicts and does not sharpen it. **The well-powered number is the
all-states one — 9.4 million observations putting the artifact rate under
0.6%** — and the decision-instant result is consistent with it rather than
independent evidence.

So the honest statement: *the artifact share is under 1% by a well-powered
measure, and a per-partition advantage comparison is not computable at that
rate without a far larger decision-instant sample.* If anyone wants the
partition anyway, it needs the canonical action population, not a proxy.

## 4. (b) C1 cannot differ from Identity by more than HALF THE SPREAD

**The bound is exact and it is by construction.** At the code:

    Identity : value = 0.5 * (best_bid + best_ask)              (line 337)
    C1       : v = (best_bid*ask_size + best_ask*bid_size)/(bid_size+ask_size)

C1 is a convex combination of the same two prices — the module enforces it with
an explicit invariant that refuses `OUT_OF_RANGE` if the result escapes
`[bid, ask]`. A convex combination of the endpoints lies in the interval, and
Identity is its midpoint, therefore

    |C1 − Identity| ≤ (best_ask − best_bid)/2 = spread/2

with equality approached as one side's size → 0. **There is no book on which
C1 can differ from Identity by more than half the quoted spread.**

Putting the measurement into the bound:

| | BTC | ETH |
|---|---|---|
| max \|C1 − Identity\| at p50 | **0.0050** | **0.0050** |
| at p90 | **0.0050** | 0.0150 |
| at p99 | 0.0200 | 0.0400 |
| at the observed max spread | 0.35 | 0.345 |

**C1 is Identity ± half a tick over at least 90% of the population.** Your
reading is right and this is the number under it: `m = 2 forever` counts as two
a baseline and a deterministic perturbation of that baseline bounded by 0.005
almost everywhere. The Holm correction costs power against a candidate that was
never independent, and buys no protection against the dependence that is
actually there.

**Do not change m** — agreed, and for your reason: it is frozen, and shrinking
it after seeing the coupling is selection on the outcome. What the freeze's
`M_IS_TWO_FOREVER` clause deserves is the **stated caveat**: that C1 and
Identity read one book event by construction (the wrapper says so in its own
comment) and that C1's deviation is bounded by half the spread, measured at
0.005 at p90. A reader who sees "m = 2" and infers two independent candidates
is reading something the artifact does not say and the code contradicts.

**One interaction worth recording, and then dismissing.** C1's scope to differ
is *largest exactly where Identity is weakest* — the bound is spread/2, and a
wide spread is what makes Identity an artifact. So the two questions are
mechanically coupled. The coupling is real and **immaterial**: the population
where it bites is the 0.17–0.58%.

## 5. The owed tx result, now valid — and it closes the fee thread

REVIEW 252 §1 and REVIEW 253 §1 left this inconclusive because the positive
control failed. **The control has now passed** on the same files:

    20260904  transaction_hash lines = 724,950
    20260905  transaction_hash lines = 532,140
    20260906  transaction_hash lines = 557,455

The query finds transaction hashes in abundance in exactly the files it
scanned, so the original result is a **valid absence**: **none of the five
charged transactions appears in our tape**, across 09-04..09-06, scanning only
`raw/*.jsonl.gz` on those three days.

So `fee_rate_bps` has **never been observed on a charged fill**. It is not
"wrong on the charged fills" — it is **untested on every charged fill we know
of**, which is a cleaner statement than either reading I offered in REVIEW 252
and supports the same conclusion more directly.

## 6. Owed

- The full-day 576-file run, reported if it disagrees with §2.
- `limits[2]`, the role-assignment line — still unchecked, and both its
  neighbours are wrong.
- REVIEW 247 Part 1 awaits a resolver; 246 and 245 items remain open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  closed and still unread by me.
