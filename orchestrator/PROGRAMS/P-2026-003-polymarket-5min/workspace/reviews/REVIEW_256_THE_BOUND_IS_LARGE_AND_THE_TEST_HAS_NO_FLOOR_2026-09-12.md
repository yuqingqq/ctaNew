# REVIEW 256 — the bound is comfortably large, and the test has no magnitude floor

REV round 219. Filed 2026-09-12T00:45:04Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 0. The §7/§9 adjudication is already delivered

**I ruled in REVIEW 255, landed at the canonical ref (`e1ca1a8`, count 1)
before this round's dispatch: CONFIRMED — `fee_rule` in §7 means DECLARED, not
KNOWN.** All three of your break-attempts fail and so does a fourth of mine.
DA's predicate should unblock on that filing; nothing further is owed from me
on it except the (c) check when the edit lands.

## 1. THE ANSWER: C1 is NOT arithmetically dead. The bound is large.

**I am not reaching for the conclusion you offered, because the arithmetic does
not support it, and I am stating that with the same confidence I would have
stated the other.**

### 1.1 The derivation, computed rather than asserted

Identity `p = ½(bb+ba)`. C1 is a convex combination of the same two prices, so
`C1 = p + d` with `|d| ≤ (ba−bb)/2 = s/2`, tight as one side's size → 0.

Log loss on the realised outcome with assigned probability `q` is `−ln q`.
Under **perfect foresight** every deviation points at the realised outcome, so

    improvement per action  =  −ln q + ln(q + d)  =  ln(1 + d/q)

This grows with `d` and shrinks with `q`, so the upper bound takes `d = s/2` and
the smallest `q` nature can hand us at that state, `q = min(p, 1−p)`:

    improvement  ≤  ln(1 + (s/2) / min(p, 1−p))        per action

— an upper bound over **both** the size configuration and the outcome.

**And it is structurally capped at ln 2.** Since `p = bb + s/2` and
`1−p = (1−ba) + s/2`, with `0 ≤ bb ≤ ba ≤ 1` we get `q ≥ s/2`, hence
`(s/2)/q ≤ 1` and the bound never exceeds `ln 2 = 0.693147`. Checked on every
state: `max ≤ ln 2` holds exactly.

### 1.2 The measurement

Day 2026-09-09, `price_change.best_bid`/`best_ask` (never `book` snapshots —
CLAUDE.md's rule; snapshots are p90 6.2 s stale).

| | states | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| BTC | 2,544,510 | **0.0823** | 0.0202 | 0.1823 | 0.6931 | 0.6931 |
| ETH | 808,372 | **0.0584** | 0.0220 | 0.1054 | 0.6931 | 0.6931 |
| both (40 files) | 4,973,558 | **0.0745** | 0.0202 | 0.1335 | 0.6931 | 0.6931 |

**Maximum attainable mean per-day `delta_LL` for C1 under perfect foresight:
≈ 0.06–0.08 nats per action.** That is not a small number — log loss at
`p = 0.5` is 0.693 nats, so the ceiling is roughly a **10% relative**
improvement.

Positive controls, run before the measurement: the `(0.01, 0.99)` artifact book
gives 0.683 nats and the tight `(0.495, 0.505)` book gives 0.00995 — the measure
separates the case where the bound must be large from the one where it must be
tiny. **Excluded:** every coin but BTC/ETH; non-`price_change` events; states
with an unparseable side; states with `min(p,1−p) = 0` (reported separately —
there were **none**, 0 of 4,973,558).

## 2. AND THE COMPARISON YOU ASKED FOR CANNOT BE MADE — for a reason that matters more

You asked me to compare the bound to "what the exact sign test can DETECT at
G=10". **The exact sign test has no magnitude resolution at all.** It tests the
*sign* of `delta_LL_g`, not its size. A candidate positive on ten of ten days by
`1e-9` nats yields `p = 2/2^10 = 0.001953125` and passes, exactly as one
positive by 0.08 nats would.

I checked §8's four adoption conditions for a magnitude floor and there is none:

1. Holm-corrected `p < 0.05` on the primary increment — sign-driven;
2. mean and median `delta_LL_g` **positive** — sign, not size;
3. the 95% native-coverage gate — about coverage, not effect;
4. population, timestamp, complement and reconciliation predicates — structural.

The only magnitude-sensitive clause anywhere is the tie rule — *"an exactly zero
daily increment is a reported tie and is excluded from the sign count"*, with
≥8 nonzero required — and **exactly zero** needs `bid_size == ask_size` at every
action of the day. (I cannot measure how often that holds: `price_change`
entries carry `best_bid`/`best_ask` but not the sizes *at* the touch. Naming the
limit rather than estimating around it.)

**So the finding is the opposite of the one you were braced for, and I think it
is the more serious one.** Because the test is sign-only:

> **C1 can pass §8 with an effect of any size above zero.** Its deviations are
> tiny — half a tick, 0.005 at p90 — but they are not zero, so its daily
> increments will be signed and nonzero, and ten favourable signs pass. A pass
> would then carry C1 into §9's economic clock on an effect that may be
> economically indistinguishable from nothing.

The plan specifies a minimum *sample* (§8's ≥200 permutations, satisfied at
1,024) and no minimum *effect*. Reliability rule 6 requires the null to be
declared in advance with its sample; nothing requires the alternative to be
large enough to matter. **That gap was invisible until the bound was computed,
and it is fixable before the clock in the same way the futility rung was: state
a minimum economically meaningful `delta_LL` now, as a declared field, and
report the observed effect against it.** I am not proposing a value — choosing
one after seeing tonight's numbers would be rule 11 — but it should be declared
by whoever owns the estimand, before day one.

**What this does NOT establish.** It does not say C1 will pass, or that its
deviations lean the right way; perfect foresight is a ceiling, not a forecast.
And it says nothing about C2, whose reported 0.234 coverage against a 95% gate
is another seat's measurement that I have not verified.

## 3. (i) The fee sentence, in the words you asked for

> **`fee_rate_bps` has never been observed on a charged fill.** The five
> transactions the chain charged appear **zero** times in our tape across
> 2026-09-04..09-06, on a query whose positive control finds 724,950 / 532,140 /
> 557,455 `transaction_hash` lines in the same files — so the zero is a measured
> absence, not a failed search. Across three full days the field takes the
> single value `"0"` in **1,881,868 of 1,881,868** occurrences, with no
> exception. The field is **populated and non-discriminating**: it reports the
> same value for the fills the chain charged at 9.9% and 49.5% as for every
> other fill. A field that cannot separate a charged fill from an uncharged one
> cannot identify a fee schedule, and §9 permits a zero fee only where the
> receipt identifies the supporting rule.

## 4. (ii) The `M_IS_TWO_FOREVER` caveat, as an exact sentence

> **m = 2 is frozen and remains 2.** It counts as two candidates a baseline and
> a bounded perturbation of that baseline. C1 reads the **same single book
> event** as Identity by construction — `de_fair_price_wrapper`: *"ONE book
> event, so C1 cannot read a different event from Identity"* — and is a convex
> combination of the same two prices, so `|C1 − Identity| ≤ spread/2` exactly,
> measured at **0.005 at the 90th percentile** of 9.4 million observed states
> and capped structurally at `ln 2` in log-loss terms. The Holm correction
> therefore spends power against a candidate that was never independent of the
> baseline, and buys no protection against the dependence that is actually
> present. **C2 is cross-venue and is not subject to this caveat.** m is not
> reduced: shrinking it after measuring the coupling would be selection on the
> outcome.

## 5. Owed

- (c) from REVIEW 255, at the artifact, when DA's `fee_rule` edit lands.
- The full-day 576-file Identity run (REVIEW 254 §6), if it disagrees.
- `limits[2]` still unchecked; REVIEW 247 Part 1 awaits a resolver; 246 and 245
  items remain open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); 09-11
  closed and still unread by me.
