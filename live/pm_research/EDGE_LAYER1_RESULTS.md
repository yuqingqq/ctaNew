# EDGE_LAYER1 — fill quality at a fixed horizon (`edge_l1_v1`)

**Verdict: `HORIZON_DEPENDENT`.** Research only, not decision eligible.

Probe `edge_layer1.py` (28 selftest checks) · receipt
`data/pm_5min/derived/edge_layer1_v1.json` · 30 windows/coin, `JOIN_BBO`
two-sided, 5 shares, 250 ms knowledge lag.

**Provenance: `source_days = 20260819, 20260820, 20260821, 20260822` (n=4).**
Every previously published edge figure was computed on **three** days. Numbers
here are not comparable with those without accounting for the extra day.

---

## The headline: spread capture is positive, and adverse selection is larger

btc and eth, all fills, cents per share:

| coin | h | n | markout | CI95 | spread captured | drift |
|---|---:|---:|---:|---|---:|---:|
| btc | 5 | 10,294 | **−0.532** | [−0.797, −0.287] | +0.642 | **−1.175** |
| btc | 15 | 10,122 | **−0.476** | [−0.765, −0.178] | +0.639 | −1.115 |
| btc | 30 | 9,714 | **−0.637** | [−1.047, −0.216] | +0.636 | −1.273 |
| btc | 60 | 8,077 | −0.085 | [−0.834, +0.633] | +0.612 | −0.697 |
| eth | 5 | 1,999 | **−1.243** | [−1.726, −0.759] | +0.778 | **−2.021** |
| eth | 15 | 1,966 | −0.586 | [−1.284, +0.089] | +0.738 | −1.324 |
| eth | 30 | 1,932 | −0.672 | [−1.393, +0.059] | +0.734 | −1.406 |
| eth | 60 | 1,752 | **−1.609** | [−2.479, −0.807] | +0.695 | −2.305 |

**The decomposition inverts the sign.** Spread capture is positive and stable
everywhere — +0.61 to +0.64 ¢ on btc, +0.70 to +0.78 ¢ on eth — but post-fill
drift is **1.8× larger on btc and 2.6× larger on eth** at 5 s, and negative. The
maker captures the spread and then gives back more than it captured.

**Sanity check on the harness:** measured spread capture of +0.642 ¢ on btc sits
just above the 0.50 ¢ half-spread that a 1-tick book implies, which is what
should happen when the spread is occasionally wider than one tick. The probe is
measuring what it claims.

## Why `HORIZON_DEPENDENT` and not `EDGE_NEGATIVE`

Signs by horizon:

```
btc   h=5 negative · h=15 negative · h=30 negative · h=60 spans zero
eth   h=5 negative · h=15 spans zero · h=30 spans zero · h=60 negative
```

Six of eight cells are negative with the interval excluding zero. The two that
span zero are eth at 15 s and 30 s — where the point estimates are still −0.59
and −0.67 and only the width saves them — and btc at 60 s, which is discussed
next and is **not** evidence of improvement.

## btc's h=60 result is a population artefact, not attenuation

Naively the ladder reads "adverse selection shrinks at long horizons" on btc
(drift −1.175 → −0.697). It does not. **The horizons see different
`r`-populations by construction**, exactly as the protocol warned:

| h | n | excluded (truncated) | median `r` of exclusions | `r`-population p50 |
|---:|---:|---:|---:|---:|
| 5 | 10,294 | 7 | 3.2 s | 166 s |
| 15 | 10,122 | 66 | 13.0 s | 167 s |
| 30 | 9,714 | 289 | 27.3 s | 170 s |
| 60 | 8,077 | **1,611** | 51.6 s | **190 s** |

At `h = 60` the probe discards **1,611 btc fills**, all of them inside the
terminal minute, and the surviving population shifts 24 s earlier in the window.
`h = 60` **cannot see the final minute at all, by construction.** The apparent
improvement is that selection, not a real decay in adverse selection.

Gap- and tick-touched fills are marked `UNAVAILABLE` and reported, not dropped:
btc 86 at `h=5` rising to 699 at `h=60`.

## R-DUAL: the micro class is not driving this

Excluding the 0.02-share class moves btc from −0.532 to −0.521 at `h=5`
(10,294 → 10,268 fills) and eth from −1.243 to −1.126. **The result is not a
micro-actor artefact.** Every conclusion above holds on both weightings.

## Descriptive coins — a wide spread buys about fifteen seconds

Not verdict coins, and all far below the 500-fill floor, so these are
observations rather than findings:

| coin | h=5 markout | h=30 markout | spread captured |
|---|---:|---:|---:|
| hype | **+3.802** [+2.358, +5.164] | −3.707 | +5.862 |
| bnb | +3.073 [+1.054, +5.675] | +1.202 | +4.815 |
| sol | −3.151 | −0.822 | +1.832 |
| xrp | −1.408 | −2.145 | +2.809 |

hype and bnb are **positive at 5 s** — their 5.9 ¢ and 4.8 ¢ spreads are wide
enough to cover the drift briefly — and hype turns negative by 30 s. A wide
spread does not confer edge; it buys time before adverse selection overtakes it.
n = 58 and 62, so this is directional at best.

## What this does and does not license

**It licenses:** the statement that on btc and eth a passively-quoting two-sided
maker is adversely selected by more than the spread it captures, at horizons of
5–30 s, on four days of data.

**It does not license** any claim about profitability. Layer 2 — inventory carry,
the terminal residual, the `r≈60` decision — is separate accounting and is
deliberately not combined here. Combining the two layers is what broke the
previous estimand.

## Unexplained, and not narrated

The settlement-anchored census reported **+0.173 ¢/share** pooled while Layer 1
reports **−0.53 ¢** on btc at `h=5` with the interval excluding zero. These are
different estimands over different populations — the census covered all observed
fills, this simulates a specific two-sided `JOIN_BBO` maker; the census marked at
settlement, this marks against mid — so they are not in direct contradiction.
But the direction differs and the reconciliation has **not** been measured. It
should not be narrated in either direction until it is.

## Contradiction with `FLOW_MODEL_STATE.md`

None. §1b records the estimand as broken and specifies precisely this two-layer
split; this is that split, executed. §2's `+0.173 [−0.251, +0.596]` is the
settlement estimand and remains as recorded.

**One addition the state page should carry:** the settlement estimand was
`UNDETERMINED` — every interval spanning zero — while the **narrower Layer-1
question has a determinate negative answer at short horizons on both verdict
coins**. Asking a smaller question got a sharper answer.

## Scope

Four UTC days, one collector era, 30 windows/coin, `JOIN_BBO` only. Window-
clustered intervals; **day-clustered intervals are not computable at this day
count and are not claimed**. `NEW_BBO` was not run — it carries ~9.4× the
inventory risk and belongs with the placement work, not here.
