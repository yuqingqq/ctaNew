# Flow model — `f_r` and `f_p`, measured

Probe: `flow_intensity.py` (24 selftest checks, incl. flat/ramp vacuity controls).
Scope: `clob_v3_1` covered set, **945 windows, 135 per coin**, 35 gap-touched,
exposure-corrected with exact gap boundaries. 423,134 folded arrivals.

Everything below is **within-era, under two days**. No day-clustered interval is
computable; the window-clustered CIs shown **understate** true uncertainty.

---

## 1. The pooled "16.3 % of events" is a BTC-dominated statistic

**The most consequential thing measured here, because the figure is already
treated as established in the plan, in STATUS and in a commit message.**

| coin | arrivals | micro (0.02) | micro % | **primary weighting** |
|---|---:|---:|---:|---|
| btc | 270,404 | 5,417 | **2.0 %** | either; count is usable |
| eth | 56,265 | 12,599 | 22.4 % | notional; count secondary |
| sol | 24,977 | 7,426 | 29.7 % | notional; count secondary |
| xrp | 24,107 | 14,444 | 59.9 % | **notional only** |
| doge | 14,296 | 9,857 | 68.9 % | **notional only** |
| bnb | 16,925 | 13,241 | 78.2 % | **notional only** |
| hype | 16,160 | 14,538 | **90.0 %** | **notional only** |
| **pooled** | **423,134** | **77,522** | **18.3 %** | — |

**BTC supplies 64 % of the pooled denominator.** The single-actor class runs from
**2 % of BTC arrivals to 90 % of hype arrivals** — a 45× range the pooled number
conceals. On thin coins the count layer is not *contaminated by* that actor; it
largely **is** that actor.

### R-DUAL is not a uniform reporting convention

Both weightings are not co-equal everywhere. On btc, count-weighted intensity is
a market measurement. On hype it is not one at all — it is a measurement of one
participant with a 10 % market residual. The `primary weighting` column above is
binding: **above ~35 % micro share, count-weighted intensity may not be used as
evidence about market flow**, only as a diagnostic reported beside notional.

Sixth instance of the loop's denominator/population defect — a pooled statistic
whose denominator is dominated by one member.

---

## 2. `f_r` — two claims of very different strength

### 2a. "Arrival intensity does not rise into settlement" — WELL SUPPORTED

This refutes a directional expectation the plan carried (`f_r ← deterministic
window clock, DOMINANT`). Count-weighted arrivals/s per 15 s bin:

```
r_mid    292   262   232   202   172   142   112    82    52    38    22     8
btc     8.67  6.86  8.20  6.50  7.54  6.57  7.85  6.99  9.83  6.00  4.03  1.80
eth     1.73  1.56  1.89  1.37  1.83  1.48  1.80  1.44  1.56  0.81  0.48  0.28
sol     0.53  0.82  0.92  0.68  0.83  0.68  0.87  0.60  0.56  0.32  0.20  0.07
xrp     0.59  0.85  0.91  0.64  0.82  0.71  0.67  0.49  0.50  0.26  0.16  0.07
bnb     0.38  0.57  0.75  0.49  0.59  0.41  0.46  0.36  0.26  0.16  0.08  0.02
doge    0.33  0.46  0.58  0.41  0.48  0.38  0.41  0.28  0.31  0.15  0.10  0.02
hype    0.32  0.51  0.59  0.48  0.52  0.48  0.45  0.39  0.30  0.20  0.10  0.03
```

No coin shows a rising count profile. That much holds on every weighting.

### 2b. "Flat, then collapses" — SUPPORTED ON BTC ONLY

`mean(bins 17–19) / mean(bins 0–15)`:

| coin | count | count ex-micro | notional |
|---|---:|---:|---:|
| **btc** | **0.562** | **0.571** | **0.692** |
| eth | 0.339 | 0.413 | 0.711 |
| sol | 0.283 | 0.356 | 0.554 |
| xrp | 0.243 | 0.440 | 0.630 |
| doge | 0.221 | 0.564 | 0.932 |
| bnb | 0.177 | 0.421 | 0.685 |
| **hype** | **0.241** | **1.055** | **1.421** |

On btc the collapse is weighting-insensitive and is a finding. **On hype it is
entirely the micro actor going quiet** — ex-micro 1.055, notional **1.421**, so
real flow *rises*. **A conclusion whose sign flips with the weighting is not a
finding about the market.** "Flat then collapses" must not be read as a
market-wide property; it is a btc property.

---

## 3. Notional-weighted `f_r` — the series that survives contamination

USDC/s per 15 s bin, on 5 of 7 coins the **only** admissible weighting.

```
r_mid     292    278    262    248    232    218    202    188    172    158
btc     98.43  82.11  81.53  72.54  91.52  84.56  83.47  84.56 101.06  92.60
eth     10.99   7.71  10.35   9.29  11.61   9.85   8.86   9.30  14.07  12.24
sol      2.34   3.56   4.69   2.93   3.86   3.03   3.23   2.83   4.34   4.12
xrp      1.04   1.30   1.90   0.86   1.53   1.27   1.13   1.18   2.64   1.84
bnb      0.19   0.40   0.27   0.42   0.62   0.44   0.60   0.49   0.72   0.65
doge     0.29   0.68   0.53   0.34   0.52   0.49   0.56   0.66   1.15   0.81
hype     0.09   0.25   0.10   0.20   0.18   0.14   0.15   0.14   0.14   0.28

r_mid     142    128    112     98     82     68     52     38     22      8
btc     94.52  91.12 120.35 104.83 113.32 124.61 152.20  86.60  67.56  43.26
eth     11.47  10.91  16.40  12.14  14.63  15.50  18.47   9.46   6.34   8.90
sol      3.95   3.22   5.15   3.84   5.55   4.96   4.80   2.21   2.06   2.14
xrp      3.68   2.81   3.86   2.10   3.51   2.94   3.72   1.29   1.22   1.46
bnb      0.69   0.76   1.10   1.14   2.29   1.22   1.19   0.67   0.39   0.48
doge     0.86   0.82   1.23   0.94   1.12   1.19   1.94   0.80   1.01   0.32
hype     0.32   0.35   0.35   0.27   0.49   0.69   0.56   0.50   0.42   0.19
```

**Notional flow RISES through the window, peaks at `r ≈ 52–112 s`, then falls.**
That is a different shape from the count profile, and it is invisible in counts
because count is flat while size grows.

| coin | peak bin | peak `r` | peak / open |
|---|---|---:|---:|
| btc | 16 | 52 s | 1.55× |
| eth | 16 | 52 s | 1.68× |
| sol | 14 | 82 s | 2.37× |
| xrp | 12 | 112 s | 3.71× |
| doge | 16 | 52 s | 6.69× |
| hype | 15 | 68 s | 7.67× |
| bnb | 14 | 82 s | 12.05× |

The rise is significant — window-clustered CIs at open and peak do not overlap:

```
btc   bin  0  r=292s   98.43  [ 90.44, 106.55]      bin 16  152.20  [126.99, 178.83]
doge  bin  0  r=292s    0.29  [  0.20,   0.41]      bin 16    1.94  [  1.39,   2.57]
hype  bin  0  r=292s    0.09  [  0.03,   0.17]      bin 16    0.56  [  0.30,   0.91]
```

**Precision, stated rather than implied.** CI half-width as a share of the point
estimate: btc **8–17 %** across bins 0–16, degrading to 28 % and 45 % in the
final two. Alts run 15–60 % mid-window and exceed 80–115 % in bin 19. hype is
34–113 % throughout. **Read alt tails as directional only.**

**The peak location is entangled with §5.** Five of seven peaks fall on a
60 s-grid bin (12, 16), which is exactly the oscillation whose source cannot be
identified. Treat `r ≈ 52–112` as "the second half of the window", not as a
located peak.

### Both weightings fall into the close. They disagree about the middle.

`peak → final bin (r = 8 s)`:

| coin | count peak → b19 | notional peak → b19 | USDC/arrival, peak → b19 |
|---|---:|---:|---:|
| btc | 9.83 → 1.80 (**0.184×**) | 152.20 → 43.26 (**0.284×**) | 15.5 → 24.0 |
| eth | 1.56 → 0.28 (0.177×) | 18.47 → 8.90 (0.482×) | 11.9 → 32.3 |
| sol | 0.60 → 0.07 (0.124×) | 5.55 → 2.14 (0.384×) | 9.3 → 28.8 |
| xrp | 0.67 → 0.07 (0.109×) | 3.86 → 1.46 (0.378×) | 5.7 → 19.9 |
| bnb | 0.36 → 0.02 (0.068×) | 2.29 → 0.48 (0.211×) | 6.4 → 19.9 |
| doge | 0.31 → 0.02 (0.068×) | 1.94 → 0.32 (0.168×) | 6.3 → 15.7 |
| hype | 0.36 → 0.03 (0.090×) | 0.69 → 0.19 (0.267×) | 1.9 → 5.7 |

**These are two different terminal facts and both must be stated.** Notional does
not merely rise-then-fall: it peaks in the second half **and then falls sharply
into the close** on every coin. So *both* weightings fall at the end — they
disagree about the **middle**, not the close.

But they fall by different amounts, and the gap is the point. On btc, count drops
to **18 %** of peak while notional holds **28 %**, because individual arrivals
grow from 15.5 to **24.0 USDC**. On eth the divergence is wider still: count 18 %
against notional **48 %**, with arrivals nearly tripling to 32.3 USDC.

**A maker reading counts alone would badly misjudge the last 40 seconds** — the
count series says the terminal regime is quiet, while notional says it is
smaller than the peak but still substantial, delivered in far larger individual
arrivals. btc's final bin still carries 43.26 USDC/s (CI [25.75, 64.42]).

---

## 4. Count falls while size grows — the operational consequence

Mean USDC per arrival:

```
r_mid    292   232   172   112    82    52    22     8
btc     11.4  11.2  13.4  15.3  15.5  15.5  16.8  24.0
eth      6.4   6.1   7.7   9.1   8.1  11.9  13.2  32.3
sol      4.4   4.2   5.2   5.9   6.4   8.6  10.5  28.8
```

Mean trade size **doubles to triples** into settlement.

**INFERENCE, not a measurement.** This combination — falling arrival count with
rising size per arrival — describes a **concentrated** terminal regime rather
than a quiet one. For a market maker that is the adverse combination: fewer,
larger and plausibly better-informed counterparties arriving precisely when
inventory is hardest to unwind before settlement. It follows from the two
measured series above; **the informedness of terminal flow is not measured here**
and this is flagged as an inference so it is not later cited as a result.

---

## 5. A 60 s oscillation — structure present, source UNIDENTIFIABLE

Bins 0, 4, 8, 12, 16 — exactly 60 s apart — are local maxima for **every coin**.
On btc the CIs separate peaks from adjacent troughs:

```
bin  3  r=248s   6.41  [5.90, 6.92]
bin  4  r=232s   8.20  [7.65, 8.78]   <- 60s grid
bin  5  r=218s   6.91  [6.41, 7.46]
```

**The structure is real. Its source cannot be identified on this data.** All 952
window starts are `≡ 0 (mod 60)` *and* `≡ 0 (mod 300)`, so **window phase and
wall-clock minute phase are perfectly collinear**. A minute-boundary effect in
the underlying crypto market — well documented — reproduces this exactly, and no
window in the data starts off-minute to break the tie.

Structurally identical to §3.1's fee/moneyness non-identification: the covariate
varies only along an axis another effect also varies along, and the venue offers
no independent variation.

**What would identify it**, so nobody re-derives this as new:
1. Windows on a schedule **not** aligned to 60 s (this venue has none).
2. The same profile measured against an underlying series with **no**
   minute-boundary artefact, isolating the venue-side component.
3. A venue with the same window length on a different phase grid.

**Do not attribute the oscillation to the window clock.**

### The most attractive explanation for the terminal shape is ALSO confounded

**Read this before reaching for the obvious story.** Settlement is `S60(T)` vs
`S60(t0)` — a **60-second trailing mean**. So through the final minute the
settlement value is progressively locking in; by `r ≈ 30 s` half the settlement
TWAP is already realised and cannot be moved by subsequent trading. That is a
clean mechanism for activity falling in **both** weightings from around
`r ≈ 52–60 s`, and it is exactly where the fall begins.

**It is not identifiable here, because it is collinear with the artefact.** The
settlement TWAP is 60 s, the unexplained oscillation is 60 s, and every window
start is `≡ 0 (mod 60)`. "Activity falls because the settlement window is locking
in" and "this is the unidentified minute-boundary structure" predict the same
profile on this data, and **nothing here separates them**.

This note matters more than the oscillation note alone: a reader who sees a peak
at `r ≈ 52` and knows the settlement rule will reach for the TWAP story
immediately and treat the shape as **explained**. It is not. Record it as a
**named candidate mechanism, confounded with a known artefact** — not as an
explanation, and not as evidence for the settlement rule's behavioural effect.

**What would separate them:** a venue running a **different settlement-window
length on the same 5-minute grid**. The TWAP timescale would move while the
minute grid did not, breaking the collinearity. The three separators listed above
break the minute-grid confound but **not** this one, since they leave the TWAP
length untouched.

---

## 6. `f_p` — hump-shaped, peaking mid-book

Arrivals/s per unified-price bin, dwell from the Up-token mid path.
**Subsample: 6 windows per coin** (the quote stream is ~97 % of message volume).

```
p-bin      [0,.05) [.05,.15) [.15,.35) [.35,.65) [.65,.85) [.85,.95) [.95,1]
btc          2.24    8.22      9.54     10.05     12.72     7.60      3.27
eth          0.20    2.23      3.18      2.00      2.68     3.34      0.95
sol          0.26    1.35      0.92      0.92      0.85     1.02      0.40
xrp          0.21    1.54      1.77      1.31      1.84    FENCED    FENCED
bnb          0.06    0.71      0.77      0.61      0.91     0.77      0.15
doge         0.07    0.71      1.10      0.70      0.70     0.48      0.15
hype         0.07    0.58      0.64      0.64      0.70    FENCED     0.01
```

Intensity is **low in both extreme tails and peaks across the middle**.

**FENCED bins — minimum dwell 60 s.** Rates off negligible dwell are excluded,
not reported: `xrp [.95,1]` has **0 s** of dwell, `xrp [.85,.95)` **6 s**, and
`hype [.85,.95)` **9 s** — the last produced a spurious 3.72/s that alone drove
a `shape_ratio` of 295. A ratio off 9 seconds of denominator is noise wearing a
number. Dwell per bin is recorded in `data/pm_5min/derived/flow_fp.json`; every
reported cell above clears the fence.

**Not evidence about the fee.** The crossing cost `0.07·p(1−p)` is maximal at
ATM and intensity is also high there. That is the **unidentified** comparison
§3.1 already rules out — uncertainty peaks at ATM too — and it must not be cited
in either direction. Recorded as a shape only.

---

## What this contradicts

1. **Plan §2.2** — `f_r` as a dominant rising window clock. Count intensity does
   **not** rise into settlement on any coin. Notional intensity **does** rise,
   peaking in the second half — so the plan's direction is wrong for counts and
   right-ish for notional, which are different claims about different series.
2. **Plan R-DUAL section, STATUS, and commit `6a0e593`** — "16.3 % of events" is
   pooled and btc-dominated. Per coin, 2.0 %–90.0 %.
3. **"The count layer is contaminated"** — too weak for the alts, and R-DUAL is
   not a uniform convention. Above ~35 % micro share, count-weighted intensity
   is not a market measurement at all.

## What is NOT established

- **The mechanism behind the terminal fall.** The `S60` settlement TWAP is a
  named candidate and is **confounded** with the 60 s artefact — both predict the
  same profile and this data cannot separate them (§5). Do not treat the shape as
  explained.
- **The source of the 60 s oscillation.**
- **Whether terminal flow is better informed.** §4 is an inference from two
  measured series, not a measurement.
- **What the micro actor is doing.**

None of these is narrated, and the first is the one most likely to be assumed
resolved by a reader who knows the settlement rule.
