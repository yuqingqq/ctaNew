# Non-uniform `f_r` grid, and the terminal mechanism

**Status: DEVELOPMENT.** Research only, not decision eligible, no forward-day
claim. Probe `flow_grid_terminal.py` (25 self-test checks). Receipts
`data/pm_5min/derived/flow_grid_nonuniform_v1.json` and
`flow_terminal_mechanism_v1.json` (gitignored).

Population for everything below: `clob_v3_1` covered slugs, days
2026-08-19/20/21, **361 windows per coin**. Denominator is observed exposure per
bin, gap-corrected on exact boundaries; numerator is folded taker arrivals in
the same bin. Window-clustered intervals throughout, and the standing caveat is
not optional: **window clustering cannot capture day-level common factors, so at
three collected days these intervals understate uncertainty.**

---

## Task 1 — the replacement grid

### Chosen grid

```
BODY      elapsed [0, 240)   ->  4 bins x 60 s     (r = 300 .. 60)
TERMINAL  elapsed [240, 300) -> 12 bins x  5 s     (r =  60 ..  0)
                                16 bins total
```

### Why each half is what it is

**Body at exactly 60 s.** A 60 s bin spans **exactly one period** of the
unidentified 60 s component, so it absorbs that component *by construction*
rather than by assumption. That is the only honest treatment of a term whose
**source** is unidentifiable — window phase and wall-clock minute phase are
perfectly collinear on this venue. Option (d) tried to *estimate* the component
and subtract it; the interaction test showed that estimate is not separable from
`f_r`. Absorbing it is weaker and true, where subtracting it was stronger and
false.

The body pays almost nothing for this. Within-minute log range in the body is
**0.333–0.454 on btc** and 0.365–0.503 on eth — against **2.855 and 2.649** in
the terminal. The body's within-minute structure is real (see Task 2) but small,
and it is exactly the part we cannot attribute.

**Terminal at 5 s.** The collapse is `f_r`'s largest feature and 60 s bins bury
it in one bin. Twelve 5 s bins resolve its shape, and the binding constraint is
the thinnest coin: btc holds **1,130** arrivals in its smallest terminal bin and
eth **259**, both comfortably above the 100-event floor used in Task 2.

**A control, not a convenience.** The selftest asserts each body bin is exactly
60.0 s wide — if that ever drifts, the whole justification for the body half is
false, so it fails loudly rather than silently.

### What changed versus the 15 s-uniform grid

| coin | windows old → new | shape ratio, count | shape ratio, notional |
|---|---|---|---|
| btc | 135 → 361 | 5.45 → **17.38** | 3.52 → **9.46** |
| eth | 135 → 361 | 6.87 → **14.14** | 2.91 → **4.10** |
| sol | 135 → 361 | 12.47 → 13.97 | 2.69 → 7.37 |
| xrp | 135 → 361 | 12.44 → 11.09 | 4.48 → 3.70 |
| bnb | 135 → 361 | 31.04 → 31.20 | 12.01 → 4.86 |
| doge | 135 → 361 | 27.86 → 82.54 | 6.66 → 62.86 |
| hype | 135 → 361 | 18.06 → 51.09 | 7.49 → 12.50 |

**This comparison is confounded and must not be read as a pure grid effect.**
The earlier `f_r` ran at 135 windows/coin and this one at 361, because era
coverage grew. Sample and grid both changed.

What *is* attributable to the grid: shape ratio rising on the liquid coins is a
**mechanical consequence** of resolving the terminal region at 5 s instead of
15 s. A max/min statistic necessarily widens when you stop averaging over the
steepest part of the profile. It is not a new finding, and it should not be
quoted as one.

**The finding that is real:** on btc, eth, sol and doge the **terminal collapse
ratio equals the full-profile shape ratio exactly** — both the maximum and the
minimum of `f_r` lie inside the final minute. On notional the same holds for six
of seven coins. **The terminal minute contains essentially the entire dynamic
range of `f_r`**, which is the strongest available justification for spending
resolution there and not in the body.

### btc, notional-weighted, on the new grid

```
BODY  r 240-300      97.4   CI [ 93.5, 101.5]
BODY  r 180-240      89.8   CI [ 86.1,  93.6]
BODY  r 120-180     103.3   CI [ 99.0, 107.8]
BODY  r  60-120     118.9   CI [112.4, 125.4]
TERM  r  55-60      170.4   CI [149.4, 191.2]   <- peak, immediately inside r=60
TERM  r  50-55      143.2   CI [124.4, 161.5]
TERM  r  45-50      132.0   CI [112.4, 152.4]
TERM  r  40-45      104.1   CI [ 87.7, 120.3]
TERM  r  35-40       74.9   CI [ 62.2,  87.5]
TERM  r  30-35       79.4   CI [ 64.7,  94.8]
TERM  r  25-30       67.2   CI [ 53.7,  81.2]
TERM  r  20-25       64.3   CI [ 49.8,  80.8]
TERM  r  15-20       55.9   CI [ 41.2,  71.8]
TERM  r  10-15       59.9   CI [ 41.2,  81.9]
TERM  r   5-10       33.0   CI [ 21.4,  47.1]
TERM  r   0-5        18.0   CI [  9.5,  29.0]
```

Notional intensity **rises through the body, peaks in the first 5 s inside the
settlement window, then declines monotonically by 9.5× to expiry.** The 15 s
grid could not show the peak's location; the 60 s grid would have hidden the
decline entirely.

**Per-coin primary weighting is unchanged and still binds** (R-DUAL): micro
share is btc 2.3 %, eth 25.4 %, sol 34.9 % → `EITHER`; xrp 63.4 %, doge 70.0 %,
bnb 75.9 %, hype 91.4 % → **`NOTIONAL_ONLY`**. The count column for those four
is a participant measurement, not market flow, and the count shape ratios above
(doge 82.54, hype 51.09) should not be read as flow structure.

---

## Task 2 — terminal mechanism: `TWAP_FAVOURED`

Rule stated in the probe before the measurement ran. Verdict coins btc and eth
per `FLOW_MODEL_PROTOCOL_V4`.

| coin | body mean log range | terminal log range | ratio | ratio CI95 | terminal ρ | min bin |
|---|---:|---:|---:|---|---:|---:|
| **btc** | 0.390 | 2.855 | **7.32** | **[6.06, 8.51]** | **−1.00** | 1130 |
| **eth** | 0.436 | 2.649 | **6.08** | **[4.62, 7.49]** | **−0.99** | 259 |
| doge | 0.849 | 4.364 | 5.14 | [4.14, 6.46] | −0.99 | 12 |
| sol | 0.716 | 2.637 | 3.68 | [2.70, 5.22] | −0.99 | 99 |
| bnb | 0.982 | 2.994 | 3.05 | [2.30, 4.44] | −0.99 | 40 |
| hype | 1.194 | 3.586 | 3.00 | [2.39, 3.72] | −0.99 | 20 |
| xrp | 0.858 | 2.312 | 2.69 | [2.02, 3.66] | −0.99 | 128 |

Both verdict coins clear the 3.0 material bar with **lower bounds of 6.06 and
4.62**, and both terminal declines are essentially perfectly monotone.

`doge`, `hype`, `bnb` and `sol` fall below the 100-event floor in their smallest
terminal bin and are **reported but excluded from the verdict**. They point the
same way, which is reassuring and is not evidence.

### What this refutes, and what it does not

**Refuted: a uniform minute-boundary artefact as the explanation of the terminal
collapse.** Such an artefact predicts comparable within-minute amplitude at every
boundary. Measured, the terminal is **6–7× the body's** on the verdict coins.

**A periodic component nevertheless EXISTS, and this is the part not to
over-read.** The body minutes decline monotonically too — btc within-minute
ρ = −0.96, −0.97, −0.95, −0.50; eth −0.83, −0.95, −0.93, −0.56. So there *is* a
within-minute declining pattern at every boundary, whose source remains
unidentifiable. It is simply 6–7× too small to account for the terminal collapse.

**NOT established: TWAP lock-in.** The test can refute the uniform-artefact
explanation. It cannot confirm the alternative: a *non-stationary* artefact, or
any distinct real effect confined to the final minute, predicts the same shape.
`TWAP_FAVOURED` means the uniform-artefact explanation is dead and TWAP remains
the standing candidate — nothing more.

### Interpretive, deliberately excluded from the verdict

Log-rate jump crossing `r = 60`: **+0.422 (btc, 1.53×)** and **+0.260 (eth,
1.30×)**. Activity steps *up* on entering the settlement window and then decays.

This is consistent with the TWAP story — the last moment at which a trade can
influence the full 60 s mean — and it is **excluded from the verdict on
purpose**: the pattern was visible in the `phase × r` receipt before this rule
was written, so using it to decide would be circular. It is reported for
interpretation only.

Terminal profiles, log, centred within the minute, r = 60 → 0:

```
btc  +1.04 +0.78 +0.73 +0.56 +0.30 +0.18 +0.11 -0.11 -0.25 -0.52 -1.00 -1.82
eth  +1.12 +0.84 +0.65 +0.44 +0.25 +0.16 +0.15 -0.17 -0.49 -0.48 -0.94 -1.53
```

Smooth and monotone, with no step at any 15 s or 30 s sub-boundary — the shape a
continuous lock-in predicts, not a discrete one.

---

## Contradicts a current claim in `FLOW_MODEL_STATE.md`

**§2, "Measured and UNDETERMINED", entry `f_r`'s terminal mechanism.** It reads:

> The settlement TWAP lock-in is the natural explanation and is **confounded
> with a 60 s artefact** — window phase and wall-clock minute phase are
> perfectly collinear here.

**The confound is partially broken and that entry needs amending.** Collinearity
of *phase* is real and unchanged, but the two hypotheses were never
observationally equivalent: they differ in **amplitude across minutes**, which
nobody had tested. On that axis a uniform artefact is refuted at 6–7× on both
verdict coins.

Suggested replacement, for the coordinator to apply — this probe does not edit
that page:

> **`f_r`'s terminal mechanism — the UNIFORM-artefact explanation is REFUTED.**
> Terminal within-minute amplitude is 6–7× the body's (btc ratio 7.32
> [6.06, 8.51], eth 6.08 [4.62, 7.49]), with monotone declines at ρ ≈ −1.00, so
> a constant-amplitude minute-boundary effect cannot account for the collapse. A
> periodic component does still exist in the body (within-minute ρ ≈ −0.95) and
> its **source remains unidentifiable**. TWAP lock-in is now the standing
> candidate but is **NOT established** — a non-stationary artefact or any effect
> confined to the last minute predicts the same shape.

Two entries in §1 are **sharpened rather than contradicted**: "notional rises,
peaks in the second half, then falls" is now located precisely — the peak is the
first 5 s inside `r = 60`, and the subsequent decline is 9.5× on btc.

## What this does not touch

Nothing here bears on the maker-edge sign, the fill bracket, queue position, or
the rebate. Task 2 speaks to the *shape* of the terminal effect and its
*amplitude across minutes*; it says nothing about the collapse's economic
consequence.
