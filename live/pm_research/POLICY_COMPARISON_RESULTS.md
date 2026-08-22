# Policy comparison — new-BBO vs join-BBO, paired

Protocol `policy_v1`, frozen before measurement. Probe `policy_comparison.py`
(25 self-test checks). 70 windows, 10 per coin, `clob_v3_1`, both complete days.
Receipt: `data/pm_5min/derived/policy_comparison_v1.json`.

**Verdict under the rule as written: `UNRESOLVED`.** But the verdict is not the
finding, and it should not be read as one. See §3 — the estimand is confounded,
and the protocol's own instruction is what caught it.

## 1. The result

Headline horizon 15 s, all-flow weighting, per coin. `Δ` is
`NEW_BBO − JOIN_BBO`. **`NEW_BBO` is an upper bound on itself throughout**:
being first assumes we win the race against others quoting the same new level,
which depends on latency not present in this tape.

| coin | n paired | Δfill | Δmarkout ¢/sh | Δedge | Δedge CI95 |
|---|---:|---:|---:|---:|---|
| **btc** | 12,280 | +0.070 | +0.90 | **+3.66** | [1.85, 5.46] |
| **eth** | 15,945 | +0.222 | +0.66 | **+2.56** | [−1.10, 5.74] |
| bnb | 13,398 | +0.613 | +15.36 | −2.76 | [−9.45, 3.11] |
| doge | 10,823 | +0.630 | +9.38 | −10.72 | [−19.63, −0.22] |
| hype | 9,915 | +0.729 | +4.19 | −6.73 | [−11.41, −1.94] |
| sol | 9,428 | +0.466 | +3.84 | −1.71 | [−9.39, 3.54] |
| xrp | 20,546 | +0.520 | +6.41 | +2.93 | [−5.64, 10.32] |

`NEW_BBO_DOMINATES` requires the `Δedge` interval to exclude zero on **both**
verdict coins. btc does; eth does not. So `UNRESOLVED`, and the rule's own
wording applies: the interval reaches 5.74, far wider than the 0.25 bound, so an
effect worth acting on is **not excluded**. This is not `NO_DIFFERENCE`.

**Availability.** Touch formations are abundant — 948 (sol) to 2,048 (xrp) per
window — so `NEW_BBO` is not a rare-opportunity policy. Unavailable actions
(gap- or tick-change-touched) are reported not dropped: btc 736, sol 307, eth
248, hype 16, bnb 3, doge and xrp 0.

**R-DUAL.** Removing the micro class moves `Δedge` by at most 0.16 on any coin
(btc identical to 3 dp). The single-actor class does not drive this comparison,
which is expected — it trades 0.02 shares against a 5-share action.

## 2. The verdict is SAMPLE-UNSTABLE, which is the first warning

At 3 windows/coin the same code returned **`NEW_BBO_DOMINATES`**. At 10
windows/coin it returns **`UNRESOLVED`** — eth's interval opened from
[5.31, 8.68] to [−1.10, 5.74].

A verdict that flips between two samples of the same era is fitting noise. On
its own that is sufficient reason not to act on the earlier reading, and it
matches the pattern already recorded for B2, whose sign flipped between 12 and
24 windows.

## 3. The estimand is confounded — this is the real finding

The protocol pre-specified: *"a result showing `NEW_BBO` better on BOTH is a
reason to suspect the measurement, not to celebrate."* The 3-window run showed
exactly that. Checking the pairing and availability, as instructed, found the
pairing **sound** — same level, same instant, same subsequent flow, differing in
exactly one field. The defect is upstream of the pairing, in what is being
measured.

**Marginal-fill edge, implied from the levels** (the decisions where `NEW_BBO`
fills and `JOIN_BBO` does not):

| coin | marginal share | common markout | **marginal markout** |
|---|---:|---:|---:|
| btc | 0.070 | −1.45 | **+10.31 ¢/share** |
| xrp | 0.520 | −6.48 | +3.34 |
| eth | 0.222 | −0.65 | +1.88 |

**+10 ¢/share on a book whose entire half-spread is 0.50 ¢ is not a
market-making edge.** It is directional P&L, and two specified choices produce
it together:

1. **Markout is taken against settlement**, per the protocol, so a fill at
   `t=30 s` is marked at `t=300 s`. That is hold-to-expiry exposure, not spread
   capture. The horizon controls only the *fill window*, not the holding period.
2. **The order rests untouched for the whole horizon** — never cancelled, never
   repriced. Nothing in the tape lets us model a maker who would pull a stale
   quote.

Together these mean the marginal fills — small volume reaching the level — are
precisely the cases where **price moved away and never came back**. A resting
bid that catches a few shares and then watches the market run is holding a
directional position to settlement. The measured advantage of front-of-queue is
mostly that advantage.

The underlying mechanism is real and well known: fill size correlates with
adverse selection, so being filled *completely* means the market went through
you. But this construction cannot separate that from directional drift, and the
drift term is an order of magnitude larger.

## 4. Rule revisions raised, not applied

**R1 — `Δfill > 0` is an IDENTITY, not a measurement.** `action_fill` returns
`front = min(size, cum)` and `back = min(size, max(0, cum − queue_ahead))`, so
`front ≥ back` for any `queue_ahead ≥ 0`. The two policies differ only in that
field, so `Δfill ≥ 0` holds by construction on every action and every coin.
`TRADE_OFF_CONFIRMED` requires `Δfill > 0` with its interval excluding zero —
a condition that is satisfied automatically and carries no evidence. Only
`Δmarkout` and `Δedge` are informative. A self-test pins the ordering as an
identity so this cannot be mistaken for a result later.

**R2 — units mismatch in the `NO_DIFFERENCE` bound.** `Δedge` is defined as
fill-weighted edge *per decision*, so its units are shares × cents (up to 5
shares of capacity). The bound is stated as ±0.25 **¢/share**. In these units
that is ±1.25. The verdict is unaffected — the intervals reach 5.74 either way —
but the two must be reconciled before any run can return `NO_DIFFERENCE`.

**R3 — the estimand does not isolate placement, and this is the one that
matters.** Markout against settlement plus a never-cancelled order makes the
comparison a test of hold-to-expiry directional exposure conditioned on fill
size. Isolating placement requires marking out at a **fixed short horizon after
the fill** rather than at settlement, so that spread capture is separated from
drift. That is a different estimand and needs a new pre-registration; it must
not be swapped in after seeing this result.

## 5. Against `FLOW_MODEL_STATE.md`

Nothing is contradicted. The page's fill-bracket entries (btc 94.6 % front vs
76.9 % back at 15 s) are unpaired, clock-time-anchored figures from
`flow_fill_development`; the rates here (0.914 / 0.844 on btc) are
formation-anchored and paired, so they are a different population and are not
comparable. Both are reported against their own denominators.

The page's §3 correction — that queue position is an output of the placement
policy rather than an unknown — is **supported**: both policies were simulable
from the tape without assuming any queue parameter, and the pairing worked.

## 6. What would settle it

A pre-registered successor marking out at a fixed horizon after fill, with a
cancellation policy stated rather than assumed. Until then this comparison
should not be cited as evidence about placement, in either direction.
