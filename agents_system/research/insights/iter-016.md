# Research Insights — iter-016 (SCOPING DIAGNOSTIC: is event-driven / variable-horizon worth building?)

**Question (human):** the held book holds 24h via 6 overlapping 4h sleeves for COST AMORTIZATION,
not because the signal predicts 24h. Does the 4h-trained mean-rev `pred` still have edge beyond
4h (justifying capturing more of the hold), or does it decay fast (=> sleeves 2–6 hold STALE
signal, and an event-driven exit-on-decay could trade less while preserving edge)? The decisive
sub-question for VARIABLE horizon: is the decay HOMOGENEOUS (fixed hold already optimal — iter-014
swept it) or HETEROGENEOUS by an observable-at-entry (=> variable has real potential)?

Diagnostic, not a build. All numbers computed on HL70 (prod) + EXT (2021–26), SIDE regime (where
the mean-rev pred operates), bull/all for reference. Scripts: `iter016_decay.py` (decay curve +
heterogeneity + event-entry), `iter016_varhold.py` (variable-hold vs fixed vs G4 placebo +
nested-OOS). 4h-grid entries; forward returns from 5m klines at h∈{4,8,12,16,24,36,48,72}h.

---

## TL;DR

The 4h mean-rev signal **decays FAST and goes NEGATIVE well before 24h** on the production
universe: HL70 SIDE cross-sectional IC(pred, fwd-ret) peaks at h=4 (+0.0018, t+0.6 — already weak),
crosses zero by **h≈10–12h**, and is **significantly NEGATIVE by 24–72h** (IC −0.0052 @24h t−1.5,
−0.0154 @48h t−4.5, −0.0178 @72h t−5.3). So the **24h held-book hold is BEYOND the signal's life**
— sleeves 2–6 are holding stale, mildly anti-correlated signal. That part of the human's hunch is
**confirmed**: the 24h hold is a cost-amortization device, NOT signal capture, and at h≥24 the
stale book is a small drag in expectation (the iter-014 longer-hold Calmar gain comes purely from
turnover/cost, never from the signal living longer).

**BUT** the heterogeneity that variable-horizon needs is **NOT exploitable**. The decay *shape*
does differ by signal-strength tercile (heterogeneous), and an oracle variable rule "hold each
tercile its in-sample-best horizon" beats fixed-24h AND beats the G4 matched-avg-hold placebo
(HL70 p100, EXT p97) — the same seductive in-sample win the log keeps producing. It **dies in
nested-OOS**: the tercile→best-horizon mapping is unstable in time. On HL70 the in-sample map
{weak:24, mid:4, strong:16} vs first-half map {weak:12, mid:72, strong:12} — when you pick the
horizon on the first half and apply it forward, the variable rule loses **−45.2 bps** vs fixed-24h
(second-half: fixed +4.6 → variable(OOS) −40.6). EXT nested-OOS is only +3.3 bps (noise). The
heterogeneity is **real but noise-dominated** — optimal hold-per-tercile does not generalize.

**Verdict: NOT worth building an event-driven / variable-horizon engine.** The signal decays fast
(so a "hold longer to capture more" event engine has nothing to capture past ~12h), and the
"exit fast-decayers early / hold slow-decayers" variable rule fails nested-OOS exactly like the
K3-cost-margin and decay-weighted-sleeve attempts before it. The one honest, structural lever in
this space is **already characterized** (iter-014: longer FIXED hold = lower DD via cost
amortization, but fails production nested-OOS/G6). There is no new structural win here.

---

## 1. SIGNAL DECAY CURVE — cross-sectional IC(pred, fwd-ret) by horizon

**HL70 (production):**
| regime | h=4 | h=8 | h=12 | h=16 | h=24 | h=36 | h=48 | h=72 |
|---|---|---|---|---|---|---|---|---|
| **SIDE** | **+0.0018** | +0.0013 | −0.0006 | −0.0026 | −0.0052 | −0.0098 | −0.0154 | −0.0178 |
| (t) | +0.6 | +0.4 | −0.2 | −0.8 | −1.5 | −2.9 | −4.5 | −5.3 |
| bull | +0.0099 | +0.0050 | +0.0044 | +0.0023 | +0.0019 | −0.0042 | −0.0059 | −0.0205 |
| all | +0.0032 | +0.0030 | +0.0027 | +0.0008 | −0.0011 | −0.0049 | −0.0089 | −0.0125 |

**EXT (2021–26):**
| regime | h=4 | h=8 | h=12 | h=16 | h=24 | h=36 | h=48 | h=72 |
|---|---|---|---|---|---|---|---|---|
| SIDE | −0.0042 | −0.0057 | −0.0045 | −0.0051 | −0.0034 | −0.0049 | −0.0056 | −0.0029 |
| bull | −0.0044 | −0.0116 | −0.0177 | −0.0176 | −0.0210 | −0.0235 | −0.0239 | −0.0142 |
| all | −0.0025 | −0.0054 | −0.0056 | −0.0051 | −0.0056 | −0.0078 | −0.0083 | −0.0040 |

**Reads:**
- **HL70 SIDE: peak at h=4, zero-cross at h≈10–12h, significantly negative ≥24h.** The signal's
  natural "life" is ~one 4h bar, maybe two. The 24h held book is firmly **beyond** signal life —
  sleeves 2–6 carry a stale, slightly *anti*-signal position. This is consistent with iter-006's
  finding that the side mean-rev alpha is near-zero-edge noise: even the h=4 peak IC is +0.0018
  (t+0.6, not significant). There is no edge to "capture more of" past 4h.
- **EXT SIDE: no positive IC at ANY horizon** (all slightly negative, none significant) — matches
  iter-006 "zero-edge with a fat tail" on the longer panel. Variable horizon can't rescue a signal
  with no positive IC to begin with.
- **Bull regime decays slower** (HL70 bull positive to ~h=24) but bull uses momentum, not the
  mean-rev pred, and is not the DD-driver — out of scope for the side-book question.

> The bull/all curves and the bull/side contrast confirm the 24h hold is NOT justified by signal
> persistence on the production universe. The hold is cost-amortization, full stop.

## 2. HETEROGENEITY (the variable-horizon win condition) — LS-spread fwd-ret by strength tercile

K=5 long-short spread forward return (bps) by per-cycle |pred|-strength tercile, SIDE regime:

**HL70** (n=1587):
| tercile | h=4 | h=8 | h=12 | h=16 | h=24 | h=36 | h=48 | h=72 | peak |
|---|---|---|---|---|---|---|---|---|---|
| weak | +0.0 | −4.0 | −1.2 | +8.5 | **+25.7** | +17.4 | −4.6 | −7.6 | 24h |
| mid | −3.1 | −3.5 | −1.2 | −11.6 | −14.8 | −27.0 | −38.8 | −38.3 | 12h |
| strong | +2.4 | +2.4 | +5.5 | **+6.6** | −8.5 | −12.6 | −32.1 | −65.4 | 16h |

**EXT** (n=5232):
| tercile | h=4 | h=8 | h=12 | h=16 | h=24 | h=36 | h=48 | h=72 | peak |
|---|---|---|---|---|---|---|---|---|---|
| weak | +1.2 | +3.4 | +3.0 | +4.6 | +6.9 | +9.9 | +12.8 | +4.4 | 48h |
| mid | +3.6 | +3.9 | +6.9 | +8.7 | +16.0 | +25.9 | +28.0 | **+32.1** | 72h |
| strong | +1.7 | +3.4 | +5.0 | **+8.1** | +6.2 | +0.8 | +1.7 | +1.4 | 16h |

**The decay IS heterogeneous** (terciles peak at different horizons) — superficially the
variable-horizon win condition. BUT the pattern is **inconsistent and non-monotone**:
- On HL70 the **strong** tercile peaks early (16h) and then crashes hardest (−65 bps @72h);
  **weak** peaks latest (24h). That is the OPPOSITE of the intuitive "strong signal persists
  longer" — strong mean-rev names snap back fast then reverse.
- On EXT the mapping flips: **strong** peaks at 16h again, but **mid/weak** peak at 48–72h.
- The mid tercile on HL70 is monotonically *negative* — pure noise.

There is no stable, panel-consistent "hold X type longer" rule. Heterogeneity exists but is not a
clean function of the observable (|pred| strength).

## 2b. MARGINAL per-window spread — where holding adds vs goes stale (SIDE, bps per 4h window)

**HL70:** strong adds +2.4/−0.1/+3.2/+1.0 in the first four 4h windows then goes **−15/−4/−20/−33**
from 16h onward — i.e. the held book is earning the signal in the FIRST ~12–16h and bleeding it
back after. weak is noise early, +17 @16–24h, then −22 @36–48h. Past ~24h every tercile is
negative-marginal on HL70 (stale-signal drag, masking the cost-amortization benefit iter-014 found).
**EXT** marginals are mostly small-positive (the cost-amortization regime) — which is why iter-014's
longer-hold Calmar lift was cleaner on EXT/S44 than on HL70.

## 3. EVENT-DRIVEN ENTRY value — IC by horizon, all rows vs strong-|pred| subset (SIDE)

| panel | rows | h=4 | h=8 | h=12 | h=16 | h=24 | h=48 |
|---|---|---|---|---|---|---|---|
| HL70 | all | +0.0018 | +0.0013 | −0.0006 | −0.0026 | −0.0052 | −0.0154 |
| HL70 | **strong (|pred|≥cycle-p70)** | **+0.0101** | +0.0060 | +0.0011 | −0.0054 | −0.0036 | −0.0164 |
| EXT | all | −0.0042 | −0.0057 | −0.0045 | −0.0051 | −0.0034 | −0.0056 |
| EXT | **strong** | **+0.0056** | +0.0042 | +0.0117 | +0.0089 | +0.0048 | −0.0013 |

Per-symbol strong-signal entry **does** sharpen the short-horizon IC (HL70 +0.0101 @4h vs +0.0018
all; EXT flips from −0.0042 to +0.0056 and stays positive to ~24h). This is the most promising
single number in the diagnostic — a per-symbol magnitude entry has a real, if small, h=4 edge that
the full grid dilutes. But (a) it still decays to ~0/negative by 24h on HL70, and (b) prior
magnitude/dispersion gates "mostly didn't add honest value" (memory ledger), and (c) the decisive
test below shows the variable-hold built on this heterogeneity fails nested-OOS.

## 4. DECISIVE TEST — variable-hold vs fixed-24h vs G4 matched placebo + nested-OOS

Oracle-ish variable rule: each cycle holds its strength-tercile's in-sample-best horizon.

| panel | fixed-24h spread | variable spread | avg hold (var) | turnover proxy (var vs fixed) | **G4 placebo rank** | **NESTED-OOS lift** |
|---|---|---|---|---|---|---|
| **HL70** | +0.4 bps | +9.4 bps | 14.7h | 0.118 vs 0.042 (MORE churn) | **p100** (placebo p95 +6.2) | **−45.2 bps (CATASTROPHIC)** |
| **EXT** | +9.7 bps | +17.7 bps | 45.3h | 0.032 vs 0.042 (less churn) | **p97** (placebo p95 +16.4) | **+3.3 bps (noise)** |

- **In-sample, variable beats fixed AND beats the matched-avg-hold G4 placebo** (p100/p97). This is
  exactly the trap the contract warns about — a tuned win that clears the placebo in-sample.
- **Nested-OOS kills it.** Pick best-horizon-per-tercile on the first half, apply forward: HL70
  tercile map flips from in-sample {weak:24,mid:4,strong:16} to first-half {weak:12,mid:72,strong:12},
  and the forward variable rule **loses −45.2 bps** vs fixed-24h (2nd-half fixed +4.6 → variable −40.6).
  EXT forward lift +3.3 bps = noise. **The optimal hold-per-tercile does not generalize in time.**
- **Turnover:** on HL70 the variable rule (avg 14.7h) actually trades MORE than fixed-24h (the
  fast-decay terciles want short holds) — so even the "trade less" premise fails on production; you'd
  pay more cost for a forward-losing rule. On EXT it trades slightly less but the edge is noise.

This is the **same failure mode** as cost-margin swap (nested +0.24 vs +1.88 in-sample), decay
sleeves (CI crossed 0), and the iter-014 longer-hold lever (production nested-OOS churns): an
untuned discrete architecture (fixed K, fixed hold) survives; a *tuned/selected* parameter (which
horizon per condition) embeds in-sample optimization that fails honest forward selection.

---

## Honest verdict — NOT worth building

1. **Signal decays fast** (HL70 SIDE IC peaks h=4, zero-cross ~h12, negative ≥24h). An event-driven
   engine designed to "hold longer to capture more signal" has **nothing to capture past ~12h** — the
   24h sleeves hold stale/anti-signal, confirming the human's hunch that the hold is cost-amortization
   only. But that does NOT imply a variable engine wins, because:
2. **Heterogeneity is real but NOT exploitable.** Decay shape varies by tercile, but the mapping is
   non-monotone, panel-inconsistent, and **fails nested-OOS hard on the production universe** (−45 bps
   forward). The variable rule clears the in-sample G4 placebo (p100/p97) and then dies forward —
   textbook overfit.
3. **The honest structural lever is already known and characterized:** iter-014 — a *longer FIXED*
   hold cuts DD via cost amortization (S44/EXT clean; HL70 fails G5/G6/nested-OOS). Event-driven /
   variable adds nothing on top and costs an engine build plus a forward-losing tuned parameter.
4. **The one live thread:** per-symbol strong-|pred| entry sharpens the h=4 IC (HL70 +0.0101 vs
   +0.0018; EXT flips positive). If anything in this space is ever revisited it should be a *per-symbol
   magnitude ENTRY gate at the native 4h horizon* (NOT a variable hold), pre-checked against G4 — but
   the memory ledger already shows magnitude/dispersion gates mostly fail honest value, so the prior
   is low.

**Recommendation: do NOT build the event-driven / variable-horizon engine.** The fixed 4h-entry /
24h-hold grid is already capturing the (near-zero, fast-decaying) signal as well as the data allows;
the 24h hold is justified as cost-amortization, not signal capture, and its mild post-24h stale-signal
drag on HL70 is the price of the smooth turnover. Variable-horizon's win condition (exploitable
heterogeneous decay) is **absent under honest validation**.

Artifacts: scripts `iter016_decay.py`, `iter016_varhold.py`;
`research/convexity_portable_2026-05-20/results/iter016_decay_summary.json`.
