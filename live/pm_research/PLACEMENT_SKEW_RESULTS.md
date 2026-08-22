# Placement skew, and whether the one-book identity generalises

Protocol: the two tests named by `plans/DE_PLACEMENT_POLICY_PLAN.md` as its own
first tests. Probe `placement_skew.py` (22 self-test checks). Research only, not
decision eligible, no forward-day claim, **no edge or PnL claim** — the edge
estimand is broken (settlement markout measures hold-to-expiry drift), so this
is about the inventory process and the identity only.

Receipts: `data/pm_5min/derived/placement_skew_{t1,t2}.json`.

---

## T2 — the one-book identity: `IDENTITY_HOLDS`, and it is EXACT

The plan named this the **most likely falsification of the whole design**.
It does not falsify. It is stronger than the design assumed.

| | |
|---|---|
| archives scanned | **560** — 7 coins × 4 UTC days × 20 |
| checks | **1,081,800** |
| violations at 0.005 | **0** |
| **worst absolute deviation** | **0.00000** |
| mean absolute deviation | **0.00000** |

Both identities were tested on every check: `bid(Up)+ask(Down)=1` and
`ask(Up)+bid(Down)=1`.

**By day** — every day 1.000000:

```
20260819  224,510 checks   0 viol      20260821  304,728 checks   0 viol
20260820  248,472 checks   0 viol      20260822  304,090 checks   0 viol
```

**By stress condition** — zero violations in all of them:

```
0.001 tick regime      8,468 checks    0 violations
terminal minute       23,828 checks    0 violations
during collector gaps                  0 violations
within 5 s of a tick-size change       0 violations
```

Per coin, all seven at 1.000000. btc contributes no 0.001-regime checks, which
is consistent with the recorded finding that the 0.001 tick lives only in the
tails.

**The result is stronger than "holds within tolerance".** The worst deviation
across 1.08 M checks is **exactly zero**, not merely under 0.005. That is not two
books that happen to agree — it is **one book with the second side derived
arithmetically**. The design consequences all stand and are now on firmer ground
than when they were asserted: the state is one scalar, skew and
complement-quoting are one mechanism, and a complete set is worth exactly one
spread.

**Method note.** Checks are **within-message only**. A single `price_change`
payload carries both tokens, so there is no staleness to confound the identity;
comparing quotes across messages would measure our own read latency instead.
Population: every 8th `price_change` line per archive, capped at 2,500 checks per
archive.

---

## T1 — placement skew: `SKEW_SUFFICIENT` on both verdict coins, **as an upper bound**

Four policies, **paired on the same 25 windows per coin and the same decision
times**: `JOIN` symmetric (the published baseline), `NEW` symmetric (the
published 9.4×-risk case), `SKEW` (reducing side fronted past a 5-share band),
and `SKEW_IDEAL` (same, but permitted to jump an existing queue on flip).

| coin | JOIN p95 | SKEW p95 | IDEAL | NEW p95 | cut | JOIN half-life | SKEW half-life | JOIN $ | SKEW $ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **btc** | 194.6 | **21.4** | 13.2 | 1805.1 | **89.0 %** | none | **6 s** | 121.80 | **8.11** |
| **eth** | 92.0 | **20.0** | 17.5 | 305.0 | **78.3 %** | 1021 s | **8 s** | 44.66 | **2.87** |
| hype | 22.4 | 9.9 | 9.9 | 95.8 | 55.6 % | none | 317 s | 4.19 | 4.19 |
| xrp | 32.4 | 15.0 | 15.0 | 145.3 | 53.9 % | 1694 s | 34 s | 20.46 | 4.90 |
| sol | 25.0 | 14.4 | 14.4 | 239.6 | 42.2 % | 793 s | 23 s | 24.57 | 3.87 |
| bnb | 15.4 | 10.0 | 10.0 | 90.1 | 34.9 % | none | 161 s | 7.03 | 4.45 |
| doge | 15.0 | 10.0 | 10.0 | 86.8 | 33.3 % | none | 227 s | 8.33 | 0.62 |

**btc and eth carry the verdict: `SKEW_SUFFICIENT`.** Terminal `|net|` p95 falls
89 % and 78 %, cash at risk falls **15×** and 15×, and the implied half-life
collapses from "no reversion detected" and 1021 s to **6 s and 8 s** — well
inside the 300 s window. hype is the one coin where the realistic and idealised
arms disagree (`SKEW_HELPS_INSUFFICIENT` vs `SKEW_SUFFICIENT`).

### The coordinator's hypothesis is refuted — in the measured arm

The brief expected the 1.23× fill-rate lever (94.6 / 76.9) to be far too small to
move a 519 s half-life inside 300 s. It is not, and the reason is that **the lever
is not applied once — it is a persistent feedback bias.** The skew engages
whenever `|net|` exceeds the band and keeps biasing fills toward reduction, so a
1.23× directional advantage compounds across hundreds of fills per window. A
drift is not a scaled-up reversion rate; it is a different object, which was the
point of the correction that prompted this test.

### **THE LIMITATION THAT BOUNDS EVERYTHING ABOVE**

`SKEW_SUFFICIENT` is an **upper bound, not an achievable result.**

When a fronted side is fully lifted it **re-posts at `queue_ahead = 0`** — first
in the queue again, immediately. That is the same idealisation `NEW_BBO` already
carries, and it is documented there as an upper bound because being first
assumes winning a race whose latency is not observable in this tape. `SKEW`
inherits it, and inherits it **often**: btc runs ~476 fills per window under
`SKEW`, so the assumption is exercised hundreds of times per window rather than
once. `JOIN` pays the displayed queue on every re-post; the fronted side never
does.

Two things bound how much this matters, and they point in opposite directions:

- **Reassuring:** `SKEW_IDEAL` — which additionally permits jumping an existing
  queue at the moment of a flip — is barely better than `SKEW` (btc 13.2 vs 21.4)
  and *identical* on five of seven coins. So the **flip** idealisation is not
  driving the result.
- **Not reassuring:** the flip is not the generous part. The **re-post** is, and
  both arms share it, so their agreement cannot test it.

**The untested arm is the lower bound:** front only on genuine level
re-formation, and re-join the back after every lift. That has not been measured
and is the first thing that should be, before any of the above is acted on.

### A second finding, not in the pre-registered rule

**Skew does not merely redirect fills — it increases them by ~40 %.** btc goes
from 4,249 buy / 4,287 sell under `JOIN` to 5,934 / 5,975 under `SKEW`
(17,313 → 24,508 shares bought). Both sides rise, because as `net` crosses the
band in each direction, each side spends time fronted. More fills means more
spread capture *and* more gross exposure; whether that is favourable cannot be
answered here, because the edge estimand is broken.

### Verdict rule as pre-registered

`SKEW_SUFFICIENT` required a ≥20 % cut in terminal `|net|` p95 **and** an implied
half-life inside 300 s. `SKEW_HELPS_INSUFFICIENT`, `SKEW_INEFFECTIVE`,
`SKEW_HARMFUL` and `UNRESOLVED` were all reachable, with **underpowered
defaulting to `SKEW_INEFFECTIVE`** — keeping the dump mechanism. A half-life of
exactly 300 s reads as insufficient, pinned by a self-test.

---

## Controls

The harness had to be shown to measure what it claims, in three ways:

1. **`SKEW` with an infinite band reproduces `JOIN` exactly** — identical `net`
   series on every window tested, so the skew arm differs from the baseline only
   because of the policy.
2. **`placement_skew` `JOIN` reproduces `inventory_walk`'s published baseline
   exactly**, and `NEW` reproduces its `front=True` arm exactly (btc −1046.29 on
   both). The new probe is not a reimplementation with different behaviour.
3. **A live skew genuinely differs** from `JOIN` (btc 14.34 → 9.08 on the first
   window), so the policy is not a no-op.

Self-test additionally pins: an extreme long never fronts the *adding* side; a
tolerance-boundary deviation is not a violation while one just over it is; and
the underpowered branch defaults to retaining the dump mechanism.

---

## Against `FLOW_MODEL_STATE.md`

**Nothing is contradicted, and one entry is sharpened.**

§1c records that no coin is self-balancing and that the dump mechanism is not
deleted. That remains correct: it describes the **uncontrolled** process, and T1
measures a **controlled** one. The two are consistent, and the distinction is
exactly the drift-versus-reversion point. §1c should not be read as bounding what
a control can achieve.

`DE_PLACEMENT_POLICY_PLAN`'s falsification #5 — that the 1.23× lever may be too
small — is **refuted in the measured arm** and remains open in the untested
lower-bound arm.

## Scope, and one defect found in a shared helper

25 windows/coin, era `clob_v3_1`, one era inside the collected days. Window-
clustered bootstrap at 400 resamples; **no day-clustered interval is computable**
at this day count, so all intervals understate uncertainty. Per coin, never
pooled.

**`flow_intensity.DAYS` is stale.** It reads `("20260819", "20260820",
"20260821")` and omits `20260822`, which has **1,141 archives on disk**. Anything
routed through `fi._archive_paths()` — including T1's window pool via
`inventory_walk.select` — silently excludes the most recent day. T2 enumerates
the days independently and therefore covers all four. This is outside the scope
of these two tests but affects any probe importing that constant.
