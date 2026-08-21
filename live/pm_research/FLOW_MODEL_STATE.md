# FLOW MODEL — CURRENT STATE

**Read this first, and read nothing else to find out what we currently believe.**
Updated 2026-08-21. Supersedes every other flow document *for the question "what
is true now"*. The others remain valid as provenance — how we got here, and why
particular things were withdrawn — but they argue with each other by
construction, because they were written in sequence as claims were corrected.

If a statement anywhere else conflicts with this page, **this page wins** and
the other document is stale. Say so rather than reconciling it privately.

---

## 1. Established, with scope

| fact | evidence | scope |
|---|---|---|
| **`side` is the taker's** | G-FF1 `PASS`, 600/600, Wilson [0.9936, 1.0000] | 300 BUY / 300 SELL, 7/7 coins, 5/5 moneyness |
| **Taker pays the fee, maker does not** | `0.07·p·(1−p)` $/share to 4 dp; 600/600 taker legs charged, 744/754 maker legs zero | n=600 transactions |
| **Crossing costs ~2.25 ¢/share ATM** | 0.50 ¢ half-spread + 1.75 ¢ fee ≈ **225 bps** on a $1 binary | btc/eth |
| **Settlement is `S60(T)` vs `S60(t0)`** | 99.8 % on 1,465 windows; the 300 s reading is refuted at 86.9 % | all |
| **ATM spread is 1 tick on btc/eth** | median 0.0100, p90 0.020, 2.29 M executable quotes | btc; ticks by coin: btc 1 · eth 1 · sol 3 · doge 3 · xrp 5 · bnb 5 · hype 7 |
| **The 1-cent spread is a CONSTRAINT, not a convention** | where the 0.001 tick is available the spread is 1 tick in 99.9 % of quotes | btc |
| **0.001 tick exists only in the tails** | 6.75 % at `p<0.15`, 6.73 % at `p≥0.85`, **0 % in the middle three buckets** | btc |
| **One address is a large share of arrivals** | 0.02 shares exactly, 99.98 % SELL, all seven coins, **0.0145 % of notional** | **per coin 2.0 % (btc) → 90.0 % (hype)**; the pooled 16.3 % is btc-dominated |
| **That address is NOT independent of market flow** | A1 fails on all 7; ratio 1.75–2.79, p=0.000; direction **bidirectional/common-driver** | τ=0.25 s, circular-shift null |
| **`f_r` does not rise into settlement** | count: flat then terminal collapse. notional: **rises, peaks in the second half, then falls** | 945 windows, `clob_v3_1` |
| **The terminal regime is FEW AND LARGE** | count drops to 18 % of peak while notional holds 28 %; USDC/arrival 15.5 → 24.0 (btc), 11.9 → 32.3 (eth) | btc/eth |
| **Book state carries real information** | B3 placebo does not reproduce the gain on any coin (btc −0.03 share, hype 0.02) | 24 windows/coin |
| **Market self-excitation is not deletable** | bivariate 2×2: diagonal 0.18–0.45 dominates cross 0.02–0.18 on every coin | scalar fit re-run below; the BIVARIATE fit is still grid-seeded |
| **Clustering runs at 80–218 ms, and the censoring was OUR GRID** | scalar Hawkes re-fit 2026-08-21 with the instrument floor + continuous optimiser: **no coin is censored**, branching 0.33–0.55, half-life 80.8 ms (btc) to 217.7 ms (hype) — **81×–218× the venue tick** | 24 windows/coin, `clob_v3_1` |

## 2. Measured and UNDETERMINED — not negative, not positive

- **The maker-edge sign.** `+0.173 ¢/share [−0.251, +0.596]` pooled; **all seven
  per-coin CIs span zero on both weightings**; permutation p=0.0482 names no
  coin. Resolving it needs roughly **25–30× current data** — over a month — and
  day-clustered intervals (the correct unit, not yet computable) would be wider.
- **`PING_TIMEOUT` missingness class.** `MNAR-suspect` stands at n=7 of a
  required 12. It is **49 % of all lost time**.
- **The maker rebate `ρ`.** No per-trade in-transaction rebate found; that is
  **not** absence of a rebate. Every `ρ`-dependent estimand stays `Unavailable`.
- **`f_r`'s terminal mechanism.** The settlement TWAP lock-in is the natural
  explanation and is **confounded with a 60 s artefact** — window phase and
  wall-clock minute phase are perfectly collinear here.

## 3. Not knowable on this data — do not schedule work against these

- **Queue position CANNOT BE INFERRED FROM THE TAPE** — displayed L2 shows how
  much size sits at a level, never whose or in what order, and cancellation
  cannot recover it (we see 40 shares leave, not *which* 40). Cancellation
  **volume** is abundant, 86–99 % of actions saturated; cancellation
  **position** is the missing quantity.

  **CORRECTED 2026-08-21 — but this does NOT mean fill is undeterminable, and an
  earlier version of this page said it did.** Queue position is an **OUTPUT OF
  THE PLACEMENT POLICY**, not an unknown of nature. Quote a level as it forms
  (new-BBO) and you are at the front; join an existing level and you are behind
  its displayed depth. So `FRONT`/`BACK_DISPLAYED` is **the span across placement
  policies, not an epistemic bracket**, and it collapses to a definite number the
  moment a policy is named. **The strategy defines the measurement; it is not
  downstream of it.**

  What genuinely remains unobserved is narrower: **whether a new-BBO quote
  actually wins the race** against other participants doing the same. That
  depends on our latency and their behaviour, neither of which is in this tape.
  So `q ≈ 0` is an **upper bound on the new-BBO policy**, not a guarantee, and
  the interior of the bracket is reachable by losing races rather than by
  ignorance.
- **Sub-millisecond structure.** The venue timestamps in **milliseconds**. Our
  `recv_ns` is stamped at parse time and manufactures a 0–50 µs pile-up
  (26 µs median, 16.2× Poisson on btc). No collector change fixes this — the
  data is not there. The Hawkes grid is floored at **10 venue ticks**.
- **Own impact, acknowledgement delay, hidden liquidity.** All require placing
  orders. The 2.3–12.1 % of trade volume with no matching displayed decrease
  (SOL one share in eight) is consistent with hidden liquidity and is not
  separable passively.
- **Branching *values* — SUBSTANTIALLY IMPROVED 2026-08-21, and one earlier claim
  is now dead.** The scalar fit was re-run with the instrument floor and a
  continuous optimiser. **Every coin now selects an INTERIOR half-life; none is
  censored**, btc included:

  | coin | branching | half-life (op) | half-life (wall) |
  |---|---:|---:|---:|
  | btc | 0.554 | 0.6935 | **80.8 ms** |
  | xrp | 0.343 | 0.1141 | 97.6 ms |
  | eth | 0.509 | 0.2209 | 115.3 ms |
  | doge | 0.358 | 0.0746 | 130.1 ms |
  | bnb | 0.434 | 0.0858 | 168.6 ms |
  | sol | 0.403 | 0.1485 | 183.4 ms |
  | hype | 0.325 | 0.1143 | 217.7 ms |

  **The earlier "btc is censored at ~36 ms, which is order-splitting" reading is
  WITHDRAWN. That was our grid reaching below venue resolution, not a market
  fact.** Excluding the two sub-resolution grid points on btc, the fit selects a
  sensible interior value. Real clustering sits at **80–218 ms** on every coin —
  reaction-time scale, not slicing and not our processing cadence. So the floor
  did not merely flag the problem; it resolved it.

  **STILL OUTSTANDING:** the **bivariate** C2 fit (the 2×2 type matrix) has NOT
  been re-run with the floor or the optimiser — it lives in `queue_and_type.py`,
  which was untouched. Its btc entry is still censored and its intervals are
  still grid-quantised. Treat the 0.18–0.45 diagonal as provisional until it is
  re-fitted the same way.

## 4. Withdrawn — if you find these anywhere, that document is stale

| claim | replaced by |
|---|---|
| `+0.45 ¢/share` maker markout | `+0.17`, and that spans zero |
| `+95 bps maker gross / +136 with rebate` | the same number as above; falls with it |
| "on real flow, makers lose per fill" (`−0.211`) | spans zero `[−0.849, +0.457]` |
| wide spreads price adverse-selection hazard | unsupported; only "width does not predict edge" survives |
| "ATM runs 6–8 c" | refuted for btc/eth; true only for thin coins |
| "16.3 % of events" unqualified | pooled and btc-dominated; 2.0 %–90.0 % per coin |
| "count layer available / volume layer blocked" | **inverted** — volume is unblocked, count is contaminated |
| the FLB edge | 0.0004 Brier, one-sided; measured on stale books |

## 5. Binding rules

- **R-DUAL, per coin.** Above ~35 % micro share the **raw** count is a
  participant measurement, not market flow. `verdict_coins` for fills are
  **btc and eth**; the rest are descriptive only.
- **Delete nothing; label and condition.** A1's failure kills ex-micro
  *deletion* quantities and *vindicates* the multi-type model. Notional
  weighting reweights rather than deletes and needs no independence assumption.
- **State the population of every denominator.** Six instances of that defect so
  far; three read as findings before they were caught.
- **The name is not the definition.** Four instances, one self-inflicted.
  Confirm any field, file or contract label against the code that writes it.
- Read book state from `price_change.best_bid/ask`, never `book` snapshots
  (p90 6.2 s stale). Knowledge time is `recv_ns`. Never pool across
  `collector_version` eras.

## 6. Live artifacts

| | |
|---|---|
| governing protocol | `FLOW_MODEL_PROTOCOL_V4.yaml` (V3 is `governs: false`) |
| specification | `FLOW_MODEL_SPEC_REV2.md`, `plans/BE_FLOWANDFILLS_MODEL_PLAN.md` |
| probes | `flow_intensity.py` · `flow_fill_development.py` · `flow_uncertainty.py` · `queue_and_type.py` |
| results | `FLOW_INTENSITY_RESULTS.md` · `FLOW_FILL_DEVELOPMENT_RESULTS.md` · `QUEUE_AND_TYPE_RESULTS.md` |
| ledger | `FLOW_UNCERTAINTY_LOOP.md` |

Everything else under `live/pm_research/*.md` is **provenance**: correct about
its own moment, not a statement of current belief.

## 7. What is next, and what blocks it

1. **The fill bracket is a POLICY COMPARISON, not a pending measurement.**
   Specify concrete placement policies — new-BBO (front) and join-BBO (back) at
   minimum — and measure fill **and fill-conditional markout for each**. That
   yields a comparison of real strategies instead of a bracket over an unknown.
   Expect the two to trade off rather than rank: new-BBO wins on fills (94.6 %
   vs 76.9 % on btc) and plausibly **loses on markout**, because it quotes when
   the level is forming, thin, and information is freshest. Measuring only the
   fill side would flatter it.
2. **Layer retention** needs ~10 forward days in one collector era. Days only.
3. **The maker-edge sign** needs ~25–30× current data. Days only, and many.
4. **`U9`** re-runs itself unchanged when `PING_TIMEOUT` reaches n=12.

**Collecting more days does not touch item 1**, which is the item that gates the
programme's central question.
