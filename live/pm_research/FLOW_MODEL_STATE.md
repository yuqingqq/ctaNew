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
| **Market self-excitation is not deletable** | bivariate 2×2: diagonal 0.18–0.45 dominates cross 0.02–0.18 on every coin | see §3 on why this is not a validated value |

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

- **Queue position.** The fill bracket *is* the queue-position ambiguity.
  Cancellation cannot reduce it: cancellation **volume** is abundant (86–99 % of
  actions saturated) while cancellation **position** is the same missing
  quantity. Credit all → `FRONT`; credit none → `BACK_DISPLAYED`; the interior
  needs an assumption, not a bound.
- **Sub-millisecond structure.** The venue timestamps in **milliseconds**. Our
  `recv_ns` is stamped at parse time and manufactures a 0–50 µs pile-up
  (26 µs median, 16.2× Poisson on btc). No collector change fixes this — the
  data is not there. The Hawkes grid is floored at **10 venue ticks**.
- **Own impact, acknowledgement delay, hidden liquidity.** All require placing
  orders. The 2.3–12.1 % of trade volume with no matching displayed decrease
  (SOL one share in eight) is consistent with hidden liquidity and is not
  separable passively.
- **Branching *values*.** `RETAIN` means not-deletable-on-this-evidence. It is
  not a validated estimate; intervals were grid-quantised until 2026-08-21 and
  showed fit stability, not sampling uncertainty.

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

1. **Nothing about the fill bracket.** It is structural. Either proceed under an
   explicit queue-position assumption with sensitivity across FRONT↔BACK, or
   treat it as the programme-ending answer. That is a decision, not a
   measurement.
2. **Layer retention** needs ~10 forward days in one collector era. Days only.
3. **The maker-edge sign** needs ~25–30× current data. Days only, and many.
4. **`U9`** re-runs itself unchanged when `PING_TIMEOUT` reaches n=12.

**Collecting more days does not touch item 1**, which is the item that gates the
programme's central question.
