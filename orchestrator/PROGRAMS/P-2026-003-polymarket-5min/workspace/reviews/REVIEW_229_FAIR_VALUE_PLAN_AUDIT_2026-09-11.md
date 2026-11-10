# REVIEW 229 — independent check of `fair_value_plan.md` v1.2 (`0575444`): the load-bearing stream claim verifies from the data, ETH does not exist in the pipeline, and one "missing" item already has a producer

**REV, 2026-09-11T18:26Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Facts audited against the code and the data, not the plan's
judgement.

## THE THREE THINGS THE SEATS SHOULD READ BEFORE THEY BUILD

1. **ETH does not exist in the derived pipeline.** §2 makes the combined **BTC+ETH** UTC day
   the primary statistical unit and §8 gates on 95% native coverage **for each coin**. There
   are **51 day-books, none `_eth`**; 106 `_btc` derived artifacts against 4 `_eth`, and those
   four are old `harmful_scores_eth_*.jsonl.gz`, not books, tapes or fragments. The *input*
   exists (`markets.jsonl` carries 6,667 mentions of each coin); the *build lane* does not.
   **Nothing in steps 1–6 can produce an ETH day as the tree stands.**
2. **"Missing item 1 — a fixed point-in-time sigma producer" already exists**, and BE is
   building step 2 against it as if it did not. `live/pm_research/be_sigma_30m.py:104`
   `sigma_30m(ticks, decision_recv_ns, *, era_floor_ns=…)` is point-in-time **by signature**,
   returns a typed record, refuses a non-integer decision time, and
   `sigma_30m_for_symbol(symbol, decision_recv_ns)` already drives it from bookTicker ticks.
   Beside it: `sigma_kernels.py` (738 lines) and a protocol/results corpus —
   `SIGMA_ROUTE_A_PROTOCOL.md`, `SIGMA_ROUTE_A_V2_PROTOCOL.md`,
   `SIGMA_ROUTE_A_RESULTS_2026-08-20.md` and five review iterations. **What is missing is the
   wiring, not the producer.** Building a second sigma is the two-validators defect that
   `microprice_from_book`'s own docstring warns about ("fix one, leave its twin").
3. **§8's floor of 8 nonzero days tolerates zero disagreeing days** — see §4 below. The plan
   states the floor and not its consequence, and that is the exact reading I got wrong on the
   forward test.

---

## 1. §3 "ALREADY BUILT" — BOTH CLAIMS HOLD, ONE VOCABULARY CORRECTION

**`pm_microprice` with Identity's exact admissibility population — TRUE, and by construction
rather than by coincidence.** `microprice_from_book` (`da_fair_price_identity.py:350`) calls
`identity_from_book` for the verdict and, on any non-OK status, returns
`replace(base, estimator=MICROPRICE)` — same refusal, same cause, different name. Its docstring
gives the reason the plan relies on: two validators could disagree about which instants are
admissible and "the pairing would silently compare different populations".

*The one qualification, and it is a strengthening:* the microprice adds two refusals Identity
does not have — a convex-combination invariant and a probability-range check. Both are
**provably unreachable** given the checks that precede them (`PROB_LO, PROB_HI = 0.0, 1.0` and
both sides already validated into that range; the code itself marks the size guard
"unreachable while min_depth > 0" and keeps it anyway). So the populations are identical, and
where they could differ the difference is arithmetic that cannot occur.

**`bn_bookticker_s60_probability` exists as described** (`:667`): **BUILD ONLY, NOT SCORED** in
its own docstring, `reference_source` forced to the Chainlink relay and `spot_source` forced to
bookTicker — both by raising `Inadmissible`, not by comment — with `sigma` a required input.

**The vocabulary correction.** §3 calls it a *"moment-matched-average transformation"*. The
code's estimand is **`P(S60(T) >= S60(t0))`** — two sixty-second endpoint averages — with **two
regimes**: for `t <= T-60` the realized past is **irrelevant** and `partial` MUST be `None`
("a pure forecast"); for `t > T-60` `partial` is required. The docstring records that an
earlier claim of its own ("a challenger must carry the realized average at every instant") was
**false under the true convention**, with a selftest that feeds two different pasts and
requires the same answer. A reader of §3's one-liner would not learn there is a regime switch
at `T-60`. **DE's step-3 wrappers must not assume a single regime.**

## 2. §3 "STILL MISSING" — THREE ABSENT, ONE OVERSTATED, AND ONE CLAIM THE CODE DOES NOT MAKE

| # | claim | audit |
|---|---|---|
| 1 | a fixed point-in-time sigma producer | **OVERSTATED** — see the header. A producer exists; the wiring does not |
| 2 | a wrapper to the typed `FairPrice` without collapsing the two times | **absent** — `FairPrice(` is constructed in exactly one file, its own definition |
| 3 | a policy seam that consumes the value | **absent** — nothing reads it |
| 4 | a scorer and a full replay comparison against Identity | **absent** |

**And one factual claim that the code does not support:** *"The current trajectory exporter
explicitly refuses the fair-price composition."* It does not refuse. `be_trajectory_export.py`
takes `fairprice_estimator=None` as a **defaulted parameter** and writes it straight into the
object; there is no refusal anywhere in the file. **The plan's very next sentence — "An unused
schema field is not a partial implementation" — describes the code correctly.** Delete the
claim and keep the sentence.

**What else steps 1–6 assume and the tree does not have:** the ETH lane (header), and a
`FairPrice` consumer of any kind — item 3 is not one gap but the join between 2 and 4, and
nothing in the tree reads the record today.

## 3. §2's STREAM CLAIM — VERIFIED FROM THE DATA, AND NOW QUANTIFIED

Checked at the recorded streams, not at a comment.

```
crypto_prices_twap_sixty  payload STATES  "window_s": 60   symbol "btc/usd"
crypto_prices             CSV             symbol "btcusdt" (Binance convention)
2026-09-11 17:00Z, BTC:   3,601 raw ticks   3,514 TWAP messages   (~1 Hz each)
```

**The decisive test — does the recorded raw stream reproduce the published 60 s statistic?**

```
| 60s rolling mean(crypto_prices) - crypto_prices_twap_sixty |, n = 1,000
      median  $14.57      p90  $17.02      max  $20.07      = 187 ppm of price
for scale: median tick-to-tick move in crypto_prices = $0.01
```

**It does not.** The residual is ~1,450× the tick size, so `crypto_prices` is **not** the
process whose 60 s mean is published. **§2's claim holds and its consequence follows**: the
recorded data does not expose a model-free realized integral of the settlement path, and any
Binance integral inside the terminal minute is a cross-venue proxy.

**And the audit adds a number the plan should carry.** The offset is a persistent level
difference, not noise — a **venue basis of ~187 ppm on BTC** at this hour. §2 labels the
Binance path a "proxy"; for a five-minute at-the-money binary, 187 ppm is not obviously small
relative to the move being predicted, and the plan should state the measured basis rather than
only the word.

## 4. §8's ARITHMETIC — ALL CORRECT, AND ONE CONSEQUENCE UNSTATED

```
2^10 = 1,024                                    correct
minimum two-sided p = 2/1024 = 0.001953125      correct
Holm m = 2  ->  adjusted 2 x 0.001953125 = 0.00390625 < 0.05        ATTAINABLE
             (equivalently 0.001953125 < alpha/2 = 0.025)
2^8 = 256 >= the plan's own 200-draw null minimum                   CLEARS IT
```

**And 8 is exactly the smallest count that clears it** — `2^7 = 128 < 200`. The threshold is
well chosen and the plan's arithmetic is right throughout.

**The consequence the plan does not state.** The tolerance for a single disagreeing day depends
sharply on how many nonzero days accrue:

```
 8 nonzero:  min p = 2/256  = 0.0078125   one wrong day -> 2*9/256  = 0.0703  > 0.05   UNANIMITY REQUIRED
10 nonzero:  min p = 2/1024 = 0.0019531   one wrong day -> 2*11/1024 = 0.0215 < 0.025  SURVIVES
```

**At the floor the test tolerates zero disagreements; two days above the floor it tolerates
one.** That is the same tolerance-0 structure that ended the forward test — and it is the
reading I got wrong there (REVIEW 226 §1), which is why it is worth writing into the plan
before anyone builds on it.

**One thing §8 gets right that the forward test did not**, and it deserves saying: §4 declares
a **closed** candidate family, C1 and C2. `m = 2` therefore covers the candidates actually
looked at, not a screen behind them. The forward test's `m = 2` sat behind a 69-candidate
screen and could not have delivered evidence at any outcome (REVIEW 226 §6). **This design does
not have that defect** — provided §4's family stays closed, which is a discipline question, not
an arithmetic one.

## 5. RANKED, FOR THE SEATS BUILDING NOW

1. **ETH** (header §1) — DA/BE/DE are building steps 1–3 toward a statistical unit the pipeline
   cannot produce. Either the ETH lane is built, or §2's unit and §8's per-coin gate change.
   **This is the one that invalidates the plan's population if it is left.**
2. **The sigma producer exists** (header §2) — BE should read `be_sigma_30m.py` and
   `SIGMA_ROUTE_A_PROTOCOL.md` before writing step 2, or the programme gets two sigmas.
3. **The two regimes at `T-60`** (§1) — DE's step-3 wrappers.
4. **§8's unanimity-at-the-floor** (§4) — into the plan text, before the accrual rule is frozen.
5. **The 187 ppm basis** (§3) — into §2, as a measured number.
6. **The exporter does not "explicitly refuse"** (§2) — a two-word correction.

## SCOPE

Closed over: `da_fair_price_identity.py`'s microprice delegation and the S60 transformation's
guards read to the line; the four "missing" items searched by symbol across `live/`; the two
price streams read from disk and the rolling-mean reproduction computed over 1,000 points of a
real hour; §8's arithmetic recomputed including the tolerance table; the ETH question answered
by counting derived artifacts and day-books. **Not closed over:** the plan's judgement, which
is not mine to audit; §§5–7 and §§9–11, which this round did not name; whether the 187 ppm
basis is stable across hours or coins — one hour, one coin, one day.

## ROUTED

1. **Coordinator — ETH is the blocker** (§5.1), and it is a population question, not a build
   question.
2. **BE — the sigma producer exists** (§5.2).
3. **DE — two regimes, not one** (§5.3).
4. **Whoever holds the plan — four text corrections** (§5.4–6).
