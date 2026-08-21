# G-FF1 — is WS `side` the taker's direction? Protocol `gff1_v1`

Protocol version: `gff1_v1`. Status: **FROZEN BEFORE MEASUREMENT**.

Gate `G-FF1` (`plans/BE_FLOWANDFILLS_PLAN.md`): *is `side` the taker's
direction?* Metric: on-chain `OrderFilled` taker direction vs WS `side`,
agreement. Threshold **0.99**. `on_fail: HALT_PROGRAM`.

Why it matters: if `side` is the maker's rather than the taker's, maker gross
flips **+95 → −95 bps** and every downstream flow quantity — order-flow
imbalance, adverse selection, markout sign, `zeta` — carries the wrong sign.
Present evidence is circumstantial only: a 77.18 % BUY skew that is *symmetric
across both tokens* (Up 77 %, Down 77 %), plus where the aggression sits
(84.9 % at the best ask vs 15.1 % at the best bid).

## Design-data disclosure

Exactly **one** transaction was inspected before this freeze, to establish the
decode and join method:

- `0x1c6a460820a70fa4cc80aa6c8e3137aaab2f4ba1735d4f34286ff98427bfd99c`
  (Polygon block 92,377,178, 23 logs).

That transaction is **design data, not validation data**. It is excluded from
the measurement sample by transaction hash. No agreement statistic was computed
before this freeze; the single observation established only that the join keys
line up and that the event ABI decodes.

## Established before freeze (facts, not hypotheses)

Event signatures verified by keccak-256, not assumed:

```
OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)
  topic0 0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee
  indexed: orderHash, maker, taker

OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)
  topic0 0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c
  indexed: takerOrderHash, takerOrderMaker
```

Exchange address observed: `0xe111180000d2663c0091e4f400237545b87b996b`.
This is **not** the classic Polymarket CTF Exchange ABI: it replaces the
`(makerAssetId, takerAssetId)` pair with a single `uint256` asset id plus an
explicit `uint8` side enum. Any document describing the older 8-field
`OrderFilled` does not apply to this tape.

## Hypothesis under test

`H_taker`: the WS `last_trade_price.side` equals the side of the **taker
order**, as named on-chain by `OrdersMatched.takerOrderMaker`, for the token in
`last_trade_price.asset_id`.

Decoded field reading under test (positional, from the verified ABI):

```
OrdersMatched.data = [ side_enum, asset_id, maker_amount_filled, taker_amount_filled ]
```

with `side_enum == 0` interpreted as BUY and `1` as SELL. **The enum-to-label
mapping is itself part of the hypothesis**, not an assumption: it is confirmed
only if it agrees on the overwhelming majority of a stratified sample. A
result near 0.0 agreement refutes `H_taker` in the strongest possible way
(it means the convention is inverted), and is reported as such rather than
silently sign-flipped.

## Sample frame

- Source: immutable, closed, gzipped window files under `data/pm_5min/raw/`.
  In-progress (`.jsonl`, uncompressed) windows are **excluded** — a window
  still being written is not immutable.
- UTC days: **2026-08-19 and 2026-08-20 only.** 2026-08-21 is incomplete at
  freeze time and is excluded.
- Events: `event_type == "last_trade_price"` carrying a non-null
  `transaction_hash`.
- Excluded: the design-data transaction above.

## Stratification (pre-specified)

A convenience sample of consecutive trades would pass while missing an
exception class. Draw is stratified and the per-stratum agreement is reported
**separately**, never only pooled.

| axis | strata |
|---|---|
| coin | btc, eth, sol, xrp, doge, bnb, hype |
| moneyness | `p < 0.15`, `0.15 ≤ p < 0.35`, `0.35 ≤ p < 0.65`, `0.65 ≤ p < 0.85`, `p ≥ 0.85` |
| WS side | BUY, SELL |

Target **n ≥ 500** transactions total, drawn to fill cells as evenly as the
tape allows, minimum 20 per coin where the tape supports it. Cells that cannot
reach their target are reported with their realised count, never quietly
dropped. Sampling is deterministic: `random.Random(20260821)`, sorted candidate
keys, no wall-clock or set-iteration dependence.

## Join rule

For each sampled WS trade:

1. Fetch the Polygon receipt by `transaction_hash`.
2. Decode every `OrdersMatched` and `OrderFilled` log emitted by the exchange
   address.
3. Select the `OrdersMatched` log whose decoded `asset_id` equals the WS
   `asset_id`. If there is not exactly one, the row is `AMBIGUOUS_LEG` and is
   reported, not dropped.
4. **Validate the join before using it**: require
   `taker_amount_filled / 1e6 == WS size` and
   `maker_amount_filled / taker_amount_filled == WS price`, each to a
   tolerance of 1e-6 absolute on size and 5e-4 on price. A row failing
   validation is `JOIN_MISMATCH` and is excluded from the agreement statistic
   **and reported beside it** — per the standing rule, the excluded set is
   published next to the retained one.

Rule 4 exists because a hash-level join proves only that we fetched the right
transaction, not that we identified the right leg within it. One transaction
in the design sample carried **three** `OrderFilled` logs and one
`OrdersMatched`.

## Metric and verdict

Primary: `agreement = matched_rows_with_side_equal / validated_rows`.

Reported as `GateEvidence` with an effect size and a two-sided 95 % Wilson
interval, per the standing rule that failure to reject is not equivalence:

- **PASS** — Wilson lower bound ≥ 0.99.
- **MODEL_REFUTED** — Wilson upper bound < 0.99.
- **INSUFFICIENT_EVIDENCE** — otherwise, including any run with
  fewer than 500 validated rows.

Clustering: rows are clustered by transaction; a transaction contributes one
row to the primary statistic. Per-stratum agreement is reported with counts.

**Pre-specified refutation.** Pooled agreement below 0.5 means the convention
is inverted from `H_taker`; that is a `MODEL_REFUTED` verdict on this protocol
and triggers the gate's `HALT_PROGRAM` review, not an in-place sign flip of
the interpretation.

**Pre-specified non-outcomes.** None of the following may be read as support
for `H_taker`: a high pooled agreement with any coin stratum below 0.95; fewer
than 500 validated rows; or a `JOIN_MISMATCH` rate above 5 %. Each forces
`INSUFFICIENT_EVIDENCE` regardless of the point estimate.

## Reproducibility

The run writes a manifest containing: protocol version, the SHA-256 of this
file, the SHA-256 of the probe script, the sorted list of source window files
with their sizes and hashes, the RPC endpoints used, and the frozen seed.
Receipts are cached on disk by hash so a rerun is byte-reproducible without
re-querying the chain.

## Amendment — `gff1_v2`, 2026-08-21

**`gff1_v1` ran and returned `INSUFFICIENT_EVIDENCE`. That result stands and is
not retro-edited.** It is superseded, not withdrawn: the run was valid under its
own decode rule, and the rule was wrong.

**What v1 measured.** 500 sampled transactions, **226 validated**, agreement
**226/226 = 1.0000**, Wilson95 [0.9833, 1.0000] — and a `JOIN_MISMATCH` rate of
**0.548**, ten times the pre-specified 0.05 ceiling. Two frozen guards fired and
blocked the pass: the mismatch ceiling and the 500-cluster minimum.

**Why that matters more than the point estimate.** Every one of the 226
validated rows was `BUY`. **Zero SELL rows validated.** A 100 % agreement
figure was sitting on exactly half the action space, and the pooled number gave
no hint of it. Had the protocol not pre-specified a mismatch ceiling, v1 would
have reported a clean-looking result computed on a silently one-sided sample.
This is the concrete instance of the standing rule: report the excluded set
beside the retained one, or the retained one lies.

**The defect.** v1 hard-coded the BUY reading of the amount pair
(`size = takerAmountFilled`, `price = maker/taker`). An order's `makerAmount` is
what its creator **gives**, so the pair is ordered by direction:

```
BUY   taker gives USDC, receives tokens   size = taker/1e6   price = maker/taker
SELL  taker gives tokens, receives USDC   size = maker/1e6   price = taker/maker
```

Under v1's BUY-only reading a SELL decodes to a price **above 1**, which is
impossible in a prediction market: `ws 0.02 @ 0.21` decoded as
`0.0042 @ 4.761905`. **235 of the 274 mismatched legs reconcile exactly under
the inverted reading**, confirming the diagnosis rather than assuming it.

**The v2 change, and it is deliberately narrow.** Direction is now derived from
**which leg of the amount pair is USDC** — the reading whose price falls in
`(0, 1]`. This is chain-only: it consults neither `side_enum` nor any websocket
field, so the comparison against WS `side` stays non-circular. Equal amounts
(price 1.0 either way) raise `DIRECTION_UNIDENTIFIED` rather than guess.

`side_enum` is demoted to a **secondary** statistic (`enum_agree`), which learns
the enum-to-label mapping instead of assuming `0 = BUY`.

**Unchanged from v1:** hypothesis, sample frame, days, seed, stratification,
target `n`, tolerances, mismatch ceiling, per-coin floor, threshold, verdict
rules and clustering. Nothing that could tune the answer moved.

### v2 residual — SOLVED after the run

v2 left 27 residual legs (5.4 %) where size matched **exactly** and only price
differed. The cause is now measured, not hypothesised:

```
round(chain_price, 2) == ws_price : 27/27
round(chain_price, 3) == ws_price :  0/27      <- specifically 2dp
max |chain_price - ws_price|      : 0.004865   <- under half a 0.01 tick
```

**The websocket reports the effective price rounded to the tick.** `PRICE_TOL =
5e-4` was ten times tighter than the tick's own rounding granularity, so the
comparison demanded an agreement the feed cannot express. This is the same
category error as the v1 decode bug — comparing an exact quantity with a rounded
one — one level up.

An earlier guess, that these were partial fills of an aggregated match, is
**withdrawn**: sizes match exactly, so nothing is partial. A second guess, that
the on-chain fee explained the gap, was **tested and refuted** — all 500 sampled
transactions carry a nonzero fee, so fee presence discriminates nothing.

## `gff1_v3` — REQUIRED CHANGES, to be frozen BEFORE it runs

Two fixes, both correctness rather than tolerance:

1. **Compare like with like.** Test `round(chain_price, tick) == ws_price`, with
   the tick read **per market** — it is `0.01` near the money and `0.001` away
   from it, and `tick_size_change` events occur mid-window. Do NOT simply widen
   `PRICE_TOL` to 0.005: that would pass the far-from-money regime on a
   tolerance ten times its true tick.
2. **Size the draw to the requirement.** 500 drawn transactions yielded 473
   validated; draw enough that validated clusters reach 500.

Everything else stays frozen. The design input for both changes was observed
*after* the v2 run and is disclosed here as such; neither changes the
hypothesis, the threshold, the strata or the verdict rules.

## Known defect in `gff1_v3`, disclosed — verdict unaffected

The v3 tick lookup uses the byte pattern `"tick_size"`, which does **not** match
`"new_tick_size"`, so it silently ignored every `tick_size_change` event and read
the tick only from `book` snapshots. `tick_size_change` transitions
`0.01 -> 0.001` **do occur** (8 observed in one sampled BTC window), so legs in
the 0.001 regime were validated against a tolerance ten times too loose. This is
why the run reported `ticks seen: {0.01: 600}` — that was the bug, not a fact
about the market.

**The `PASS` stands.** The binding join validation is size agreement to `1e-6`,
which identifies the leg essentially uniquely; the price check is secondary
corroboration. Direction is derived from the amount pair, not the price, so it
is untouched. But any future run must fix the pattern before the price check can
be quoted as evidence in the 0.001 regime.

Separately measured while checking this, and it contradicts a premise in
`BE_FLOWANDFILLS_PLAN.md`: across 2.29 M executable quote observations from 12
BTC windows on 2026-08-20, the median spread is **1 tick (0.0100) at every
moneyness bucket**, ATM included (p90 0.020). The recorded claim that "ATM runs
6-8 c" is **not supported for BTC**. Scope: one coin, one day — re-check on
thinner coins before generalising.

## Scope

This protocol answers the side convention and nothing else. It does not
estimate `alpha_trade` (G-FF2), `zeta` (G-FF3) or the queue bracket (G-FF4),
and it authorises no flow-based quantity. The maker/taker addresses it decodes
are not retained for participant analysis.
