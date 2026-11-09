# P-2026-005: Predict.fun BTC 5-minute market making (research)

Adapting P-2026-003's PM strategy (D1, `pmmmrust/docs/D1_SPEC.md`) to Predict.fun's
BTC Up/Down 5-minute markets. Research only: public market data, no wallet, no
order placement. Execution work belongs to a separate project.

State: `STATUS.yml` + `HANDOFF.md` in this folder (single seat, one writer).
Background on the venue and collectors: `docs/PREDICT_FUN_RESEARCH.md` (repo root).

## Capture mechanics study (2026-11-09)

The research quote planner tests retaining a small order and appending a top-up
when size grows. A paired no-fill workload on 220 consumed Nov8 windows found
incremental sizing reduced requests 4.52% and increased active time at intended
prices 3.90%, as-of 2026-11-09 14:42:01 UTC. Always splitting orders increased
requests; entry buffers removed substantial desired quote time. These results
describe order mechanics, not validated fills, PnL or inventory variance.
See [the comparison and port handoff](handoffs/20261109_incremental_capture_results.md)
for controls, limitations, reproduction and immutable receipts. Live code and
frozen fill engines were not changed.

## Replay timing repair (2026-11-09)

The exact-order audit found that the J/X trade clocks can fall outside an order's
possible lifetime. REST `executedAt` follows confirmation; anonymous pending
increments and book drops cannot identify an exact matching time. Previous PnL
comparisons remain conditional on that failed clock model.

The current input path preserves full transaction, taker, maker and settlement
identities. `pf_trade_tape_build` rebuilds the already-consumed Nov8 audit cohort
into a separate directory, retaining every unresolved trade as a status.
Matched-status timestamps remain one-second intervals, separate from confirmation
and local receipt times. No delay fitting, lifecycle clipping, cross-leg clock
substitution, price rounding or confirmation-time fallback is allowed.

`pf_step4_ladder_identity DAY SCENARIO ARMSET` is the replacement replay entrypoint;
`PF_TRADE_TAPE` names its manifest directory. It checks the complete identity and
clock population before touching strategy inputs/output. It currently **refuses
the available historical tape**, which lacks precise match events. An actual
matching-event source and validation of both filled and unfilled actions are
needed before resuming strategy/PnL ranking. A filled-order interval check alone
does not calibrate queue placement or fill probability.

The old timing estimators and forward drivers now stop with an explicit reason.
Frozen engine files and previous receipts remain unchanged; historical reproduction
belongs on the earlier commit. Do not consume the protected forward days under
the superseded timing assumptions. Repair evidence is under
`reviews/trade_timing_fix_20261109/`; details are in `HANDOFF.md`.

## Mode

**Exploratory** (user, 2026-11-01): no held-out split, freeze or forward rules yet.
Results here are descriptive, not validated. Every day read is still recorded in
`STATUS.yml` (`days_read`) so untouched days can be named later if a test is frozen.

## Plan (user, 2026-11-01)

1. Reconcile sources and measure book structure (3 sample days).
2. Check the signals D1 relies on, as they behave on Predict.
3. Run a plain baseline (no D1 rules) on the same days.

## Data (read-only inputs)

| Tape | Path | Since |
| --- | --- | --- |
| Predict books / status / matches / categories | `data/predict_fun/runs/20261021T162823Z_0dd45f2d5a/` | 10-21 16:28Z |
| Predict reference relay (Chainlink BTC/USDT, unsigned) | `data/predict_reference/runs/20261022T132901Z_8c043d17b6/` | 10-22 13:29Z |
| Binance USD-M futures bookTicker | `data/mm_hf/raw/bookTicker/BTCUSDT/` | existing |

Outputs: `data/predict_fun/research/<step>/<day>/` (gitignored).

## Run

Every job goes through the guarded launcher (one process, core 8, 4 GiB cap,
refuses below 8 GiB free memory or 100 GiB free disk; logs measured peak RSS and
disk delta to `research/run_light.log`):

```bash
predictfun/research/run_light.sh <tag> -- python3 -m predictfun.research.pf_step1 20261023
```

## Venue facts (verified from the tape, 2026-11-01)

| Item | Value |
| --- | --- |
| Settlement | Chainlink BTC/USDT = Binance **spot** top-of-book mid; end = close at window end; strictly higher = Up; equal = 50/50 |
| Start / end price | `variantData.startPrice` / `endPrice` in category metadata |
| Tick | 0.01 (`decimalPrecision 2`) |
| Taker fee | 0.02 x min(p, 1-p) per share (`feeRateBps 200`); some takers pay 90% (invite discount) |
| Maker fee | 0 (`makerFeeBps 0`); 25% taker-fee rebate advertised for Up/Down, not verified as paid |
| Liquidity rewards | per-window schedule `hourlyRate 3000`, eligibility `spreadThreshold 0.06`, `shareThreshold 100` (units/payout unverified) |
| Min order | about $1 notional (inferred from resting sizes), unverified |
| Book feed | WS full-depth snapshots ~10-20/s per market, ms `updateTimestampMs`, `settlementsPending` side book |
| Trades | REST only: maker order hash/signer/price/amount, `executedAt` whole seconds, first seen p50 ~7 s late |
| Up token | indexSet 1 on every checked market |
