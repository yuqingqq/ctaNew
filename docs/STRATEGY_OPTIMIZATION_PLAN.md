# Strategy Optimization Plan

This plan treats the documented strategy as the baseline and reviews the
workflow from first principles:

```text
net edge = prediction spread - turnover cost - slippage - funding - risk errors
```

The main objective is to improve net portfolio PnL after costs, not feature IC
or model score in isolation.

## 1. Freeze the Baseline

Define one canonical baseline before testing improvements:

- Feature set
- Universe
- Horizon and rebalance cadence
- Portfolio construction rule
- Fee, slippage, and funding assumptions
- Data source
- Model artifact

Record baseline metrics:

- Gross alpha
- Net PnL
- Turnover
- Fees
- Slippage
- Funding
- Hit rate
- Drawdown
- Sharpe
- Per-symbol contribution

All later changes should be compared against this exact baseline.

## 2. Add Strategy Diagnostics

Break every rebalance cycle into:

- Prediction spread
- Realized gross alpha
- Turnover
- Taker fees
- Slippage
- Funding
- Factor exposure
- Net PnL

Add per-symbol diagnostics:

- Alpha contribution
- Cost paid
- Turnover
- Holding time
- Spread/depth at trade time
- Funding paid or received
- Realized slippage

This makes it clear whether losses come from weak prediction, excess churn,
execution cost, funding drag, or unintended factor exposure.

## 3. Cost-Aware Rebalancing

Do not trade just because model ranks changed. Trade only when the expected
benefit of changing the portfolio exceeds the expected cost.

Decision rule:

```text
expected_alpha_gain > expected_fee + expected_slippage + funding_cost + safety_margin
```

For a switch from current symbol A to candidate symbol B:

```text
alpha_gain = predicted_alpha(B) - predicted_alpha(A)
switch_cost = exit_cost(A) + entry_cost(B)
trade only if alpha_gain > switch_cost + margin
```

Practical implementation:

- Add a no-trade band.
- Use stricter thresholds for new entries than for keeping current positions.
- Keep existing positions if their signal is still acceptable.
- Penalize turnover directly in the target portfolio objective.
- Allow partial moves toward the new target instead of forced full rebalance.

Example hysteresis rule:

- Enter long only if rank is in top 5.
- Keep long until it falls below rank 10.
- Enter short only if rank is in bottom 5.
- Keep short until it rises above bottom 10.

The goal is to avoid paying fees and slippage for marginal rank changes.

## 4. Dynamic Position Sizing

Move beyond equal-weight top/bottom buckets.

Start with:

```text
weight proportional to clipped_zscore(prediction) / volatility
```

Then add costs:

```text
weight proportional to expected_alpha / (volatility * expected_cost)
```

Constraints:

- Max weight per symbol
- Max gross exposure
- Dollar neutrality
- Beta/factor neutrality
- Liquidity cap
- Minimum trade size
- Turnover cap

The aim is to allocate more capital to strong, cheap, liquid signals and less
capital to weak or expensive signals.

## 5. Portfolio Optimizer

Replace fixed top-K selection with an optimizer:

```text
maximize:
    predicted_alpha(position)
  - transaction_cost(position_change)
  - risk_penalty(position)
```

Subject to:

- Dollar-neutral portfolio
- Market beta neutrality
- Gross exposure limit
- Per-symbol cap
- Liquidity cap
- Minimum trade size
- Turnover cap

This should be tested against the fixed top-K baseline using identical cycles
and identical costs.

## 6. Risk Model

Reduce unintended factor bets by measuring and constraining exposures to:

- Market basket
- BTC beta
- ETH beta
- High-beta alt factor
- Sector/group factors if available
- Volatility regime

Use the risk model either as hard constraints or as a penalty term in the
portfolio optimizer.

## 7. Execution Model

Estimate execution cost per symbol and per cycle from:

- Spread
- L2 book depth
- Order size
- Recent volume
- Volatility
- Funding

Compare execution styles:

- Immediate taker execution
- Post-only maker execution
- Split execution over several minutes
- Skip trading in thin books

The execution choice should be made from expected net edge after costs, not
from prediction rank alone.

## 8. Robustness Tests

Before accepting an optimization, test:

- All rebalance phase offsets
- Different volatility regimes
- Trend versus chop regimes
- Liquidation/stress periods
- Fee stress
- Slippage stress
- Funding stress
- Missing-symbol scenarios
- Bootstrap confidence intervals on cycle PnL

Reject changes that only work in one phase, one regime, or one narrow sample.

## 9. New Features Only After Portfolio/Execution Fixes

Do not prioritize more kline indicators until the portfolio and execution layer
is cost-aware.

If adding features, prioritize orthogonal information:

- L2 order book imbalance
- Depth and spread dynamics
- Funding, basis, and open interest
- Spot-perp divergence
- Liquidation flow
- Cross-venue pressure
- On-chain exchange flow

Accept a feature only if it improves net portfolio PnL after costs.

## Recommended Order

1. Freeze and reproduce the baseline.
2. Add cycle and per-symbol diagnostics.
3. Implement cost-aware rebalancing and no-trade bands.
4. Add dynamic position sizing.
5. Replace fixed top-K with a portfolio optimizer.
6. Add a factor risk model.
7. Build an execution cost and maker-fill simulator.
8. Re-run robustness tests.
9. Only then evaluate new orthogonal features.

The most likely near-term gains are from reducing unnecessary turnover,
sizing by signal quality and tradability, and lowering execution cost.
