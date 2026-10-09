# Extreme-pump mirror: does an independent market-wide pump exist to fade? (2026-10-09)

**Question.** Mirror of `docs/EXTREME_CRASH.md`: after ≥ half the liquid Binance USDT perps rise
≥ X within 15 minutes, does shorting the hardest-pumped fifth for 4 h pay?

**Calibration first, outcome-blind** (`python3 -m research.extreme_pump calibrate`).
- Window: discovery 2022-01-02 → 2025-08-30. Universe and LIQ are the flush detector's.
- Fires: first minute of each run, 24 h refractory.
- "After a crash" = within 24 h after a crash-side WATCH minute (median coin down ≥ 3% in 15 min).
- No post-trigger return was computed.

| share ≥ 50%, rise ≥ | fires | after a crash | independent |
|---|---|---|---|
| 3% | 95 | 59 | 36 |
| 5% | 26 | 23 | 3 |
| 7% | 10 | 10 | 0 |
| 10% (mirror of EXTREME) | 6 | 6 | 0 |
| 15% | 0 | 0 | 0 |

With share ≥ 25%: 3% 129 / 55 independent; 5% 33 / 5; 7% and above 0 independent.

Daily peak share of coins up ≥ 5% within 15 minutes, over 1,336 days:

| peak share | < 1% | 1–5% | 5–10% | 10–25% | 25–50% | ≥ 50% |
|---|---|---|---|---|---|---|
| days | 662 | 574 | 39 | 27 | 8 | 26 |

## What it says
- **There is no independent extreme market-wide pump.** Every rise of 7% or more across half the
  market in 2022–2025 came within 24 h of a crash: it is the crash's snap-back.
- So the mirror trade is not a separate strategy. Shorting those rebounds would bet against the
  validated crash basket during its own holding period.
- **Not tested:**
  - The mild version (half the market up ≥ 3%, independent; 36 fires in 3.7 years) is a different,
    weaker idea. The crash-side analogue (WATCH) showed no edge.
  - The rebound-fade version cannot be tested cleanly: the crash windows' post-trigger paths have
    already been read.
- No outcome was read, so 2020–2025 stays untouched for the pump question.
