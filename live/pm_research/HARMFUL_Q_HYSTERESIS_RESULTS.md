# Optimization iteration 003 — harmful-q hysteresis result

**Verdict: `REJECT` for BTC and ETH. Research only; decision eligible: no.**

Protocol: `HARMFUL_Q_HYSTERESIS_PROTOCOL.md`. Implementation:
`policy_optimizer_harmful_q_hysteresis.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_harmful_q_hysteresis_v1.json`. Artifact
ID: `f4abe2473d28e530db8fc4b29475836fac088949cef78ae37b47d6a0e4ca025e`.

The candidate replaced the incumbent's q>0.5 state with a fixed per-side 0.55
entry / 0.45 exit deadband. There was no threshold sweep.

| coin | Aug 23 delta vs incumbent | Aug 24 delta | dev2 candidate PnL | dev2 `QR_SKEW_ONLY` | terminal abs inventory: incumbent -> candidate | effective cancels: incumbent -> candidate | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| BTC | +12.33 c | +264.59 c | +98.00 c | -52.88 c | 15.81 -> 18.44 | 772 -> 143 | REJECT |
| ETH | -189.75 c | +21.39 c | -30.64 c | -65.62 c | 1.25 -> 3.11 | 663 -> 413 | REJECT |

BTC passes four of five adoption bars: it improves PnL on both development
days, beats corrected skew, and sharply lowers cancellation traffic. It fails
the frozen inventory bar, so the loop does not adopt it. ETH also reverses by
day and fails inventory.

The next iteration keeps the hysteresis unchanged and tests one inventory
control change: reduce the skew band from one full 5-share quote to half an
order, 2.5 shares. This is a single preregistered value, not a band sweep.

Verification: 12 new checks, exact raw-q reconstruction, degenerate q=0.5
incumbent parity, ten parent-arm parity, deterministic replay, artifact
self-hash, and code/protocol provenance pass.
