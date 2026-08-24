# Optimization iteration 004 — hysteresis half-band result

**Verdict: `REJECT` for BTC and ETH. Research only; decision eligible: no.**

Protocol: `HYSTERESIS_HALF_BAND_PROTOCOL.md`. Implementation:
`policy_optimizer_hysteresis_half_band.py`. Receipt:
`data/pm_5min/derived/policy_optimizer_hysteresis_half_band_v1.json`. Artifact
ID: `f142630125362f6737954eb015d14366d99bacef0f10fcec7910923dabd7d011`.

The candidate retained fixed 0.55/0.45 q hysteresis and changed only the skew
band from 5.0 to 2.5 shares, exactly half one quote.

| coin | Aug 23 delta vs incumbent | Aug 24 delta | dev2 candidate PnL | dev2 `QR_SKEW_ONLY` | terminal abs inventory: incumbent -> candidate | effective cancels: incumbent -> candidate | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| BTC | +12.33 c | +141.04 c | +36.23 c | -52.88 c | 15.81 -> 16.75 | 772 -> 123 | REJECT |
| ETH | -167.72 c | -32.63 c | -46.64 c | -65.62 c | 1.25 -> 1.24 | 663 -> 390 | REJECT |

BTC again passes both PnL days, the corrected-skew bar, and both traffic bars,
but misses the unchanged inventory bar by 0.94 mean terminal shares. ETH now
passes inventory but loses to the incumbent on both development days. Neither
coin is adopted.

No further threshold or skew-band variant will be selected on these two visible
days. The loop routes next to an action-conditioned model whose training
population matches actual queue-realistic JOIN, inventory role, and stateful
policy eligibility.

Verification: 13 new checks, eleven parent-arm parity, five-share degenerate
parity, deterministic replay, artifact self-hash, and code/protocol provenance
pass.
