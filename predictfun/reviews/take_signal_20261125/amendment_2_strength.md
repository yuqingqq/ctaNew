# TAKE-SIGNAL v1 — amendment 2: signal-strength curve (declared before any new outcome is read)

USER (2026-11-25): "have we measured the signal strength, like if the trend is how strong, we can actively take". The primary
packet measured four strength bins, all at or above the guard edges. This amendment measures the full curve, the break-even
strength, a strength in cents that is comparable across time and price, and longer BTC trends. Same data, the same Tokyo execution
and valuation code as `take_eval.py` (imported, not copied); consumed Oct 24–27; descriptive.

## Events and features (all causal at the decision t = E + 5 ms)

- Every BTCUSDT bookTicker mid change inside a market window [ws, END) with |bn_r250| ≥ 0.05 bps; direction = sign(bn_r250).
- `bn_r250`, `bn_r1000`, `bn_r3000` (bps).
- σ300 = RMS of the 300 trailing whole-second BTC log returns (bps per √s), read from the step function at the decision clock.
- τ = seconds to expiry (we − t). z = Φ⁻¹(clip(known mid, 0.01, 0.99)).
- Sensitivity = 100·φ(z) / (σ300·√τ), in cents per bps (the binary's delta).
- **Implied move IM = sensitivity × |bn_r250|**, in cents in the event direction.
- **Cost** = s·(limit − known mid)·100 + fee·100 (half-spread paid plus the taker fee), in cents.
- **Signal edge = IM − cost**.
- Trend context: CONTINUATION if bn_r3000 has the event's sign and |bn_r3000| ≥ |bn_r250|. REVERSAL if bn_r3000 has the opposite sign. OTHER otherwise.

## Valuation

`take_eval`'s Tokyo clock and fill rule, unchanged: known book at t − 63 ms, limit = known touch, arrival t + 42 ms, two-frame
executability. Q = 50; fee 200 bps × min(p, 1 − p). Values per filled share at 1 / 3 / 10 s, taker round trip at 3 s, END and
resolution. Parity control: the subset that the guard's own rule selects must reproduce `takes.parquet` (tokyo) row for row before
anything else is reported: UP at ≥ 0.2855 and DOWN at ≥ 0.4667, OFF→ON with the 2 s cooldown.

## Tables and readings

1. **Per-event curve** (events overlap inside bursts; descriptive), by fine |bn_r250| bins:
   [0.05, 0.1), [0.1, 0.2), [0.2, 0.2855), [0.2855, 0.5), [0.5, 0.75), [0.75, 1), [1, 1.5), [1.5, 2), [2, 3), ≥ 3.
   Also by IM bins ([0, 0.5), [0.5, 1), [1, 1.5), [1.5, 2), [2, 3), [3, 5), [5, 10), ≥ 10 ¢) and by signal-edge bins
   (< −1, [−1, 0), [0, 1), [1, 2), [2, 5), ≥ 5 ¢). Each table is reported for Oct 24–25 and Oct 26–27 separately.
2. **Threshold-rule curve:** for θ ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.2855, 0.35, 0.4667, 0.5, 0.75, 1, 1.5, 2, 3} bps, the rule
   "take when |bn_r250| ≥ θ", OFF→ON with a 2 s cooldown per market × direction. Report triggers, fill rate, values per filled
   share, and total USD marked at 3 s at Q = 50, per half.
3. **Trend context** × coarse strength (< 0.2855, 0.2855–1, ≥ 1 bps) at 3 s.

Readings, computed:

- (a) **Break-even strength:** the lowest |bn_r250| bin whose mark_3s is > 0 in BOTH halves, with all higher bins also > 0 in both.
- (b) Whether mark_3s rises monotonically across the fine bins (pooled).
- (c) **Calibration:** whether the realised mark_3s is > 0 for signal edge ≥ 0 and ≤ 0 for signal edge < −1, in both halves.
- (d) Whether CONTINUATION beats REVERSAL at 3 s in every coarse strength band, in both halves.

No threshold is adopted. Any threshold read off this curve is a candidate that needs later untouched days (CLAUDE.md rule 11).
