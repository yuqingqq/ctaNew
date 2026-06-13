# Lifecycle / Mechanism Probe — FINDINGS (2026-05-19, data-driven)

User pushback (legitimate): (1) pump-and-dump has temporal structure → maybe
direction IS knowable conditional on "big move coming"; (2) ex-VVV-still-
positive must have a *mechanism*, not luck — mine the data.

## What the data shows

**Mechanism (point 2) — found, NOT luck.** Event-time path of entered LONG
winners (cum % vs entry t0): pre-entry mild drift decelerating into entry
(+1.28%@−24h → +0.05%@−2h → 0); post-entry **+0.68%@+2h → +1.47%@+4h →
plateaus +1.4–1.6% to +24h**. Losers: flat pre, −1.38%@+4h, stays ≈ −1.1%.
⇒ Mechanism = **weak short-horizon momentum/continuation tilt on
high-volatility names, amplified by large realized move size (±~1.4%/cycle).**
Persists ex-VVV because the *tilt* generalizes across the high-vol cohort and
rotates to whatever name has the signature — not VVV-specific. Real,
repeatable, characterized.

**Conditional direction (point 1) — partially true, TINY, not "up-then-down".**
Within PIT top-decile-vol cohort, OOS-symbol directional accuracy of next-4h
beta-neutral residual sign:
- `r24` (24h momentum) **0.515** vs placebo 0.501 (per-grp 0.51–0.53, stable);
  `runup_z` 0.512; all others (funding_z, dist-from-high, r7d, idio_max,
  r72) ≈ 0.50 = noise.
- Winners' post-entry path **plateaus, does NOT dump within 24h** → the
  "pump *then* dump, known sequence" structure is NOT borne out at 4h.
  It is mild momentum *continuation*, not a straddle-able up-then-down.
- ex-VVV winner contribution by `r24` quintile: 0.25/0.23/0.14/0.16/0.23 —
  spread, not concentrated in one pump-phase bucket (consistent with
  "thin tilt across many names × convex magnitude", not "rotate to the
  early-pump name").

## Honest conclusion (data confirms terminal state, with a concrete mechanism)
51.5% directional accuracy = exactly the closed IC≈0.02 ceiling, now made
visible. Quantified: 1.5pp edge × ±1.4% moves ≈ ~4 bps/cycle gross vs ~9 bps
round-trip cost ⇒ **structurally sub-cost-floor**; magnitude depends on the
universe containing VVV-class big movers ⇒ why it ports to −0.33. The
mechanism is a thin momentum-in-volatile-names edge amplified by convex move
size — real, not luck — but it IS the closed ceiling, not a new lever.

Spinning a "conditional-momentum-in-vol-cohort" strategy is REJECTED a priori:
`r24`/momentum are WINNER_21 features (closed ceiling); the data shows the
edge is sub-cost-floor at 51.5%; a 3-agent review would correctly flag it as
re-derivation. Running it = re-derivation theater the discipline forbids, and
the data predicts net-of-cost failure.

⇒ TERMINAL_SYNTHESIS_2026-05-19.md stands, now grounded in a measured
mechanism rather than "convexity/luck". No new genuinely-distinct in-scope
lever. Continuation requires the user-decided scope change.
Artifacts: `probe.py`, `probe_results.json`.
