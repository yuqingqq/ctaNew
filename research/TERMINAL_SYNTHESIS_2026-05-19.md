# TERMINAL SYNTHESIS — Autonomous Research Loop (2026-05-19)

The pre-registered → 3-agent-plan-review → test → 3-agent-results-review →
re-initiate loop has run to its **honest terminal state**: every
genuinely-distinct, in-scope (free 4h Binance-perp data) hypothesis is now
closed by measurement or by review. This is the conclusion, not a stop.

## What was tested and closed (each by our own measurement, not docs)

| Arc | Mechanism | Verdict | Evidence |
|---|---|---|---|
| Portable-alpha (return forecast) | directional CS residual | **−0.33 on unseen symbols** | R3c (reproduces prior Test-3 −0.39) |
| Bottleneck: features/model/harness | best-learner on superset | no detectable portable lever | B★/B★b Δ −0.58/−0.89/−0.90, CIs∋0; univ IC ≤0.036 |
| In-universe +2.23 | meme-convexity | non-portable, ~5-name, rotates VVV→AXS→PENDLE | R1c |
| Convexity-mining (long cohort) | identify convex names | **vol detector, no direction** | C0pre full-panel AUC up0.67≈dn0.70≈abs0.75 |
| "Just long the convex" | harvest skew long | negative-EV (theory + C0pre + ex-VVV) | review-closed |
| Sell-convexity (short cohort) | sell skew premium | KILLED — re-derives −0.33 short leg; vol-detector relabel; funding sign | 3-agent DO-NOT-PROCEED |
| Orthogonal OI + aggTrade-flow | free positioning/flow data | no detectable portable lift (corrected v2) | oi_flow v2, all CIs∋0, Ridge-OI within-noise |
| Funding-carry | carry premium | KILLED — re-derivation (WINNER_21 feature, rejected ≥3×) + tautology + premium<cost-floor | 3-agent DO-NOT-PROCEED |

## The honest bottom line
There is **no portable, in-scope, free-data cross-sectional alpha**. The
production +2.0–2.2 is real in-universe but is a fragile, non-portable,
~5-effective-name meme-convexity lottery (R4 Option-B): it does not replay on
a different symbol set (−0.33), and is ~62–80% one name (VVV). Every named
free-data premium — momentum/reversal (IC≈0.02 ceiling), vol/skew (vol
detector, no direction), order-flow/OI (no portable lift), carry (re-derived,
sub-cost-floor) — has now been isolated and measured. The discipline
repeatedly caught over-claims (portable-alpha v1, convexity AUC-0.68,
sell-convexity, the OI/flow cartesian bug, funding-carry tautology) **before**
they became false conclusions — the process worked.

## Genuine byproducts (not alpha)
1. A real, portable **volatility detector** (C0pre OOS-symbol AUC ~0.67–0.75)
   — risk-control use only (size-down/avoid names primed for a big move).
2. Honest understanding of the +2.23 mechanism (convexity, not skill) and a
   sized fragility profile (drop-VVV, unseen-universe, kill-switch −6,265 bps).
3. A hardened, leak-audited test harness (R3c portable protocol; corrected
   within-group aggregation; per-group level-CI + LOFO; PIT recompute gate).

## Why the loop terminates here (not arbitrary)
All 3 reviewers, independently, concluded the genuinely-distinct in-scope
space is exhausted: every further candidate reduces to (a) a closed arc, (b)
the −0.33 pred-ranked short leg, or (c) a **scope change**. Continuing to
generate in-scope "new" plans would itself be the re-derivation / goalpost-
moving the user explicitly forbade. Honoring "don't stop" by fabricating
re-derivations would violate the deeper instruction (validate, don't
re-derive). The aligned action is this terminal report + a user scope
decision — not iteration theater.

## The only un-refuted levers (all are SCOPE CHANGES — a user decision)
1. **Paid point-in-time orthogonal data** (on-chain/Glassnode, options
   positioning). Prior: *pessimistic* — every free analogue (OI, flow,
   funding) came back null; the >11 cohort-spread bar (ethbtc 8.58,
   xs_ret_disp 7.18) was not cleared by anything free. Justified only via a
   cheap single-symbol trial slice that must clear that bar before any
   subscription.
2. **Different horizon / market structure** — the literature's *portable*
   microstructure edge lives at **HFT (~seconds) with order-book data**, not
   4h aggregated free data. Different data + infra; large commitment.
3. **Different asset class / venue** (lower-cost or higher-funding venues;
   non-perp). Out of current codebase scope.
4. **Accept the terminal negative** — stop research spend; treat the
   in-universe meme-convexity book as at most a tiny kill-switched niche bet
   (R4 Option-B), explicitly not the profitable-portable system that was the
   goal; reconcile/retire the K=4 paper bot so it can't mislead.

Operational item regardless: `live/vBTC_paper_bot.py` ships K=4/no-sleeve —
validates neither the research stack nor any negative; retire or reconcile.
