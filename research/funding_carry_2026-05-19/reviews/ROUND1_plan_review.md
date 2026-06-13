# Funding-Carry Plan v1 — 3-Agent Review (2026-05-19) — KILLED + LOOP TERMINAL

Verdicts: Methodology **NEEDS-REVISION** · Profitability **NEEDS-REFOCUS
(run-once-terminal-then-stop)** · Red-team **DO-NOT-PROCEED + terminate loop**.

Convergent decisive findings:
1. **Re-derivation.** `funding_rate`,`funding_rate_z_7d` ARE WINNER_21
   features; the closed B★/R3c stack had funding info and ported to −0.33;
   funding tested-and-rejected ≥3× (METHODOLOGY_REVIEW v7→2.80;
   AGGTRADE_NEGATIVE_RESULT −0.99; vBTC_STATUS Phase L.6 funding_disp REJECT).
   S0 distinctness gate (linear corr<0.30 vs pred) too weak — WINNER_21's
   funding value is non-linear (atr×funding tree splits), so a standalone
   funding rank can pass S0 yet harvest the same closed premium.
2. **Tautology.** Strategy ranks by `funding_rate` and books `funding_rate`
   as P&L; panel funding autocorr ac_8h≈0.49 → a positive is the sort key
   re-summed, not a tradeable premium. No decomposition guards this.
3. **Premium < cost floor.** Cross-sectional funding spread ~2.5 bps/8h vs
   4.5 bps/leg; biggest |funding| on the lowest-float names where √ADV is
   worst. P(portable net ≥ +0.5) ≈ 5%; plan's own prediction (net ≤ +0.3)
   concedes it can't clear its gate.
4. **Power:** +0.5/LCB>0 on ~5-group/0.74y with ~3 bps signal → MDE > signal;
   pre-registered to fail or auto-respawn (unfalsifiable-by-construction).
5. **Meta (all 3, independent):** the genuinely-distinct in-scope (free-data)
   hypothesis space is **exhausted**. Every remaining idea = a closed arc,
   the −0.33 short leg, or a scope change. Continuing to spin in-scope
   plans IS the re-derivation the discipline exists to prevent.

## Decision
Funding-carry NOT run (DO-NOT-PROCEED; tautological + re-derivation +
sub-cost-floor). This iteration = a completed honest negative. The autonomous
loop has reached its **honest terminal state**: no genuinely-distinct in-scope
hypothesis remains; further "iteration" would require a user-decided scope
change, not in-scope re-derivation. → write TERMINAL_SYNTHESIS; surface the
scope decision to the user. Agent ids: meth `a805893b87ff98e55`, prof
`ae27b5063bb6ac34d`, red `a2dca889287578b51`.
