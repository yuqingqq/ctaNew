# TAKE-EXIT review — plan (2026-11-25)

USER: "review the result" (TAKE-EXIT-REPLAY v1 / v2). Two parts:

1. **Independent code review** by a fresh agent given the code and outputs but no conclusions.
2. **Quantitative checks** in this packet:
   - **D1 / D2 (`diag.py`, read-only, already run):** classify every take fill by the next venue frame (SURVIVED / TRADED_AWAY /
     CANCEL_UNKNOWN) and every exit fill by its tape trade (THROUGH / AT_PRICE). Result: 1,799 of 3,575 take fills (62,160 sh)
     hit a quote that vanished without trades before the next frame; median frame age at effect 27 ms, next frame +24 ms.
     The replay's IOC uses only the last frame before its effect, so those fills may have hit quotes that were already cancelled.
   - **S1 (`strict_rerun.py`, declared here after D1, before running):** the v2 PASSIVE_EXIT arm on the same 281 markets with the strict
     IOC rule. Fill at p only what is still displayed at p in the first frame after the effect, plus public trades at p in between.
     That is the lower bound on take fills; v2 is the upper bound. Everything else unchanged.
     - Parity gate: strict=False reproduces v2 on market 3697902.
     - Readings: net, days positive, markets positive, net without young exit value; v2 vs S1.
     - Budget ≤ 600 CPU s.

## S2 (declared after the independent review, before running)

The reviewer found the 40 ms IOC effect delay optimistic against the measured order-live latency: p50 58 ms (`pf_clock_rl.py:9-11`;
FINDINGS lists "40 vs 58 ms" as a sign-flipping premise). S2 = S1's strict availability rule with TAKE_DELAY 58 ms (`strict58.py`; the
same harness with the module delay set before any branch is built), PASSIVE_EXIT on the same 281 markets. This is the most
conservative full-path bound for this 50 ms-grid client. Budget ≤ 450 CPU s. Report it beside v2 (lenient, 40 ms) and S1 (strict,
40 ms).
