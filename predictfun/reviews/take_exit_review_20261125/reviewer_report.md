# Independent review (fresh agent, given code and outputs, no conclusions) — 2026-11-25, verbatim summary

Overall: no invalidating defect found. Accounting reconciles; the result is positive under every stress computed from saved outputs.
It is essentially the taker leg: takes +$3,731 at END net of fees; exits roughly neutral to the 298 s mark (+$203).

1. **Material: the IOC wins every race inside a ~50 ms snapshot gap.**
   - pilot.py:53-71 fills against the last venue frame at or before the effect and never checks the next one. Predict publishes
     ~50 ms snapshots (gap p50 50 ms; FINDINGS P3).
   - 1,029 of 3,575 PASSIVE_EXIT fills (30,019 of 121,486 shares) took depth gone from the next snapshot, a median 25 ms later.
     87 % of those had no public trade at or through the price, so a maker cancelled. They are the best fills (+4.9 vs +3.8 ¢/sh at 3 s).
   - Take value at END: −22 % if the cancel time is uniform within the gap; −45 % under TAKE-SIGNAL's own two-frame rule, which the
     replay dropped without flagging.
2. **Material: 40 ms IOC delay vs the measured order-live latency of 58 ms.**
   - pf_clock_rl.py: the measured p50 is 58 ms, with a lower bound of 46 ms.
   - Alone, −15 % take value at END; with the uniform gap −36 %; with two-frame −66 %.
   - Caveat: the 50 ms grid already adds ~25 ms of mean decision delay, so an event-driven client would be less biased.
3. **Material for generalising: "complete" uses a future property** (the book stays fresh and two-sided to 298 s) and selects
   high-edge markets. ¢/sh at 3 s on TAKE-SIGNAL triggers: complete +3.81, missing_endpoint +2.22, stale_future_book +1.75.
   "Mostly contested at the close" is wrong: 65 % of complete markets end with the mark at ≤ 5 or ≥ 95, and 21 % between 10 and 90.
4. **Minor: the FINDINGS checklist (C1–C3, C12, C15) was not run.**
   - M 250 on the IOC public-trade debit: −3.6 % take value.
   - Cancel-pending exits: 2.4 % of exit shares, +$107.
   - WS-timed exits +$242 vs E2-estimated −$37 at END.
   - M 250 on the exits needs a replay.
5. **Minor: exit timing is ASYNC_FAST10 (10 / 10 ms), not Tokyo (58 / 37).**
   - Not inflating: early exits are −$385 vs END, and a 37 ms cancel would add +$125.
   - PASSIVE_EXIT − HOLD (+$6.90) is not a robust reading.
6. **Minor: the checks.** G3 checks limit, size, fee and ledger, not the venue frame. The reviewer's raw re-derivation matched all
   6,617 IOC prices and frames and all 10,806 passive fills. The v2 checks lack a falsifier (rule 15).
7. **Minor: the Binance signal uses the median feed delay** (event + 3.5 ms; p99 10.7 ms). 14 % of sent triggers depend on an update
   in the last 4–11 ms.

**Sound:**
- client-known limit and own-cross;
- depth not reused, partials, fee 200 bps × min(p, 1 − p);
- totals reconcile ($3,730.6 + $202.9 = $3,933.5);
- settlement valuation $3,911.84 (HOLD $4,033.54).

**Could not verify:**
- the semantics of Predict's book stamps and its clock offset against Binance;
- whether Predict has a true IOC order type;
- competitor speed (about 40 % of trades are E2-timed);
- live realism of maker exits (the replay over-credits passive fills 1.5–2×, C4);
- path effects of the static haircuts.

Scripts: session scratchpad review/ (a2.py, a4.py, a9.py, a10.py); three run_light jobs on the claude2 slot, ~80 CPU s.
