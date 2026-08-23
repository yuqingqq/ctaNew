# WW_EBX_PROTOCOL — E-BINANCE, the fourth envelope channel (strict ww_v1 extension)

**Status: DRAFT FOR COORDINATOR FREEZE — nothing here is frozen.** Authored by
the DE session 2026-08-23 under Ruling R-47. **Drafted blind**: no warning
window, no R, and no per-channel share has been computed with the new
channel. The feed-cadence and feed-lag figures in §3 are INSTRUMENT facts
measured before the freeze — the same standing as SP §4's `min_size`
measurement — and are inputs to the floor declaration, not outcomes.

**What this is:** a STRICT EXTENSION of `ww_v1`, not a new protocol. Same
population (day-series, 30 windows/coin/day, era days), same arm
(BACK_DISPLAYED / `front=False`), same estimand (rescuable share R of
negative-drift fills), same FROZEN bar carried by value under R-20:
`f*_low = {btc: 0.309, eth: 0.494}` (CANCEL_POLICY_PROTOCOL §1, R-1). The
bar does NOT move — this is a new arm against an existing bar, not a
Class-D amendment (R-47's own classification). `R_4ch` is reported BESIDE
the three-channel `R_3ch`, never pooled.

**Mechanism (R-47):** a PM taker pays ~225 bps to cross, so every PM taker
acts on something worth more than 225 bps — overwhelmingly a move just
watched on Binance. The informed flow generating our drift is downstream of
a venue we already stream. The HEDGE version of this idea is CLOSED ON
MECHANISM without measurement (R-47, user-caught): the lead that makes
hedging necessary makes it impossible — the perp leg transacts post-move
and locks the loss in. The INVERSION is the lever: Binance flow as a
predictor of incoming PM flow — a cancel BEFORE the fill, not a hedge after
it. It needs no price advantage over the book (sidesteps E-X1 entirely);
it needs only to be EARLIER ON FLOW.

---

## §0. Forbidden forms

No threshold sweeps on the trigger; the trigger is ONE pre-registered
parameter-free rule (§2). No pooling of E-BX into the three-channel R. No
retro-fitting a different trigger after seeing R — a second trigger is a
new blind-drafted protocol. No promotion of anything from the per-channel
descriptive table. If R clears the bar, the family is REOPENED on this
channel and the cancel POLICY still needs its own protocol — a bound
crossing a bar adopts nothing (the R-45 amendment-1 asymmetry, applied in
the direction this instrument actually points).

## §1. Population, arm, estimand, bar — all inherited, none re-cut

Day-series selection (`select_by_day`, 30/coin/day, era days, verdict coins
btc/eth), BACK_DISPLAYED arm, the ww_v1 episode/warning machinery
(`warning_of`), negative-drift fills at the frozen drift horizon, rescuable
share `R(τ)` at the Class-D `tau_decision_rung = 250 ms`. Verdict per
(coin, day) cell: DEAD / GO / INDETERMINATE against `f*_low` by value,
exactly ww_v1's three-way. Windows spanning a `crypto_prices` collector gap
are excluded and ledgered (the R-ADMISS analog for the second feed).

## §2. The E-BX trigger — one rule, parameter-free, pre-registered

**E-BX fires at the first NEW ADVERSE EXTREME of the deployed underlying
feed since the episode started**, evaluated at feed RECEIPT time: for a
resting maker BUY_UP, the underlying printing strictly below its running
minimum since the quote began resting; for SELL_UP, strictly above its
running maximum. The 4-channel envelope is
`min(first of {E-FLOW, E-DEPLETE, E-MID}, first E-BX)`.

Named and REJECTED alternates (not run, and why):
- *any adverse price change*: fires ~every other sample on a random walk —
  saturates the envelope by construction, R would be vacuously large;
- *strike crossing*: episode-independent and saturating near the strike,
  where the 1-tick book lives — same vacuity, different route.

Beside the primary: the E-BX-ONLY warning share (descriptive — how much of
the 4-channel R the new channel contributes) and the per-channel first-fire
table, ww_v1's own reporting shape.

## §3. Latency arithmetic — declared BEFORE the run (the crux, R-47)

The warning window is Binance-trigger to PM-fill: `W = t_fill −
t_recv(trigger)`, anchored at RECEIPT (as-knowable, the same discipline as
ww_v1's 250 ms PM knowledge lag). Rescuable requires `W > τ_cancel`
(250 ms rung). The event-anchored equivalent is `W_event > feed_lag +
τ_cancel` — the two forms are identities, the receipt-anchored one is used
because receipts are what a deployed canceller has.

**The deployed instrument, measured (2026-08-23, era-day sample):** the
ONLY underlying feed in this pipeline is Polymarket's `crypto_prices`
relay — **1 Hz per symbol**, receive lag **p50 0.46 s / p90 0.58 s**
(recv−event, epoch-ns receive clocks on the same collector host as the PM
archives; alignment spot-checked). There is NO raw Binance perp stream in
the repo.

**The declared floor:** with 1 Hz sampling plus ~0.5 s relay lag, a lead
must exceed roughly **1.5–1.7 s** (sampling quantization + relay lag +
250 ms cancel) before this instrument can register it as rescuable. The
three book channels died at a median warning of 0.16 s against the same
250 ms; the whole question is whether Binance buys MORE than that, and
this instrument can only see the answer when the lead is >~1.5 s.

**Verdict scope, stated now so the negative is read honestly:** a negative
closes E-BX **on the deployed feed** — deployment-honest, because the
cancel would run on the feed we have. It does NOT bound a
direct-exchange-websocket instrument (~50–150 ms, tick-level): that
requires new data collection, is named here as the follow-on boundary, and
would be its own blind protocol IF the coordinator ever orders the
collection. A positive at 1 Hz, conversely, is a LOWER bound on the
channel — leads long enough to survive this floor are leads a faster feed
also sees.

## §4. Verdict semantics

Per (coin, day): `R_4ch(250ms)` vs `f*_low` → GO / DEAD / INDETERMINATE
(ww_v1 cell rule, CI treatment unchanged). Roll-up over the 8 coin-day
cells, R-9's shape: **REOPENED-on-E-BX** requires ≥75 % of coin-days GO
with ZERO DEAD; **DEAD-4ch** requires ≥75 % DEAD with zero GO (the R-11
closure then stands across FOUR channels — a materially stronger negative
than today's three); anything else is INDETERMINATE and the R-11 closure
stands unchanged. `R_3ch` is re-reported beside `R_4ch` from the same run
as the conformance anchor: it must reproduce ww_v1's frozen receipts
within recomputation tolerance, or the run aborts.

## §5. Controls (before any cell is read)

1. **Conformance**: 3-channel R from the extended engine equals the ww_v1
   receipt values on the same windows (the extension touched nothing).
2. **Trigger sensitivity (must-fail)**: shifting the price series +1 s
   must change W on known-affected fills; a doctored price series with the
   extreme removed must un-fire E-BX. A trigger that cannot fail is not a
   measurement.
3. **Clock-misalignment control**: ±1 s shifts of the price stream must
   materially move the E-BX-only share — proving alignment is load-bearing
   and therefore that the unshifted alignment was checked.
4. Determinism on repeat replay, as everywhere.

## §6. Sequencing

1. This draft goes to the coordinator; **freeze before any receipt is
   read** (Q-DE-11). Build under the R-1 sealed pattern is available while
   the freeze is pending.
2. On freeze: run, read against the inherited bar, report cells first,
   roll-ups second, E-BX-only shares beside, per R-9/R-17.
3. `policy_bounds_v1` continues in parallel (R-47's CONTINUE).
