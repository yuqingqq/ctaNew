# Codex review — clob_v5 application-heartbeat repair — 2026-08-31

**Reviewed candidate:** `7aa952058385f06672e5c1008414a7a837dc053c`  
**Collector SHA-256:** `1c5291aa6d66ceef0c4a724ea7a1e9fa5128d65d1b69034df5638c0136e98ad5`  
**Governing reliability rules:** `CLAUDE.md` rules 3–5, 8, 10–13, 15–16  
**Repair record:** `live/pm_research/plans/V5_APPLICATION_HEARTBEAT_REPAIR_2026-08-31.md`

## Decision

**CODE/TEST HOLD RELEASED: the clob_v5 candidate is fit for deployment
preparation.**

**LIVE DEPLOYMENT HOLD MAINTAINED:** do not restart the production collector
into v5 until a separately reviewed boundary/preflight closes the admission
and era-stamp items below.  The current live process remains reviewed v4; this
filing does not deploy or silently change its behavior.

## Finding

The primary v4 disconnect mechanism is validated.  It is not a process crash
and the dominant population is not a remote close.  v4 treats RFC WebSocket
control Pong as liveness authority with a three-second deadline, while the
official Polymarket market-channel contract specifies application text
`PING`/`PONG` at a ten-second cadence.

At the named artifact and exact v4 boundary, the committed diagnostic reports
953 BTC disconnect actions as of `2026-08-31T03:10:32.066624Z`: 936
`PING_TIMEOUT` (98.22%), 14 `NO_CLOSE_FRAME`, and 3
`SLOW_CONSUMER_1013`.  All 936 carry the local 1011 keepalive-close identity.
In 224/936 timeout actions, market data arrived less
than three seconds before the client closed.  Loop lag is 2.1 ms median / 6.7
ms p90, and the WebSocket assembler paused in 0/936 actions.  Those controls
refuse the proposed alternatives of event-loop starvation, writer blockage, or
a paused local receive queue as the dominant mechanism.

The exact server-side reason an RFC control Pong is sometimes delayed or absent
is not observable in this tape.  It does not need to be guessed: RFC control
Pong is the wrong contract boundary, and the documented application heartbeat
works on the same live BTC channel.

## Independent implementation review

The repair is correctly inert before deployment:

- no-argument startup remains `control-v4` and stamps `clob_v4`;
- `--heartbeat-mode app-v5` is required to activate the candidate;
- app-v5 disables both library control-ping settings explicitly;
- one sender emits exact text `PING`; the sole receiver consumes exact text
  `PONG` and excludes it from the market tape;
- heartbeat starts only after a real subscription message;
- missing exact PONG raises identity-classified `APP_HEARTBEAT_TIMEOUT` and
  re-enters the existing reconciled gap/reconnect lifecycle;
- audit rows and startup identity derive from the selected instance mode, so
  candidate rows stamp `clob_v5`;
- PING/PONG counters make the mechanism observable after deployment.

I found no regression in the v4 default path.  The legacy behavioral harness
loads the committed candidate and remains 10/10; the real O1 producer→day-bar
seam remains 7/7.

## Executed evidence

| surface | executed result |
|---|---:|
| heartbeat diagnostic controls | 4/4 |
| collector selftest | 17/17 |
| v5 fake-socket behavior | 12/12 |
| shadow-probe identity controls | 2/2 |
| legacy v4 behavior at `7aa9520` | 10/10 |
| real producer→DAY_BAR_V2 seam | 7/7 |
| live v5 BTC scratch probe 1 | 3/3 PONG, 23,568 messages, 0 disconnects / 36.025 s |
| live v5 BTC scratch probe 2 | 12/12 PONG, 18,045 messages, 0 disconnects / 125.237 s |
| compile / whitespace | pass / pass |

The live probes have only 15 heartbeat actions and no day cluster.  They prove
the real transport seam and reject an immediately broken candidate; they do
not prove complete-day quality, establish an interval, or validate any model.

## Deployment blockers

1. **Era admission:** the generated 08-30 verdict currently marks ETH
   `race_accrual_eligible=true` even though 08-30 is explicitly mixed-era and
   inadmissible.  The whole-day result happens to be false because BTC fails,
   but vocabulary-level eligibility is still wrong.  A v5 boundary must not
   reuse that defect.
2. **Identity-bound preflight:** no v5 deploy instrument yet binds the exact
   candidate commit/bytes, installed systemd command with
   `--heartbeat-mode app-v5`, boundary, old/new PID, and the new process's own
   `collector_start` row.
3. **Era stamp/postflight:** deployment must restart first, verify second, and
   append the era stamp last, then reconcile code bytes, command line,
   PING/PONG counters, advancing rows, and generic admission.

Because the adverse 08-31 v4 evidence was already visible, do not create a
mid-day boundary that silently removes that day.  Preserve its close-time
verdict.  A future boundary is an operational decision recorded before
execution; candidate performance is judged only on later complete days.

## Scope preserved

This repair changes no fair-price model, harmful-flow candidate, cancel/skew
policy, queue simulation, threshold, or performance conclusion.  Iteration 011
fit/score remains on its prior hold, and the integrated baseline blockers are
unchanged.
