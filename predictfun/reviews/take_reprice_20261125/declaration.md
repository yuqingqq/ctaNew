# TAKE-REPRICE v1 — takes only on large move-implied repricings, decided on the Binance stream, under venue rules (declared before any run)

USER (2026-11-25): "Yes proceed to find their selection rules".

## Why (take_stream_20261125: selection.json, selection_dfv.json)

The takers who win the stale quotes take when the Binance move implies a large repricing of the quote: the binary fair value change over
the last 250 ms in the take's direction has a median 11.2 ¢ for their fills (59 % of their shares above 8 ¢, worth +4.98 ¢/sh after fee
at 1 s) against 1.75 ¢ for ours. Our fixed BTC threshold mostly fires on moves that reprice a cent or two.

## The trigger (per market; decision = Binance breakpoint + 1.5 ms feed + 2 ms decision)

At every breakpoint inside the market's window (Binance futures mid change points and change points + 250 ms):
- x = 1e4 × ln(F_t / F_ws) and x250 = 1e4 × ln(F_{t−250} / F_ws), F = the futures mid (event time), ws = the window start;
- sigma = realised vol of the futures mid over the previous 300 s (bps / sqrt s, from change points; a shorter history at a segment start
  is normalised by its own length); tau = seconds to the window end;
- dFV = 100 × [Phi(x / (sigma sqrt tau)) − Phi(x250 / (sigma sqrt tau))];
- ON up when dFV ≥ D, ON down when dFV ≤ −D; a trigger is an OFF→ON transition with no ON on that side in the 2 s before. Breakpoints
  with sigma not positive or tau < 0.5 s are skipped.

The take is the stream test's (MARKET order at the known best, zero slippage: one level at the limit or better, the rest cancelled),
reaching the book at decision + network + 150 ms, on the state known at the grid tick before the decision (stream_replay.StreamBranch).

## Configs (exit: the current maker exit)

`REPRICE-4-172`, `REPRICE-8-172`, `REPRICE-16-172` (network 22 ms) and `REPRICE-8-192` (42 ms). Comparator: `STREAM-172` of
take_stream_20261125 (its saved ledgers; same branch and harness). Multiplicity: three thresholds.

## Days and population

The five v3 days, 408 complete markets, consumed. The thresholds were informed by the winners' distribution on these same days, so a
positive result here is optimistic by construction; a candidate goes to untouched days before any live step.

## Readings (predicates)

Per config: total at settlement (pooled, by day, days positive), unresolved bounds, takes and fills, limit-moved share, take value after
fee at 1 s and 10 s. `beats_stream_D` (REPRICE-D-172 − STREAM-172 > 0 on ≥ 4 of 5 days); `profitable_D` (lower bound > 0 and ≥ 4 of 5
days positive); `holds_at_192` (`profitable_8` rule for REPRICE-8-192). The best D is reported with its multiplicity of 3.

## Checks

The dFV trigger on a synthetic path (a jump near the strike fires, the same jump far from the strike does not, quiet suppresses a
repeat); every take arrives at decision + network + 150 ms; the venue checks of the comparison (accounting re-derived, 1 USDT minimum,
bounds). Stage cap 2,800 CPU s (≤ 900 per day process), projection after the first day.

## Mechanical fix M1 (during the run; no rule, config or reading changes)

The 20261101 day process stopped (rc 1) on a market whose window holds no Binance breakpoint: `np.vectorize(math.erf)` raises on an
empty input unless `otypes` is set. Fix: `otypes=[float]`; identical output for every non-empty input. Days 20261023, 20261031,
20261102 and 20261103 ran the original file (sha256 4bfac424509103c89ddf20f505f0a0b6118f6c0e287dd1852a43f54034bdee60) to completion; 20261101 reruns with the fix, and the ledgers it wrote
before the stop are rewritten only if byte-identical (write_once_gz).

## Stage L — latency headroom (USER 2026-11-25: "Go ahead the test"; declared before its run)

The frozen rule and code (reprice_replay.py at 2a5aa46fc), run by reprice_latency.py on the same 408 markets with a shorter network path:
`REPRICE-8-152`, `REPRICE-16-152` (network 2 ms) and `REPRICE-8-162`, `REPRICE-16-162` (12 ms). Readings: totals at settlement, fills,
limit-moved share and take value per config, against the 172 ms results; `pays_at_152` / `pays_at_162` = lower bound > 0 and ≥ 4 of 5
days positive for either threshold (two thresholds × two latencies: multiplicity 4). Cap 2,000 CPU s, projection after the first day.
This prices a hypothetical faster path to Predict; it is not a measured latency.
