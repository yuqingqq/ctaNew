# clob_v5 application-heartbeat repair — 2026-08-31

**State:** implemented and committed, deliberately inert in the systemd
no-argument path. **Not deployed.**  Candidate commit:
`7aa952058385f06672e5c1008414a7a837dc053c`; exact `collect_pm.py` SHA-256:
`1c5291aa6d66ceef0c4a724ea7a1e9fa5128d65d1b69034df5638c0136e98ad5`.

## Why this repair exists

The live `clob_v4` collector uses `websockets.connect(ping_interval=3,
ping_timeout=3)`.  Those settings send RFC WebSocket control frames and close
the socket locally when the matching control Pong misses a three-second
deadline.

That is not the heartbeat contract published for the Polymarket CLOB market
channel.  The current official channel document says the client sends the
**text message** `PING` every 10 seconds and the server returns the **text
message** `PONG`:

- <https://docs.polymarket.com/api-reference/wss/market>
- markdown bytes observed 2026-08-31, SHA-256
  `92a02634755fd92cc1c4a3f798ea64f050f76670e677003a9a595d8a8f4c616a`

The old keepalive therefore made a mechanism substitution: an RFC control Pong
was treated as the authority for a contract defined in application messages.
The shortened 3-second deadline amplified that mistake.

## Reproduced evidence

Instrument:

```bash
python3 live/pm_research/pm_heartbeat_diagnostic.py \
  --collector-version clob_v4 --coin btc \
  --since-recv-ns 1788067802114726542
```

Population as of `recv_ns=1788145832066623985`
(`2026-08-31T03:10:32.066624Z`):

| measurement | value |
|---|---:|
| clob_v4 BTC disconnect actions | 953 |
| `PING_TIMEOUT` | 936 (98.22%) |
| `NO_CLOSE_FRAME` | 14 |
| `SLOW_CONSUMER_1013` | 3 |
| local 1011 keepalive close identity | 936/936 (100%) |
| last market message <3 s before local timeout | 224/936 (23.93%) |
| last-message age, median / p90 | 4.616 s / 9.113 s |
| loop lag, median / p90 / max | 2.1 / 6.7 / 74.3 ms |
| WebSocket queue ever paused | 0/936 |
| WebSocket queue depth, median / p90 / max | 75 / 115 / 233 |

Every one of the 936 timeout errors is the same local-close form: `sent 1011
(internal error) keepalive ping timeout; no close frame received`.  Market data
arriving inside the control-Pong deadline and the overload controls refute a
dead event loop, blocked writer, or paused input queue as the dominant cause.

The version boundary supplies a time-local control.  On 2026-08-30, v3_1
recorded 6 BTC `PING_TIMEOUT` gap closures during the 5.5 hours before the
05:30Z boundary.  v4 recorded 732 during the following 18.5 hours.  This is not
a throughput A/B and is not assigned a confidence interval; it only agrees
with the mechanism evidence above.

An initial direct protocol probe sent documented text heartbeats on the same
BTC stream: 3/3 exact `PONG`, RTT 86.6–88.4 ms, while 1,958 non-PONG messages
arrived.  The committed scratch probe then exercised the **actual v5 candidate
path**, not a reimplementation:

| probe | BTC market messages | text PING/PONG | disconnects |
|---|---:|---:|---:|
| 36.025 s | 23,568 | 3/3 | 0 |
| 125.237 s | 18,045 | 12/12 | 0 |

These are transport seam checks with `n=15` heartbeat exchanges, not a
complete-day quality verdict and not evidence about strategy performance.

## Implemented behavior

`collect_pm.py` now has two explicit modes:

- `control-v4` — restart-safe default; unchanged RFC 3/3 behavior and audit
  version `clob_v4`.
- `app-v5` — candidate selected only with `--heartbeat-mode app-v5`; disables
  library RFC keepalive, sends exact text `PING` after subscription confirms,
  accepts only exact text `PONG`, filters it from the JSON market tape, and
  audits as `clob_v5`.

The v5 response deadline is 10 seconds.  Polymarket states the 10-second send
cadence but no response SLA; 10 seconds is therefore a conservative operational
choice, not a measured venue guarantee.  A missed response is classified as
`APP_HEARTBEAT_TIMEOUT`.  Counters `app_heartbeat_pings` and
`app_heartbeat_pongs` reconcile the live behavior.

Only the receive coroutine calls `ws.recv()`.  The heartbeat task sends and
waits on an event set by that sole receiver, avoiding the concurrent-receive
failure class.  It does not begin before the first real subscription message,
so a heartbeat cannot falsely satisfy subscription confirmation.

## Validation

| command | result |
|---|---:|
| `pm_heartbeat_diagnostic.py --selftest` | 4/4 |
| `collect_pm.py --selftest` | 17/17 |
| `collect_pm_v5_heartbeat_tests.py` | 12/12 |
| `collect_pm_v5_shadow_probe.py --selftest` | 2/2 |
| legacy `collect_pm_v4_behavior_tests.py` at candidate commit | 10/10 |
| real producer → DAY_BAR_V2 seam | 7/7 |
| `py_compile` + `git diff --check` | pass |

The v5 behavioral suite includes a healthy positive, a missing-PONG refusal
while market data continues, a near-match `PONG ` refusal, exact audit-era
identity, PONG tape filtering, and the single-receiver invariant.

## Deployment and era discipline

The running process remains PID 3687786, `clob_v4`, started
2026-08-30T05:30:01Z with zero restarts.  Its command line has no heartbeat
argument.  Committing the candidate cannot deploy it accidentally: an
unexpected service restart still selects `control-v4`.

Do not activate v5 mid-day merely because the open v4 day looks bad.  The
2026-08-31 adverse v4 evidence was visible before this repair, so its closing
verdict must remain recorded rather than being erased by a result-conditioned
era change.

Before a v5 deployment:

1. choose and record a boundary before execution;
2. fix and prove the generic era-admission guard (the 08-30 artifact currently
   demonstrates that a mixed-era ETH day can read
   `race_accrual_eligible=true` even though the whole day fails);
3. bind a preflight to candidate commit `7aa9520`, exact collector bytes, the
   installed `--heartbeat-mode app-v5` command, old PID, and chosen boundary;
4. restart first, verify the new process's own exact `collector_start` row is
   `clob_v5`, then append the era stamp last;
5. re-run a postflight that checks PID, command line, exact code bytes,
   PING/PONG counter reconciliation, advancing market rows, and no foreign-era
   admission;
6. judge efficacy only on later complete, independently verified UTC days.

No model, strategy arm, threshold, day-bar, or previously consumed population
changes in this repair.
