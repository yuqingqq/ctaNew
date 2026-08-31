# Codex review — changed clob_v5 candidate and live-deploy package — 2026-08-31

**Review request / exact repository tip:**
`9a12f56967b0b1cc04083c492889d2300b062acb`

**Substantive package tip named by the request:**
`9efb932` (`9a12f56` adds the request file only)

**Changed collector SHA-256:**
`39889848ad0f056852192fc00f72e1c64b230d4298588a21d144ee67311eb0a8`

**Review method:** all code and claimed gates were executed in a detached
worktree at exact `9a12f56`; the production service was inspected read-only.
No unit, drop-in, collector, era ledger, or production tape was changed.

## Ordered verdicts

### (A) Changed collector candidate — HOLD MAINTAINED

The executable candidate delta from `7aa9520` is indeed only the two requested
constant changes, 10 s -> 3 s for both application-heartbeat interval and
timeout. The implementation derives a nominal 6 s worst-case heartbeat bound,
keeps the no-argument v4 default, uses the one WebSocket reader, and cancels the
heartbeat task on reconnect. The mechanism remains plausible.

The changed **code/test package is not releasable yet**, however. Two of the
newly advertised tests do not exercise what they claim, and the committed
cadence rationale turns worst-case bounds into realized loss/floors. Those are
result-bearing correctness defects, not cosmetic test names.

Do not rebind the deploy package to this candidate yet.

### (B) v5 deploy package — HOLD MAINTAINED

Do not arm or restart from this package. The submitted 193/147/117/35 gates all
reproduce, and the no-mode stdout and lapsed-boundary paths fail safely, but
fresh execution found three package blockers:

1. the preflight and DA era consumers still disagree on a multi-hop return;
2. the health verifier, runbook, and permanent transition receipt still encode
   the superseded 10 s cadence;
3. the gap-tail proof is still not bound to the production unit, despite the
   known shared-ledger/foreign-collector failure class.

The deliberately stale `CAND_SHA` / `CAND_COMMIT` and the absence of a new
USER-ruled instant are acknowledged sequencing states and are **not** filed as
findings. They still must be re-pointed after candidate release, as requested.

## Candidate findings

### V5-C5-1 — both new deadline “behavioral” probes are fixture-overridden

`collect_pm_v5_heartbeat_tests.py::run_market()` unconditionally sets:

```python
C.APP_HEARTBEAT_INTERVAL_S = 0.02
C.APP_HEARTBEAT_TIMEOUT_S = 0.03
```

The new tests later set the timeout to `0.05` and `5.0` immediately before
calling `run_market()`. Both requested values are therefore replaced by
`0.03` before the real heartbeat path runs. The “constant is restored” check
also saves/restores the already-mutated fixture value `0.03`, not the shipped
value `3.0`.

Independent reproduction:

```text
set timeout before run_market: 5.0
timeout after fixture setup/run: 0.03
shipped timeout: 3.0
```

Thus nominal `21/21` does **not** establish that the configured 0.05 s deadline
fires, that a PONG is accepted under the configured 5 s deadline, or that the
shipped constant is restored. Make the fixture accept explicit interval and
timeout parameters (or stop overwriting caller values), then assert the value
seen by the running heartbeat coroutine and restore both globals to their
shipped values.

### V5-C5-2 — the added PONG<=PING test is vacuous

`run_market()` returns `counted_app_pings` and `counted_app_pongs` as top-level
keys. The added check instead does:

```python
_c = counters_run.get("counts", {})
...
if _c else True
```

There is no `counts` key, so `_c` is always empty and the test always passes.
An injected fake socket that sends two exact PONG frames for each PING produced:

```text
ping_sent=11, pong_received=22, claimed_test_input=None
```

The collector counts those frames faithfully and the deploy checker currently
refuses `PONG > PING`, which is a safe direction. But the candidate suite's
specific claim that this previously uncovered case is tested is false. Wire the
real returned keys into a known-bad duplicate/unsolicited-PONG fixture and state
the intended division of responsibility: either the producer rejects/ignores
the extra frame, or the producer counts it and the deploy gate must refuse it.

### V5-C5-3 — the cadence/loss premise reverses a worst-case inequality

The 6 s number is a **worst-case heartbeat blindness bound** (`interval +
timeout`), not a per-disconnect minimum or observed mean. The receive task can
also fail earlier than the heartbeat task. Consequently:

- `68.8 disconnects/hr * 6 s = 412.8 s/hr` is an upper-bound sensitivity term
  under the stated model, not a “floor”;
- `68.8 * (20 - 6) ~= 963 s/hr` prices the difference between two worst-case
  bounds, not loss that every post-deploy day “would have” incurred;
- because `gap_start` is the last market message rather than the unobserved
  socket-death instant, the recorded gap can include quiet-market time as well
  as detection time.

The useful conclusion survives in narrower form: matching v4's 6 s worst-case
bound removes an avoidable measurement-basis regression. The committed
`DAY_BAR_V2_PREREGISTRATION.md` interpretation must describe this as a
sensitivity/upper bound, not a realized floor, and must not conclude from that
bound alone that P1 can pass only if the disconnect rate collapses.

### V5-C5-4 — the documentation claim for 3 s is stronger than the source

The [official Polymarket Market Channel documentation](https://docs.polymarket.com/api-reference/wss/market)
says **“Client heartbeat — send every 10 seconds.”** It does not state that ten
seconds is a minimum or that any faster cadence is guaranteed.

The 3 s choice has short empirical support, not documented authorization. My
independent 25 s scratch probe reproduced `8 PING / 8 PONG`, zero disconnects,
but again selected the stale/expired BTC slug and received only one market
message. That confirms transport tolerance, not concurrent-flow behavior or
long-run server policy.

The USER ruling may remain 3 s. Correct the source/comments to identify it as
an empirically tested deviation with the stated residual, rather than a venue-
documented “minimum.” Also correct the contradictory candidate comment saying
3 s still bounds blindness at `~13 s` (the same block correctly says 6 s), and
either require `<` for “well under” or rename the test that currently accepts
timeout == interval via `<=`.

### Candidate no-regression instrumentation note

The legacy v4 behavior test produced `9/10` once, then `10/10` on three
standalone reruns. The failing predicate is not jitter-safe: legal full-jitter
draws give attempt-1 delay `0.9995`, attempt-2 delay `1.0`, for which the suite's
`d2 > d1 * 1.15` assertion is false even though the exponential envelopes are
correct. Pin the RNG or test the declared envelopes. This file was not changed
in the candidate commit, so it is not evidence of a collector regression, but
the claimed `10/10` gate is nondeterministic as written.

## Deploy-package findings

### V5-P5-1 — shared chain semantics still diverge after a multi-hop return

The preflight consumer rejects a plain transition to **any** version in its
`_seen_versions` set. DA rejects only a transition to `prev_era` unless the
current era came from a rollback. The 35-case equivalence battery contains the
one-hop `v4 -> v5 -> v4` shape but not a multi-hop return.

Executed canonical ledger:

```text
legacy v3_1 -> v4
transitioned v4 -> v5 at 07:00
transitioned v5 -> v6 at 08:00
transitioned v6 -> v4 at 09:00, with no rollback evidence
```

Result:

```text
current_era_and_open_v5 => REFUSE: transitioned row 4 returns to clob_v4
DA era_timeline         => ACCEPT: ... v4, v5, v6, v4
```

This directly contradicts the package claim that the walks now share version-
general chain semantics. Decide one rule for returns to any previously used
version, implement it in both consumers (preferably from one shared validator),
and add this exact four-era fixture to the same-input/two-consumer seam. Re-run
the differential generator as an executable, retained artifact; the reported
17,729-ledger fuzz itself is not committed here, only selected repros are.

### V5-P5-2 — the 3 s candidate was not propagated into the deployment gate

`v5_boundary_preflight.py` still sets `APP_HEARTBEAT_CADENCE_S = 10` and derives
its minimum PING rate from that value. A two-sample 60 s interval with only
three PINGs, three PONGs, and advancing market rows is accepted now:

```text
ACCEPTED ping_delta=3 over 60s
lower bound under the shipped 3s cadence would be 10
```

That means the post-deploy gate can certify a sender running roughly 3.3x
slower than the reviewed candidate's cadence. Derive the verifier value from
the reviewed candidate or add an explicit cross-artifact equality assertion;
do not maintain a second silent constant.

The same stale value appears in two artifact surfaces:

- the transition receipt permanently says `10s cadence`;
- the sole deploy runbook says the deployed application heartbeat is `10 s`.

The 193-check suite and `check_runbook_consistency()` both pass this
contradiction. Add a known-bad in which candidate, checker, receipt, and runbook
cadences differ, and make pre-arm refuse it before any restart. The post-review
candidate rebind must update the release/authority text as well as the two
acknowledged hash fields.

### V5-P5-3 — the counter gate's gap-tail observation is not process-bound

The package correctly PID-binds `observe_collector_start()`, but
`observe_gap_tail_version()` still reads the newest post-boundary row written by
**any** collector process. Most gap rows do not carry a PID, so the value cannot
be associated with the active systemd unit.

Executed against a temporary shared ledger:

```text
live unit collector_start: pid=4242, clob_v5
later foreign disconnect:  clob_v4, no pid
observe_collector_start(..., 4242) => live clob_v5 row
observe_gap_tail_version(...)      => clob_v4
```

The final counter check then refuses/rolls back a healthy unit because of a
foreign writer. This is the same class R-351 made real and the PID-aware start
observer was added to close. Either put a process identity on every audit row
and filter it, or remove this unbindable tail as an authority and rely on
process-bound evidence.

## Additional correctness repairs

- `check_counters()` filters below-floor samples internally, but the CLI
  success report uses the original unfiltered `hb[0]` / `hb[-1]`. An executed
  three-line fixture passed on its two post-start rows while the CLI population
  would report `ping_delta=-989`. Return the evaluated population/metrics from
  the checker and print exactly those.
- `--log-offset` help and the runbook still describe a value printed by
  `--armed`, while the production CLI overwrites the argument from the stamped
  `log_offset_at_stamp`. The current behavior is safer; make the operator
  interface and text describe the behavior actually used.

## Submitted gates re-executed at exact `9a12f56`

| Surface | Independent result |
|---|---:|
| candidate SHA-256 | exact match `39889848...eb0a8` |
| collector executable delta vs `7aa9520` | only interval/timeout `10 -> 3` |
| collector selftest | `17/17` PASS |
| v5 heartbeat suite | nominal `21/21` PASS; C5-1/C5-2 show false coverage |
| O1 producer -> day-bar seam | `7/7` PASS |
| v4 rollback/no-regression suite | one `9/10`, then three `10/10`; flaky predicate reproduced |
| 25 s live scratch transport probe | `8/8` PONG, 0 disconnects, only 1 market message (expired slug) |
| v5 preflight selftest | `193/193` PASS |
| DA forward-day selftest | `147/147` PASS |
| shared chain equivalence | `35/35` on declared fixtures; P5-1 missing fixture disagrees |
| preflight mutation audit controls A/B/C/D | PASS |
| preflight mutation sites | 117: 104 assertion-killed, 13 crash-killed, 0 survivors |
| no-mode CLI with stdout redirected | exit 2, stdout 0 bytes; PASS |
| compile / `git diff --check` | PASS / PASS |
| real lapsed `--pre-arm` | exit 2, refused by name at +10,535 s |

Production remained clob_v4 throughout: PID `3687786`, active/running,
`NRestarts=0`, exact no-flag v4 argv, `WorkingDirectory=/home/yuqing/ctaNew`,
and only `slice.conf` installed. Nothing was armed, restarted, or stamped.

## Minimum next re-review

1. Close C5-1/C5-2 with real parameterized behavioral probes and a real
   unsolicited/duplicate-PONG known-bad; correct C5-3/C5-4 without moving the
   USER-ruled 3 s constants or the frozen day bars.
2. Make the two era consumers agree on the exact `v4 -> v5 -> v6 -> v4`
   fixture and retain an executable differential generator or exhaustive
   bounded family, not only hand-selected survivors.
3. Propagate/derive the 3 s cadence through the health gate, permanent receipt,
   runbook, and a cross-artifact refusal test.
4. Bind or remove the foreign-writable gap-tail proof; make the counter report
   use the population the checker actually evaluated.
5. Re-run 17/21/10/7, 193/147, mutation, and chain-equivalence gates.
6. Only after (A) releases: rebind candidate commit/SHA and release authority;
   then obtain a new USER-ruled future instant, re-point every instant/day-one
   field and falsifier, perform a narrow final boundary review, and only then
   install a drop-in.

Until those closures are committed and independently executed, **both holds
remain maintained**.
