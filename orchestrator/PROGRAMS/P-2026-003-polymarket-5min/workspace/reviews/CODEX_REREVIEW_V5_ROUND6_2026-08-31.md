# Codex re-review — clob_v5 round 6 — 2026-08-31

**Request commit reviewed:**
`30c888998a52b0dd1b85604bcada340fc78cff7b`

**Substantive fix tip:** `a8ad977` (`30c8889` adds the request file)

**Collector SHA-256 at the reviewed tip:**
`b219537abe3cb7ba2a8488c21cef7bb396dd58a58485d36227c0d71b3f838347`

All repository execution was from a detached worktree at exact `30c8889`.
The production service was inspected read-only. No unit, drop-in, process,
ledger, or tape was changed.

## Ordered verdicts

### (A) Changed collector candidate — HOLD MAINTAINED

C5-2, C5-3, and C5-4 are closed. The producer now exposes the real PING/PONG
counters to the duplicate-PONG test, the day-bar text correctly treats
`interval + timeout` as an upper-bound sensitivity rather than realized loss,
and the 3 s cadence is accurately described as a USER-accepted empirical
deviation rather than venue authorization.

C5-1 is **not closed**. The new tests still do not observe the deadline used by
the running heartbeat coroutine and do not behaviorally distinguish the
requested deadlines. A hard-coded wrong deadline survives the entire advertised
22-check candidate suite. The candidate source itself uses the correct constant,
but the requested behavioral evidence remains false, so the code/test candidate
is not releasable under the reliability rules.

Do not rebind the deploy package to this candidate yet.

### (B) v5 deploy package — HOLD MAINTAINED

The P5-1 shared return rule, P5-2 derived cadence, P5-3 removal of the
foreign-writable gap tail from code authority, and the evaluated-population CLI
repair all work in execution. The two era consumers also agreed on an additional
20,000 independently generated combination ledgers.

The sole operator runbook, however, still states the superseded behavior for
both the log-offset and gap-tail seams. That leaves the claimed interface repair
open and makes the runbook disagree with the safer implementation it controls.
The package is also downstream of the candidate hold above. Do not arm or
restart from this tip.

The deliberately stale `CAND_SHA` / `CAND_COMMIT` and the lapsed boundary are
acknowledged sequencing states and are not findings in this review.

## Findings

### V5-C6-1 — the C5-1 deadline tests remain mutation-insensitive

`run_market()` records `heartbeat_timeout_seen` by copying the fixture global
immediately after the fixture assigns it. That is not an observation from the
running coroutine:

```python
C.APP_HEARTBEAT_TIMEOUT_S = 0.03 if timeout_s is None else timeout_s
captured["heartbeat_timeout_seen"] = C.APP_HEARTBEAT_TIMEOUT_S
```

The two behavioral shapes do not separate correct from incorrect deadline use:

- `missing_pong` times out under either 0.03 s or the requested 0.05 s;
- `healthy` returns an immediate PONG, so it succeeds under either 0.03 s or
  the requested 0.30 s.

Independent mutation execution replaced the real coroutine's timeout argument
with the hard-coded wrong value `0.03`, while leaving the fixture and all checks
unchanged. Result:

```text
BEHAVIOURAL configured-deadline check: PASS
BEHAVIOURAL longer-deadline check:     PASS
V5 HEARTBEAT BEHAVIORAL:               22/22 pass
mutant exit:                           0
```

This directly falsifies the claim that the suite proves which deadline the
running coroutine used. In addition, the fixture does not restore the module
globals: an isolated `run_market(... interval_s=0.02, timeout_s=0.30)` changed
them from `3.0/3.0` to `0.02/0.30` after return. The check named “module carries
outside the fixture” only checks the import-time copies, not the live module.

Required closure: use a delayed-PONG behavior whose delay lies between the
wrong and requested deadlines, or measure the real timeout duration with
tolerances, so a hard-coded deadline is killed. Restore interval and timeout in
`finally`, then retain a known-bad mutation/control showing the test goes red.

### V5-P6-1 — the sole runbook still describes three superseded authorities

The request says the `--log-offset` runbook repair is complete, but the runbook
was not changed on this seam. It still instructs the operator to:

- record `LOG_OFFSET` from `--armed` and says it anchors evidence (lines 63–64);
- pass `--log-offset LOG_OFFSET` and calls it a scan hint (lines 92–97);
- expect lines after the “armed-time offset” (line 99).

The implementation does something different and safer: the CLI argument is
explicitly ignored, and `main()` overwrites it from `log_offset_at_stamp` in the
post-restart transition receipt (preflight lines 2538–2542 and 2642–2659).

The same runbook section also says counter verification checks the newest gap
row declares `clob_v5` and bounds unresolved PINGs at two (lines 101–105).
Neither statement is current:

- P5-3 deliberately makes a foreign gap tail non-authoritative; the executed
  `clob_v4`-tail fixture now accepts a healthy PID-bound v5 process.
- `check_counters()` applies an interval PONG/PING ratio floor of 50%; it has no
  absolute unresolved-PING bound of two.

`check_runbook_consistency()` accepts all of these contradictions, so the green
preflight suite does not guard the sole human-executed authority against this
drift.

Required closure: rewrite step 2/4 to say that the postflight stamp's
machine-derived `log_offset_at_stamp` is the sole offset authority (preferably
remove the ignored operator argument from the command), remove the gap-tail and
absolute-deficit claims, and add runbook known-bads for the superseded phrases.

## Verified closures and executions

| Surface | Independent result at `30c8889` |
|---|---:|
| candidate SHA-256 | `b219537a...8347` |
| collector selftest | 17/17 PASS |
| v5 heartbeat suite | nominal 22/22 PASS; V5-C6-1 mutant also passes 22/22 |
| legacy v4 behavior | 10/10 PASS in the submitted run and five further standalone runs |
| O1 producer -> day-bar seam | 7/7 PASS |
| preflight selftest | 201/201 PASS |
| DA forward-day selftest | 150/150 PASS |
| same-input/two-consumer equivalence | 38/38 agree |
| committed differential fuzz | 1,128 ledgers, 0 disagreements |
| additional independent combination fuzz | 20,000 ledgers, 0 disagreements |
| mutation audit controls A/B/C/D | PASS |
| mutation audit | 119 sites: 105 assertion-killed, 14 crash-killed, 0 survivors |
| exact P5-2 slow sender | 3 PINGs / 60 s REFUSED; need >=10 |
| exact P5-3 foreign `clob_v4` tail | healthy counter population ACCEPTED |
| filtered counter population | returned 2 evaluated rows and their exact deltas |
| compile / `git diff --check` | PASS / PASS |
| live lapsed `--pre-arm` | exit 2; named late-boundary refusal |

The exact multi-hop no-evidence chain now refuses in both consumers. Simple and
multi-hop retries after verified rollback accept in both consumers. These P5-1
closures are real, not inferred from the zero-disagreement count alone.

Production remained on clob_v4 throughout: service active/running, PID
`3687786`, `NRestarts=0`, exact no-flag v4 argv,
`WorkingDirectory=/home/yuqing/ctaNew`, and only `slice.conf` installed.

## Minimum next re-review

1. Make the deadline tests kill a hard-coded wrong timeout and restore fixture
   globals.
2. Make the sole runbook match the stamp-owned offset, PID-bound version proof,
   and actual ratio rule; bind those statements into its consistency checker.
3. Re-run 17/22/10/7, 201/150, mutation, equivalence, and differential gates.
4. Only after candidate release: rebind candidate identity and release text,
   obtain a new USER-ruled future instant, re-point every instant/day-one field
   and falsifier, and submit the narrow final boundary review before arming.
