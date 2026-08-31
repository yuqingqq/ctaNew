# Codex final review — v4_1 boundary seam — 2026-08-31

**Exact review tip:** `81d3d2cae0b1cd29d1f5671e5c2b36b11a25a431`

**Code tip named by the refreshed request:** `f5fff9e`; `81d3d2c` changes only
the review request after that code tip.

**Scope:** the `clob_v4 -> clob_v4_1` identity/boundary seam, its normal and
failure emitters, health verification, the UPDATED-2 telemetry, shared status-
line consumers, and the exact runbook sequence. The ruled instant is
`2026-08-31T22:00:00Z`.

**Live-mutation statement:** production was inspected read-only. Nothing was
armed, restarted, stamped, appended to a production ledger, or written to the
market tape. The integrated gates create and remove their own mutation scratch
file. The worktree was clean after they completed.

## Verdict

### Proceed at 22:00Z by the USER's explicit ruling; the seam is not release-clean

The exact clean path is substantially stronger than the prior filing. Era
identity, the ruled instant, candidate collector bytes, target admissibility,
start-row chronology, rollback/abort/recovery chain construction, the
half-landed recovery repair, per-coin progress, and both dated and legacy
status-line formats all pass their positive and known-bad controls. Live
`--pre-arm` is green on the still-running v4 process.

I nevertheless cannot issue an unconditional **HOLD RELEASED**. Two missing
post-arm identity checks let the gate certify a process other than the one the
stamp claims, and the new diagnostic fields are not actually connection-local.
The USER has explicitly ruled that this verdict does not gate the 22:00Z
launch, so these are filed for post-deploy correction rather than as an
instruction to miss the boundary.

If the operator follows the runbook's single-mode commands exactly, the unit is
not modified by another actor after `--armed`, and the candidate does not
auto-restart between stamp and health, I found no defect in the normal
transition that changes collector data or prevents the 10/10 process from
starting. Those conditions are narrower than the gate's present claims.

## Findings

### V41-F1 — HIGH — health can certify a replacement process, not the stamped process

`--verify-health` calls `observe_collector_start(..., unit_pid=current_pid)` and
then treats that current PID as the expected PID for the sampling interval. It
does not compare the current PID/start receipt with the open transition row's
`pid` and `collector_start_recv_ns`, and it never checks `NRestarts`.

Executed known-bad: an open transition stamped for PID `222`, followed by a
live `clob_v4_1` start for PID `333` with `NRestarts=1`, was accepted by
`check_health_identity()` and returned `333`. Thus an automatic restart after
the T+2 stamp but before the T+6 health command can pass as healthy if the
replacement stays up while two status rows are sampled. The check catches a
PID change *during* sampling; it does not catch the already-completed change
between stamp and sampling.

Post-deploy fix: resolve the exact open transition row and require its PID and
start `recv_ns` to equal the live process declaration before and after
sampling. Require `NRestarts=0` on both observations. Add a known-bad for a
replacement before sampling, separate from the existing restart-during-
sampling case.

### V41-F2 — HIGH — stamp and health omit two execution-context checks after arming

`check_pre_arm()` verifies the production `WorkingDirectory` and absence of
`ExecStartPre`. `make_stamp()` and `check_health_identity()` do not. They check
the installed argv, selected environment properties, and the repository copy's
hash, but the script token is relative, so `WorkingDirectory` is part of which
file executes; `ExecStartPre` can alter state before it.

Executed known-bad: `make_stamp()` accepted and emitted a `clob_v4_1`
transition with `WorkingDirectory=/tmp` and
`ExecStartPre=/bin/foreign-prestart`. The same two values were accepted by the
health identity check. A unit edit after T-5 can therefore invalidate the
pre-arm evidence without preventing the post-restart receipt.

Post-deploy fix: use one shared system-safety checker at pre-arm, stamp, both
health observations, and restoration. It must include `WorkingDirectory`,
`ExecStartPre`, `ExecStartPost`, environment, slice, output mode, unit identity,
argv, and bytes. Add stamp-time and health-time known-bads for both omitted
properties.

### V41-F3 — HIGH diagnostic correctness — the new “per-connection” fields are not connection-local

The disconnect telemetry's advertised interpretation does not match its state:

- `conn_msgs` subtracts two values of `self.msg_by_coin[coin]`. Discovery keeps
  overlapping current/next market tasks for the same coin, so a healthy sibling
  socket increments that counter while the failing socket receives nothing. A
  deterministic known-bad with zero target-socket messages and seven sibling
  messages reports `conn_msgs=7`, falsely classifying the target socket as
  having worked.
- `conn_lifetime_s` subtracts `scope_start_ns`, which is set once for the whole
  market task outside the reconnect loop. It is a task lifetime on the second
  and later connection, not a socket lifetime.
- `silence_before_close_s` uses `last_recv_ns`, also retained across reconnects.
  If a new connection never delivers, the value is time since the preceding
  connection's last message rather than silence on this connection.

These fields do not feed tonight's stamp or per-coin health decision, and their
calculation does not alter raw-message writes. They do invalidate the promised
future cause discriminator. The same coin-global delta already drives
`consec_fail`; that behavior predates UPDATED-2, so it is not classified here as
a new v4_1 regression, but the shared defect should be repaired together.

Post-deploy fix: create attempt-local `conn_start_ns`, `conn_msgs`, and
`conn_last_recv_ns` state inside each reconnect iteration; reset connection-
local queue diagnostics there as well. Test two simultaneous tasks for one coin
and a two-attempt sequence whose second connection receives no message.

### V41-F4 — MEDIUM — the recovery instructions select a PID the recovery gate can reject

The failure table says to record the *live* candidate PID from `--inspect-live`,
and the recovery helper prints all candidates “newest last.” But
`make_recovery_bundle()` accepts only a target start in the first 120 seconds
after the ruled boundary.

Executed known-bad: after starts at T+5 (PID `222`) and T+150 (PID `333`), the
runbook-directed live PID `333` was refused as outside the start window; the
earliest boundary PID `222` produced the valid two-row recovery bundle. This is
the natural shape when `NRestarts` is the reason postflight refused.

Post-deploy fix: make recovery discovery select and print one deterministic
`V41_PID`: the earliest exact `clob_v4_1` declaration inside `[T,T+120s]`.
Report later target starts separately as restart evidence. Test the multi-start
case through the CLI path, not only `make_recovery_bundle()` directly.

### V41-F5 — MEDIUM — `CAND_COMMIT` is inert and the era receipt omits code identity

`CAND_COMMIT = "042b787"` is declared but never read. The gate pins the working
tree and HEAD versions of `collect_pm.py` to `CAND_SHA`, but it does not prove
that the named commit resolves to those bytes, and neither a normal nor a
recovered transition records the commit or SHA.

Executed known-bad: replacing `CAND_COMMIT` with
`definitely-not-a-commit` still allowed `make_stamp()` to emit a transition;
the emitted row had neither a commit nor a SHA field. At this review tip,
`042b787`, `f5fff9e`, `81d3d2c`, and the working tree all do independently hash
to the pinned collector SHA
`b4fca5a274a60fe01a5f8a0ae859daeb50e4ff4323e977dd20eb48f71085f4a2`,
so the present bytes are reconstructable. The append-only receipt itself does
not preserve that provenance, contrary to reliability rule 12.

Post-deploy fix: port the v5 gate's commit-resolution check, include commit and
SHA in normal and recovered transitions, and repair tonight's receipt in-band
with a superseding provenance receipt rather than editing the original row.

### V41-F6 — MEDIUM — CLI modes are not mutually exclusive

Every mode is a separate boolean/optional argument and `main()` selects the
first true branch. Executed against the live read-only surface,
`--inspect-live --pre-arm` printed inspect JSON and exited `0`; it did not
refuse the ambiguous request. A combination such as `--inspect-live` plus a
receipt mode under the runbook's `>> collector_runs.jsonl` shape can append a
diagnostic object, rather than an era row, to the append-only authority.
Combining `--selftest` with an emitting mode has the same output-channel class.

Post-deploy fix: put all action modes in one required mutually exclusive
argument group and validate that companion arguments occur only with their
owner mode. Add an entry-point known-bad that asserts exit `2` and empty stdout
for every representative two-mode combination.

### V41-F7 — MEDIUM — failure of the new health ledger is silent

`_health()` catches every append exception and increments `health_errors`
without logging it. The surrounding `try/except` in `_heartbeat()` therefore
cannot execute, and neither the human status line nor another ledger exposes
`health_errors`. The selftest builds the payload but does not exercise an
actual append or the failure path.

Executed known-bad: forcing `jl_append()` to raise `OSError` produced
`health_errors=1`, no exception, and empty stdout. The collector would keep
running, which is correct, but the only new machine-readable staleness signal
could remain absent forever while the service looks clean. Tonight's boundary
health reads the human log and is unaffected; later diagnosis is not.

Post-deploy fix: keep the append failure non-fatal but emit a rate-limited
stderr/status warning and include `health_errors` in the human status line.
Add an append-success control against a temporary ledger and an injected-write-
failure control that proves the failure becomes externally observable.

## Tonight's narrow operational containment

These checks do not change the ruled launch or require a code edit before it:

1. Use only the exact one-mode commands in the runbook; never combine flags on
   a command whose stdout is redirected to the era ledger.
2. At T+2 and again immediately before accepting T+6 health, read the live
   `MainPID`, `NRestarts`, `WorkingDirectory`, and `ExecStartPre`. Require PID
   to equal the recorded `V41_PID`, `NRestarts=0`, the repository working
   directory, and no pre-start command.
3. If recovery is needed and more than one v4_1 start exists, preserve the
   complete list and use the earliest exact start within the ruled 120-second
   start window; do not substitute the newest/live PID.
4. Treat `conn_msgs`, `conn_lifetime_s`, and `silence_before_close_s` as
   **UNTRUSTED for causal attribution** until V41-F3 is corrected.
5. Confirm that `collector_health.jsonl` appears and advances after restart;
   absence is a telemetry failure, not evidence of healthy silence.

## Executed evidence

As of `2026-08-31T18:48:47Z`:

| check | result |
|---|---|
| `git pull --ff-only` | already up to date |
| HEAD vs `origin/mm-research` | exact match at `81d3d2c...a431` |
| collector SHA, tree/HEAD/`042b787` | exact match at `b4fca5a2...f4a2` |
| integrated deploy gates | **ALL 14 PASS** |
| v5 preflight selftest | **232 checks pass** |
| v4_1 seam selftest | **125 checks pass** |
| DA selftest | **173 checks pass** |
| v4_1 mutation audit | **0 real survivors**; one executed downstream-redundant survivor |
| gate-runner falsifier | injected canary is the only failure; runner reports it |
| live v4_1 `--pre-arm` | **OK**, era `clob_v4`, PID `3687786` |
| live service | active/running, PID `3687786`, `NRestarts=0`, `collectors.slice` |
| shared dated/dateless parsers | integrated selftests pass |
| `git diff --check` | pass |

The green mutation result is real for existing refusal sites. V41-F1, F2, and
F5 are absent-guard defects, which deleting existing `raise` statements cannot
discover. That is why the green mutation result and these findings are not in
conflict.

## Final disposition

The 10/10 candidate and the exact normal runbook path may proceed at 22:00Z
under the USER's explicit non-gating ruling. The review does **not** certify the
broader claim that any process surviving the current postflight is the stamped
process, nor that the new disconnect fields support connection-level causal
diagnosis. V41-F1 and V41-F2 are the first post-deploy seam repairs; V41-F3 and
V41-F7 are the first telemetry repairs; V41-F4 through F6 follow before this
machinery is reused for another boundary.
