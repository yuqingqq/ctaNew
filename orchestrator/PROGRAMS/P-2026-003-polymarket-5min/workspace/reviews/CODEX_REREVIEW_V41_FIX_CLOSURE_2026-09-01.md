# Codex re-review — v4_1 fix closure — 2026-09-01

**Exact review tip:** `f5a48105cd22812b8fe42a19e7018634b717554b`

**Scope:** closure of the three residuals filed against `a38cb6f`: truthful
late/early-start recovery, semantic provenance verification, and behavioral
coverage of F3's attempt-local reconnect backoff. Production was inspected
read-only. No collector, service, production ledger, or tape was modified.

## Verdict

### HOLD NOT RELEASED — three residuals remain

The fixes are materially closer. A same-day late recovery can now construct a
transition at the observed process start and close it with the restoration;
sidecar provenance now resolves the named Git blob and verifies its hash; and
the pending F3 patch applies cleanly, uses an attempt-local backoff helper, and
passes its isolated collector selftest.

Those code-level positives do not clear the full paths. The sole runbook still
routes a late start away from the new recovery behavior, a cross-midnight late
start is still unrecoverable, inline provenance bypasses the new verifier, and
the F3 test does not cover the production argument that supplies its helper.
Each was reproduced below. A green integrated suite is therefore not evidence
that these pipeline seams are closed.

## Findings

### V41-RR1 — HIGH — the operator recovery path still routes real v4.1 spans to an abort that consumers erase

`make_recovery_bundle()` now accepts a same-day start after T+120s and stamps
the reconstructed transition at its observed instant. That local behavior is
correct. It is not wired to the operator path:

- `recovery_pid_candidates()` still sets `v41_pid=null` when the only target
  start is late and labels late starts “NOT candidates.”
- The sole runbook still says a null PID means use abort and still says the
  recovery gate refuses late starts.
- `make_abort_row()` accepts early and late target starts and emits
  `aborted:true`; both era consumers skip aborted rows entirely. Thus the
  evidence is present as an ignored annotation while the actual v4.1 span is
  absent from the era timeline.

Executed T+150s known-bad: discovery returned `v41_pid=null` and the runbook's
abort route remained selected. In the prior exact DA fixture, that receipt
read back `era_pure=true`, `eras_touched=['clob_v4']` when all eras were ruled
admissible, despite the included v4.1 start. The new direct-function positive
does not exercise this route.

There is also a remaining hard dead end. Executed T+3h (after UTC midnight):
`make_recovery_bundle()` refused because `_refuse_cross_midnight()` still
compares the ruled T with the target start, even though the new row would open
at the observed start. The restoration checker applies the same ruled-T guard.
Abort then silently erases the span. Pre-boundary/arm-window starts have the
same truth problem: recovery refuses them while abort accepts and consumers
skip them.

Required closure:

1. Make discovery and the runbook select the actual late PID for observed-time
   recovery; do not direct any observed target start to an ordinary abort.
2. Apply midnight safety to the effective observed boundary used by a
   reconstruction, not to the missed ruled instant. Preserve the restoration
   ordering check against the target start.
3. Either reconstruct early spans at their observed starts or define an
   explicit contamination receipt consumed by both era readers. An ignored
   field on an aborted row is not an era record.
4. Add an entry-point test from `--recovery-pid` through emitted rows into
   `day_era_admission()` for same-day late, cross-midnight late, and early
   starts. The day must be impure/reconstructed and ineligible.

### V41-RR2 — MEDIUM — inline provenance bypasses semantic verification

The new sidecar branch is substantially correct: it binds era/PID/start,
resolves the commit, hashes `collect_pm.py`, and the actual production record
returns `SUPERSEDED` with the correct `2b1ea0d...` / `4d15d2dd...` identity.

The inline branch returns before any of those checks. Executed known-bad: a
transition containing `collector_commit='000...000'` and
`collector_sha256='000...000'` returned `status='INLINE'`. Neither object
exists nor identifies collector bytes.

Required closure: route inline and sidecar identities through one resolver
that validates shape, resolves the commit, hashes the named collector blob,
and compares the digest. Add the all-zero inline falsifier and a real inline
positive control.

### V41-RR3 — MEDIUM — the F3 backoff helper is tested, but its production wiring is not

The pending patch now uses `next_consec_fail()` in production and its unit
tests correctly cover silent escalation and worked-connection reset. However,
the tests call the helper with hand-built values; they do not prove that the
production call supplies attempt-local connection messages.

Executed mutation in an isolated patched clone: replace only the production
argument `conn["msgs"]` with the old coin-global delta while leaving the helper
and all its tests intact. `collect_pm.py --selftest` still exited `0`. That
mutation restores the exact sibling leak the fix exists to prevent.

Required closure: exercise `_market()` with a silent target connection and an
active same-coin sibling, or encapsulate attempt state so the production
producer and the backoff transition are tested together. The known-bad must
fail when only the production argument is changed back to coin-global state.

## Executed evidence

| check | result |
|---|---|
| HEAD vs `origin/mm-research` before filing | exact at `f5a4810...` |
| working-tree collector vs deployed/pinned collector | exact SHA `4d15d2dd...` |
| pending F3 `git apply --check` | pass |
| pending F3 isolated patched selftest | pass |
| F3 production-wiring mutant | **selftest still passes** |
| late-only recovery discovery, T+150s | `v41_pid=null`; abort instruction remains |
| cross-midnight recovery, T+3h | **refused** by ruled-T midnight guard |
| bogus inline commit/SHA | **accepted as `INLINE`** |
| actual production provenance | `SUPERSEDED`, correct commit and SHA |
| integrated deploy gates | **ALL 14 PASS**; v4_1 selftest 154 |
| live health | pass; fixed PID and all seven coins advance |
| live unit | active/running, PID `1108125`, `NRestarts=0` |
| `git diff --check` | pass |

## Disposition

The current live v4.1 collector remains healthy and byte-consistent. Do not
restart it and do not apply the pending F3 patch outside a new ruled boundary.
The hold concerns reuse of the failure/recovery machinery and deployment of
the pending diagnostic patch, not an instruction to disturb the healthy live
process.

