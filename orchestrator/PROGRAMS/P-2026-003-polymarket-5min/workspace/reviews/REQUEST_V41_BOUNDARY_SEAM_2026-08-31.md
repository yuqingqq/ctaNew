# Coordinator review request — v4_1 boundary seam — 2026-08-31

**Review tip:** `c8061ac`. **This supersedes the version filed at `0065e00`** —
the package changed materially after that (USER fixes + my review of them),
and the sections below marked UPDATED are the delta. **Scope: the narrow identity/boundary seam you named
as the final pre-deploy gate.** Not a re-audit of the historical cause.

## Ruled instant

**USER ruled 2026-08-31: boundary `2026-08-31T22:00:00Z`.** Read from "set as
9.1" as *make 09-01 the first clean v4_1 day* — which needs the boundary
before 09-01 begins. 08-31 becomes the mixed day and has already failed on btc
(~310 s/hr vs a 120 bar), so spending it costs nothing. 22:00Z is 2h clear of
UTC midnight (audit A1).

## COL-R3 closure

Era identity is now a function of the data-generating configuration, from one
mapping in `collect_pm.py`:

| mode | era | ping |
|---|---|---|
| `control-v4` | `clob_v4` | 3/3 — **reverted to what actually ran**, so historical reproduction stays exact |
| `control-v4-slow` | `clob_v4_1` | 10/10 — the rollback candidate |
| `app-v5` | `clob_v5` | held |

`MODE_SPEC` is the single source for identity AND cadence, so they cannot
drift. Six selftest invariants: identities distinct, coverage equals
`HEARTBEAT_MODES`, `control-v4` reproduces what ran, the slow mode's cadence
genuinely differs (a distinct identity over identical behaviour would be a lie
in the ledger), and every mode derives its version from the mapping.

`clob_v4 -> clob_v4_1` is accepted by the chain walk; DA's consumer rules the
following day pure and admissible when the era is ruled, and **REFUSES while
it is not** — DA has recommended ADMIT (Q-DA-188); the ruling is the USER's.

## The gate — `v41_boundary_preflight.py`, 30 checks

Built as a **separate file**, not by retargeting the 230-check v5 gate: that
carries 72 `clob_v5` literals and bulk-editing the instrument that governs a
production restart is the risk this programme has been paying for. It reuses
the reviewed primitives and takes **one backward-compatible parameter** on the
shared walk (`target=`, defaulting to `clob_v5`, so all 230 v5 selftests are
untouched).

`check_boundary_current` is reimplemented rather than imported — the v5 version
pins the instant to R-340's `07:00:00Z`. The phase keying is kept (audit S12).

## Self-review before sending — three gaps found, one serious

1. **The gate never checked the bytes.** The v5 gate has 25 sha references;
   this had **zero**. It would have certified a restart of unreviewed code
   with every other check green. `CAND_SHA`/`CAND_COMMIT` pinned and asserted
   at **both** pre-arm and stamp time, with known-bads for on-disk drift and
   for an uncommitted edit where HEAD disagrees with the tree.
2. The non-ASCII whitespace guards were **literal characters**, so the source
   contained a real U+2028 line separator. Now escapes; verified `0xa0,
   0x2028, 0x2029, 0x1680, 0x9, 0x200b`.
3. Runbook step 6 was a manual grep. Now `--verify-health`: two samples 30s
   apart, **every one of seven coins must advance**. Process-wide `msgs > 0` is
   satisfied by btc alone — including while six coins are dead.

**And one found by writing the falsifier:** the backwards-counter branch was
**unreachable** (a decrease also satisfies `<= 0`, so the stall check caught it
first and its known-bad failed). Reordered.

## What the stamp asserts, and deliberately does not

It records that the change removes a **measured amplifier** and that it does
**NOT** repair the 08-25 break (remote per-connection limit at the venue edge;
client exonerated, `ws_ever_paused=False` across 1,106 disconnects). It states
btc is expected **near** the bar (~123 vs 120), not clear of it. And it carries
the **measurement-basis warning**: v4_1 gap statistics are not comparable to
v4 ones because the cause mix shifts (~97% PING_TIMEOUT at 3/3 vs ~54% at
10/10), so a bar crossing at the boundary is a measurement change, not a
regression. DA's caution is in the runbook: the 10/10 days also differ by storm
and by the R-351 contamination, so **cadence is not established as the cause**
of the s/hr difference.

## UPDATED — USER fixes after the first filing, and my review of them

The USER made substantial changes. I reviewed each; they hold, and **two are
corrections to MY code that would have mattered**:

1. **The shadow observer could never have run.** My `main()` called
   `collect_pm.discover_current_slugs`, which does not exist — it always
   resolved `{}` and exited 1. My selftest covered `guard_output` and
   `summarise`: everything except the discovery the tool exists to do. **Same
   defect class as HJ-R1, committed after you raised it.** Fixed by injecting
   the collector's real read-only `gget`, with a discovery control.
2. **My `NRestarts` expectation was wrong and would have BLOCKED a clean
   deploy.** I required `nrestarts_at_arm + 1`, assuming a manual restart
   increments it. It counts AUTOMATIC `Restart=` activations only. Verified on
   the live unit: the 08-30T05:30Z boundary restart was manual and `NRestarts`
   is still 0. The USER's version requires 0 on both sides.

Also added and verified by execution: **admissibility is now MECHANISED**
(`require_target_admissible` reads DA's `ERA_ADMISSIBLE`; the gate REFUSES
while `clob_v4_1` is unruled — confirmed live), health identity bound to the
expected pid/mode/era so health cannot be read off the wrong process, a health
sampler that waits for DISTINCT status records rather than sleeping a fixed
30s, a rollback-restoration verifier with idempotency, and a shadow systemd
unit with `--verify-output` requiring a connected socket and fresh messages
from every coin.

**My own finding, added:** the runbook did not record that the shadow opens a
connection from the **same host and IP** as the collector. `BTC_GAP_DIAGNOSIS_
2026-08-26` is only MEDIUM on venue-infra vs network-path and does not rule out
a per-IP component. The shadow starts 20 min before the boundary, so harm from
it would be **confounded with the deploy and read as a v4_1 regression**.
Recorded with the mitigation the step order already provides (step 0's verify
runs before the drop-in, so a same-IP effect surfaces while still on plain v4)
and an explicit stop rule.

## Executed at `c8061ac`

All 13 gates pass. Selftests: v4_1 gate **47**, shadow **13**. Byte pin
verified against both tree and HEAD. `--pre-arm` against the LIVE system now
REFUSES by name on the unruled admissibility — which is the mechanised
precondition working, not a defect.

## What I am asking you to look at

The seam only: era identity, the boundary/rollback emitters, the health and
identity checks, and whether the runbook's step order is safe. **Ruled instant
`2026-08-31T22:00:00Z`, ~5h out.** Nothing armed; v4 live, pid 3687786.

## Known-open, not asking you to re-litigate

COL-R2 (research-slice enforcement, untracked drop-in) — you downgraded it
from a release gate; still open. The shadow observer is built and gated but
**not yet running**; the runbook requires starting it after the deploy.

Please file under `workspace/reviews/` and push.
