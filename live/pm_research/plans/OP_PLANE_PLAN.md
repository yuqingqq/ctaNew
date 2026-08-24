# OP plane plan — Monitor/KillSwitch and LatencyBudget

Status: **DESIGN — REVISION 3.** Applies DE cross-review iterations 1 and 2
(`DE_OP_CROSS_REVIEW_LOOP.md`): iteration 1's **11 MUST-FIX + 7 SHOULD-FIX**
(§11, one finding corrected on verification) and iteration 2's **3 MUST-FIX +
2 SHOULD-FIX** (§12, one SHOULD-FIX declined as superseded by R-55). Every
finding verified against the artifacts before applying. **Not converged** — the
loop's stop rule is two consecutive zero-MUST-FIX iterations and iteration 2
returned three.
Owner: OPS. Dispatch **B5**. Written 2026-08-23 in one pass.

**Review state: DE cross-review iteration 1 APPLIED (`DE_OP_CROSS_REVIEW_LOOP.md`,
verdict `DEFECTS_FOUND`).** 11 MUST-FIX + 7 SHOULD-FIX, each verified against the
artifacts before applying; one finding did not hold as written and is recorded in
**§11** rather than silently adjusted. The loop's stop rule is two consecutive
zero-confirmed-MUST-FIX iterations, so **this document is not yet converged** —
iteration 2 re-reviews this revision.

The prediction held. The programme's base rate for unreviewed single-pass plans
is bad — DA returned 31 MUST-FIX against `SP_PLANE_PLAN`, BE returned 20 plus
`REFUTED_IN_SUBSTANCE` against `EV_GATES_PLAN` — and this plan, written the same
way, was no exception. **The heaviest defect was at the kill switch**: `R-HALT`
cannot fire, because nothing in v22 consumes `CancelAllStatus` and Revision 1
froze that type out of `OP-Monitor`'s consume list (§1, MF1).

**Known-weak areas, flagged by the author rather than waiting to be found:**

1. **§5's four legs are asserted, not derived.** `PM_ARCHITECTURE.md` says only
   "four legs, ack unobserved"; the observe/decide/transmit/confirm decomposition
   is mine and nothing validates that it is the right cut, or that it is
   exhaustive.
2. **§2's "health is declared, never inferred" may be too strict to build.** No
   module currently publishes a telemetry port, so the rule has zero instances
   and has never been tested against a real producer.
3. **§7's retrofit table maps built artifacts to OP concepts as "analogues".**
   That word is doing a lot of load-bearing work and may be hiding a category
   error rather than naming one.
4. **§3's latching rules are copied from the contract, not reasoned from
   failure modes.** No fault taxonomy justifies why DEGRADED exists as a
   distinct rung from HALTED here.
5. **§8a's class assignments** were derived under R-6/R-8 and one of them
   (`staleness_deadline` as "Class A on a Class C base") invents a split the
   ruling does not contain.

Authority: `contracts/contracts.yaml` **v22** defines the types; this plan does
not redefine them. `PM_ARCHITECTURE.md` §1 defines the plane edges. Where this
plan and either of those disagree, they win and this file is stale.

## 0a. R-36(2) applied to THIS plan — what is IN FORCE and what is PROSE-ONLY

**R-36(2): a rule with no nameable landing evidence is PROSE-ONLY and may not be
cited as in force.** The coordinator applied that to their own rulings and
re-opened R-24 under it. Applied honestly here, **most of this plan is
prose-only**, and that must be visible at the top rather than inferred by a
reader who assumes a plan describes a system.

**R-42: this section no longer DECLARES what is in force — a script REVEALS it.**
`ops/verify_landing_evidence.py` executes the command behind every claim below
and compares it to an expected value. Anyone can re-derive the plane's true state
in one call, without trusting an OPS report, a revision string, or git:

```bash
python3 live/pm_research/ops/verify_landing_evidence.py    # exit 0 == every claim holds
python3 live/pm_research/ops/pm_lane_health.py --selftest  # exit 0 == every check demonstrated failing
```

It **does not fail open**: a command that errors, or returns nothing, is a FAIL
and never a skip. Negative controls demonstrated — a wrong expected value, a
non-existent command, and empty output all read FAIL.

### IN FORCE — built, with landing evidence and a demonstrated failing witness

| what | landing evidence |
|---|---|
| supervision units, caps, OOM priority | `systemctl --user show` — `MemoryMax=17179869184`, `MemoryHigh=infinity`, `OOMScoreAdjust=1000` on both batch units; collectors untouched at `OOMScoreAdjust=200`, same MainPIDs since 2026-08-21 |
| evaluation chained to measurement | `systemctl --user show pm-measurement-pipeline.service -p OnSuccess` → `pm-evaluation-pipeline.service`; eval timer demoted to `03,09,15,21:50 UTC` |
| the health checks | `pm_lane_health.py --selftest` → **exit 0**, every check driven to its failing state on a synthetic witness. **Counts are deliberately NOT quoted here** — iteration 3 found the literals stale at two of three sites, so the row names the command and its exit contract instead. A count in prose is a claim; an exit code is a check |
| three of them fired on real incidents | `UNITS` (failed unit + CONTENTION), `LANE_PROGRESS` (0 committed days), `NO_PROGRESS` (`cpu_frac 0.0014`, `stall_frac 0.9799`, `LIVELOCKED_IN_RECLAIM`) |
| `derivation_lag_days` beside every day count | `pm_lane_health.py --json` → `LANE_PROGRESS.lanes[].derivation_lag_days` with `lag_floor_days: 1` |
| alert delivery | `lane_health.jsonl` records the channels each alert went out on |

### PROSE-ONLY — no mechanism exists; MAY NOT BE CITED AS IN FORCE

| section | why it cannot be cited |
|---|---|
| §2 telemetry-port observation model | **zero modules publish a telemetry port.** The rule "health is declared, never inferred" has no instance and has never met a producer |
| §2.1 heartbeat / `PortLiveness` | zero `HeartbeatRegistration`s exist; `LIVE/STALE/UNKNOWN` is never computed |
| §2.2 registry, reconciliation, unregistered-publisher | no registry exists, and the built analogue is hard-coded open-world — the opposite of what §2.2 requires |
| §3 ladder · §3.1 transitions · §3.2 reset | **no code produces a `HaltState`.** The transition table, skip-rung legality and reset race rules are unexercised |
| §4 the three halt edges | no consumer; nothing reads a halt |
| §5 LatencyBudget | legs 2–4 unmeasured, no code; only leg 1 has a number (47 ms book, 1,700 ms signal clock) |
| **`R-HALT` as a protection** | **REPAIRED AT CONTRACT LEVEL, STILL PROSE-ONLY.** v23 (R-68) added `CancelAllStatus` to `OP-Monitor.consumes`, so the rule is no longer **unevaluable** — the specific R-24-class defect (a gate whose type had no consumer) is gone. It remains PROSE-ONLY for the ordinary reason: **no code produces a `HaltState` or reads a `CancelAllStatus` at runtime.** The distinction matters — *could never fire* is a design defect; *is not built yet* is a build state |

**The honest summary: the OP plane is a supervised research pipeline plus a
design for a runtime that does not exist.** That is the correct state — DE is
unbuilt and live quoting is a programme non-goal, so building it would be
speculative — but the two halves must never be cited as though they were one
thing.

## 0. Why this document exists, and what it is not allowed to do

The build-readiness audit (§6) found `OP-Monitor` and `OP-LatencyBudget` with
**no owning plan while being built**. That is the sequencing the programme has
just been told to stop, so this plan is written to catch up to the build and
then constrain it — not to bless it.

**The most important thing this retrofit has to prevent is a naming error.**
What OPS has built is supervision for the **research collection and derivation
lanes**: systemd units, a lane-health check, an alert path. What this plan
specifies is a **runtime** OP plane that observes a live decision system and can
halt it. **They are not the same system and must not converge by accident.**
There is no live quoting in this programme (an explicit non-goal: no live
orders, no exchange adapters), so `OP-Monitor` as specified below **has no
runtime to attach to today** — `DE` is unbuilt and nothing can consume a halt.

The programme has recorded *the name is not the definition* five times. This
would be the sixth: `pm_lane_health.py` is **not** `OP-Monitor`. §7 keeps them
apart explicitly.

## 1. Plane position, and the edge that must never exist

`SP ← DA ← BE ← DE`. **EV reads all planes and is read by none.** OP has both a
data path and a command path (M5-9) — it cannot both function and depend on
nothing, since the monitor observes health and the budget consumes measurements.

```
HealthEvent  (owner: OP-Monitor; sources: DECLARED TELEMETRY PORTS of
              DA/BE/DE modules · OP-LatencyBudget — NEVER EV)
   → HaltState          LATCHED, FAIL-CLOSED (unknown health ⇒ HALTED)
   → halt port ──┬──→ DE-Constraints.halt_in    HARD constraint: FeasibleSet = ∅
                 ├──→ DE-Actuator.halt_in      HARD constraint, second door
                 └──→ DE-Actuator.cancel_all   PRIORITY command (cancel_cmd_in)
```

**EV is not a health source (M7-5), and this is a hard rule, not a preference.**
v7 listed EV among the sources, which created `EV → OP → DE` and let
**evaluation state reach decisions**. That edge was removed deliberately. Any
future request to "halt on a bad markout", "degrade when calibration drifts", or
"gate quoting on a gate result" is that same edge wearing new clothes and must
be refused at this plane.

The legitimate path for evaluation to stop the programme already exists and is
separate: `EV-Gates.on_fail = HALT_PROGRAM` is **programme control operated by a
human owner**, not a runtime edge. The distinction is the point — one is a
person deciding to stop the research, the other would be evaluation output
silently steering live decisions.

**Rule OP-1.** `OP-Monitor.consumes` is `HealthEvent`,
`HeartbeatRegistration`, `HeartbeatPulse` — **plus `CancelAllStatus`, which
Revision 1 wrongly omitted (MF1)**. Adding any EV type to that list is a
cross-plane contract change and is coordinator-gated (§2.2).

**MF1, verified and load-bearing: `R-HALT` cannot fire as currently wired.**
`contracts.yaml` v22 declares
`R-HALT.fail_closed: CancelAllStatus.Unconfirmed => HaltState.HALTED`, and
`CancelAllStatus` is **produced** by `DE-Actuator` (v22 line 1796) and
**consumed by nothing** — grep confirms exactly three occurrences: the rule, the
type, and the producer. Revision 1 then froze `OP-Monitor`'s consume list
*without* it, so this plan actively specified out the one type its own kill-switch
rule depends on. **A gate that cannot fire, at the kill switch** — the
programme's fifth logged instance, and the worst-sited one.

**Dependency handed to the coordinator (additive, §8):** `OP-Monitor.consumes`
must gain `CancelAllStatus`. Until that lands, **`R-HALT` is UNEVALUABLE and must
not be cited as an active protection**, including in DE's carry analysis.

## 2. OP-Monitor — what it observes, and from where

**MF4 — the premise is currently FALSE against v22, and Revision 1 asserted it
as fact.** Not every module carries `telemetry_out`: two of the four
HealthEvent sources DE's plans depend on have **no telemetry port at all** until
DE's queued additive contract fix lands, and the v22 ports map is
self-inconsistent between the per-module records and the wildcard defaults. So
this section describes the **target** wiring, not the wiring that exists.
**Dependency handed to the coordinator and DE in §8**, which Revision 1 failed to
do: until those ports exist, `OP-Monitor` has no lawful source for those modules'
health and must treat them as `UNKNOWN` — which, by §2.1, means `HALTED`. That is
the fail-closed answer and it is deliberately inconvenient.

**Only from declared telemetry ports.** Where a module carries `telemetry_out`
in its port record; `OP-Monitor` carries `telemetry_in`,
`heartbeat_in`, `halt_out`, `cancel_cmd_out`, `clock`. Observation is confined
to those ports: the monitor does not read module internals, does not read
`StateView`, and does not infer health from data content. **Health is declared
by the observed module, never inferred by the observer** — an observer that
infers health has invented an undeclared dependency on that module's internals.

### 2.1 The two observation mechanisms

| mechanism | type | answers |
|---|---|---|
| **event** | `HealthEvent{id, source_port, severity: INFO/WARN/FAULT, observed_at, detail}` | "something bad happened" — presence of error |
| **liveness** | `HeartbeatRegistration{port, period, staleness_deadline}` + `HeartbeatPulse{port, seq, observed_at}` → `PortLiveness{port, last_seen, state: LIVE/STALE/UNKNOWN}` | "is this port still working" — **absence** of progress |

**Both are required, and the second is the one that gets forgotten.** An
event-only monitor sees every failure that announces itself and none of the
failures that consist of nothing happening. That distinction is not theoretical
here — it is exactly the 26 h outage in §6.

**Each liveness state has a halt consequence, and `STALE` is the one that was
missing (MF5).** A registered module going silent is *the* D-1b class this plane
exists to catch, and in Revision 1 it dead-ended before the halt:

| `PortLiveness.state` | meaning | consequence |
|---|---|---|
| `LIVE` | pulse inside `staleness_deadline` | none |
| `STALE` | registered, pulsed before, now past its deadline | **`HealthEvent{severity: FAULT}` ⇒ `DEGRADED`**; a second consecutive deadline, or `STALE` on a module the halt path depends on, ⇒ `HALTED` |
| `UNKNOWN` | registered but never pulsed, or clock unusable | **fail-closed `HALTED`** |

Computed from registration + pulses + the monitor's own clock. Note the clock is a declared
port: a monitor that reads liveness against an unsynchronised or stalled clock
reports LIVE for a dead port, so clock health is itself a `HealthEvent` source.

### 2.2 Registration is mandatory and closed-world

A port that has not registered is not "healthy by default" — it is **UNKNOWN,
therefore HALTED**. The registry of expected ports is an `SP`-owned constant
(§8), because "which modules must be alive" is a configuration fact, not a
runtime observation. **An unregistered producer is the failure mode that reads
as silence**, and silence must never read as health.

**MF6 — and Revision 1 reproduced D-1b inside the fix for D-1b.** A closed-world
registry is only as complete as the human who wrote it: a module that is never
registered is never missed, so the omission is silent in exactly the way the
outage was. Two rules, which Revision 1 lacked:

> **RECONCILIATION.** `OP-Monitor` MUST periodically compare the registry against
> the modules actually present in the composition root, and raise
> `HealthEvent{severity: FAULT}` for any **registered-but-absent** module. A
> registry that is never checked against reality is documentation.
>
> **UNREGISTERED-PUBLISHER.** Telemetry arriving on a port with no registration
> is itself a `FAULT`, never discarded silently. An unexpected publisher means
> the registry is wrong, and which of the two is wrong is not the monitor's
> call to make quietly.

**Recorded honestly (§7.2): the built analogue does neither.**
`pm_lane_health.py` hard-codes its unit and producer lists open-world — it checks
the things it names and is blind to anything it does not — so today the built
supervision has precisely the defect this section forbids in the design.

## 3. The HaltState ladder — latched and fail-closed

```
HaltState { level: RUNNING | DEGRADED | HALTED,
            latched: bool, since: Timestamp,
            reason: list[HealthEvent],
            reset_authority: OP_OWNER_ONLY }
```

| property | rule | why |
|---|---|---|
| **monotonic** | `RUNNING → DEGRADED → HALTED` escalates **automatically** | escalation must not wait for a human who may be asleep |
| **latched** | the level does not fall when the triggering condition clears | a fault that flaps is not a fault that fixed itself; the state records that it happened |
| **de-escalation** | **explicit operator reset only** (`OP_OWNER_ONLY`), never automatic, never timed | auto-recovery re-arms a system whose fault was never diagnosed |
| **fail-closed** | unknown health ⇒ `HALTED` | the conservative state must be the default, not the exception |

**R-HALT, from contracts v22:** while cancellation is `Unconfirmed` the system
state is `HALTED` — `CancelAllStatus.Unconfirmed ⇒ HaltState.HALTED`. This is
the fail-closed rule applied to the one command whose completion cannot be
observed (§5). "We asked for a cancel and do not know whether it happened" is
**not** a running state.

### 3.1 The transition function (MF9)

Revision 1 named the levels and the latch and never said what *causes* a
transition, so `DEGRADED` had no producing rule at all — a state nothing on the
OP side could ever enter, which two of this plan's own rules nonetheless require.

| from | trigger | to |
|---|---|---|
| `RUNNING` | `HealthEvent{severity: WARN}`, or one `STALE` port | `DEGRADED` |
| `RUNNING` | `HealthEvent{severity: FAULT}`, any `UNKNOWN` port, or `CancelAllStatus.Unconfirmed` | `HALTED` (skip-rung, legal) |
| `DEGRADED` | `FAULT` · `UNKNOWN` · second consecutive `STALE` deadline · `Unconfirmed` | `HALTED` |
| any | operator reset **after** clean reconciliation | `RUNNING` |

**Skip-rung escalation `RUNNING → HALTED` is explicitly legal**: requiring a stop
at `DEGRADED` would delay the halt by one observation interval for a fault that
is already conclusive. `INFO` severity never transitions; it is recorded only.
**De-escalation never appears in this table by any path except operator reset**,
which is the latch.

### 3.2 Reset semantics (MF10)

Three gaps Revision 1 left, each with a concrete failure:

1. **`reason` list growth is unbounded.** A flapping port appends for ever.
   **Rule:** `reason` retains the **first** event of the current halt episode
   plus the most recent N; the full sequence is appended to an immutable episode
   archive keyed by `since`. "The state records that it happened" is otherwise a
   claim with no mechanism behind it.
2. **The fault-during-pending-reset race can erase an undiagnosed fault.** If a
   new `FAULT` arrives between the operator's reconciliation and the latch clear,
   a naive clear drops it. **Rule:** reset is **compare-and-clear on the episode
   id** — a reset naming episode *k* is refused if the current episode is not
   *k*, and the operator must re-run reconciliation.
3. **Reset is not a single act.** It is: reconcile → verify no active `FAULT`
   and no `UNKNOWN`/`STALE` port → clear latch naming the episode. Failing any
   step leaves the latch set.

**Reset requires reconciliation, not just an operator.** Per
`DE_MODULE_PLAN.md`: on every recovery, restart, and halt-reset, **reconcile
open orders and fills against the venue before any quoting resumes** —
reconcile-before-quote, no exceptions. Divergence between reconciled and
internal state is itself a `HealthEvent`. So a reset that clears the latch
without a clean reconciliation is not a valid reset.

## 4. The edges out of the halt port — and why one is not enough

**Three edges, not two (MF12).** Revision 1 described only the Constraints edge
and `cancel_all`, omitting `HaltState → DE-Actuator.halt_in`, which v22 declares
(`DE-Actuator.ports` carries `halt_in`) and which DE's plans rely on as the
second door. They do different things and none substitutes for another:

| edge | consumer | effect | what it cannot do |
|---|---|---|---|
| `HaltState` as a **hard constraint** | `DE-Constraints` (`halt_in`) | stops **NEW** risk: the feasible set collapses | **does not retract resting orders** — existing exposure stays live |
| `HaltState` on the **Actuator's own** `halt_in` | `DE-Actuator` (`halt_in`) | second door: the Actuator refuses new venue writes even if a `Decision` reaches it | does not retract what is already resting |
| `cancel_all` as a **priority command** | `DE-Actuator` (`cancel_cmd_in`) | retracts resting orders | does not by itself prevent new quoting |

**The critical routing fact:** `cancel_all` **bypasses the solver but NOT the
Actuator.** During a fault the solver may itself be `Unavailable` — that is a
normal, typed outcome in this architecture, not an exotic one — so **a path that
runs only through the DecisionScheme cannot fire when it is most needed**. The
command therefore routes `OP-Monitor.cancel_cmd_out → DE-Actuator.cancel_cmd_in`
directly.

It stops at the Actuator rather than reaching the venue because **`DE-Actuator`
is the sole venue writer**. OP publishes state and commands; it never touches
the venue. A kill switch with its own venue connection is a second writer, and a
second writer is how you get an order the system does not know it has.

```
CancelAllCommand { command_id, idempotency_key, issued_at, deadline,
                   backoff: BackoffPolicy, rate_budget_class: PRIORITY }
```

**Issuance is bound to the transition (MF2).** Revision 1 typed and routed the
command but never said *when* it is sent, while DE's carry analysis assumes it
fired. Binding, stated as a rule:

> **On every transition into `HALTED`, `OP-Monitor` MUST issue exactly one
> `CancelAllCommand`**, retried under its `BackoffPolicy` until `Confirmed` or
> `FailedTerminal`, and MUST NOT issue one on `RUNNING → DEGRADED`. Re-entering
> `HALTED` while a command is `Unconfirmed` reuses the same `idempotency_key`
> rather than issuing a second command.

Until the `CancelAllStatus` consume edge exists (MF1), issuance can be commanded
but its completion **cannot be observed by OP**, so `R-HALT`'s conservative
`HALTED` cannot be entered from `Unconfirmed`. That is the gap, stated rather
than papered over.

`idempotent` + `retry_until_ack` + a `PRIORITY` rate class: the command must
survive retry (so it may be delivered more than once) and must not queue behind
ordinary traffic in the very rate budget a fault is likely to have exhausted.

## 5. OP-LatencyBudget — four legs, the fourth unobserved

The budget decomposes the observe→act path. Legs 1–3 are measurable from our own
clocks; **leg 4 is not observable at this venue**.

| # | leg | from → to | observable | measured / status |
|---|---|---|---|---|
| 1 | **observe** | venue event → our knowledge (`recv_ns`, normalized) | yes | book arrives in **47 ms**; the *signal* clock is 1,700 ms of which **1,440 ms is PM-side publish delay** and untouchable |
| 2 | **decide** | knowledge → `Decision` emitted | yes | unbuilt; DE has no runtime |
| 3 | **transmit** | `Decision` → venue accepts the write (`t_submitted`) | yes | out of scope — no venue access in this programme |
| 4 | **confirm** | acceptance → effect confirmed (`t_acked`) | **NO** | `OrderRecord.t_acked` is nullable *because of this* |

**Venue acks are not observed.** Confirmation is by **open-order reconciliation**
(`CancelAllStatus`: `Confirmed` carries a `ConfirmationSource`), which means
deployment can only ever measure an **upper bound** on cancel latency. The
design consequence is stated so nobody later reads the bound as a point
estimate: everything downstream must degrade conservative on it.

### 5.1 Ack latency SELECTS the τ rung — it does not trigger re-research

This is a **designed seam**, and the distinction matters for sequencing:

- the DE cancel ladder is `τ = 0 … 1000 ms`, with **τ=0 the knowledge-lag floor
  (250 ms), not zero latency** — the race is not in the tape;
- the operative rung is the **smallest ladder rung ≥ the measured upper bound**;
- it is stored as an **`SP-Params` value with provenance** and consumed by the
  Actuator;
- if reconciliation cannot resolve adjacent rungs, **the coarser rung applies** —
  the seam degrades conservative, never optimistic.

So a deployment measurement **moves the operative rung along a pre-built ladder**.
It does not invalidate the replay, and it does not send the cancellation family
back for re-research. Replay results are computed **per rung** precisely so this
substitution is legal.

### 5.2 The 1000 ms top rung — **CLASS D, FROZEN by Ruling R-8, 2026-08-23**

**What the bar decides, stated so it cannot be reinterpreted later:**

> If the deployment-measured `DE-Actuator` ack-latency bound **exceeds 1000 ms**,
> the **cancellation lever is dead at deployment — independent of any replay
> result.**

Frozen under R-8 after OPS raised that `SP` §4 classified `tau_ladder` as Class A
(freely changeable) while `DE_MODULE_PLAN.md` §7.3 made this rung a falsifier.
Interior rungs `{0, 50, 100, 500}` remain **Class A** — they set resolution and
no verdict turns on them. **`τ = 250 ms` is NOT among them (MF2):** it is the
R-1-frozen **Class-D decision rung** on which the `ww_v1` verdict is computed,
and Revision 2 re-planted the old misclassification here while correcting it in
§8a's table. Stated at both sites now, because a reader of this section alone
re-inherited the error MF8 removed. The top rung is **Class D**: it may not be moved
now that it is frozen, and extending the ladder after a bound is measured would
convert a refutation into a pass by fiat.

Frozen at the only moment it was free: `tau_operative` is measured at deployment,
`DE-Actuator` is unbuilt, venue access is out of scope, so the measurement cannot
have run and **no verdict depends on the bar**. This is a real kill condition for
a whole DE lever, owned by this plane, and it is why leg 4 being unobservable is a
plane-level fact rather than an implementation footnote.

**Recorded accurately, not quietly dropped: the lever this rung guards is already
dead by a different route.** DE's `ww_v1` returned **DEAD on eight of eight
coin-days**, so the cancellation family is closed **on this tape** — by a
warning-window measurement, not by latency. The bar therefore currently decides a
question measurement has already answered.

That is a reason to keep it, not to retire it. `ww_v1` returned DEAD on **8 of 8 coin-days** (the R-9 day series; Revision 1
also said "one UTC day" in §8d, which was the pre-R-9 scope and is corrected
here — SF15). Its scope is still tape-bound: a different venue, a re-sample under
R-ADMISS, or a tape with different warning-window structure could reopen the
family. **If that happens the
latency question returns, and the bar frozen today is what makes the re-test
honest** — a bar set before anyone knows the answer, rather than one negotiated
once a bound is in hand. Two independent closes on the same lever is the
programme's evidence being consistent, not redundant.

**R-38 clause (d), applied to this rung.** An amendment to the 1000 ms bar may
not, by itself, change a verdict: it renders any verdict computed under it
**UNDETERMINED**, never the opposite, and re-establishing one requires
**re-running the ack measurement under the new bar at the original evidentiary
standard**. Concretely — nobody can amend the ladder from "cancellation lever
dead" to "lever alive"; they can only amend to "not yet determined", and then
they owe a deployment measurement. **Status today: no verdict has been computed
under this bar** (the ack measurement cannot run — `DE-Actuator` is unbuilt and
venue access is out of scope), so there is nothing to vacate, and the bar is
frozen ahead of its evidence exactly as R-8 intended.

**One thing this rung cannot do:** a favourable ack measurement **cannot revive
the family**. Latency was never what killed it. Only a re-sample that overturns
`ww_v1` can reopen the question, and then this bar applies to what follows.

## 6. The alerting gap — the failure class, and the invariant that catches it

### 6.1 What actually happened

Both batch units failed **every hour for ~26 h**. The journal recorded all of it.
The collectors stayed green. `tier1/batches/` and `tier2/` did not exist — **no
receipt had ever committed** — and the programme found out because a coordinator
ran `systemctl` by hand.

### 6.2 The failure class, named precisely

**Absence of work is invisible; only presence of error is visible.** Two
independent mechanisms produced that, and both must be named or the fix is
partial:

1. **The error had no reader.** Failure was reported to the journal, which is
   pull-only. A channel nobody polls is not a channel. This is the same defect
   as reading `open_gaps=[]` as "clean" while the prices lane logged 58 gaps in
   11 h — the check ran, the result was never *delivered*.
2. **Success and no-op are indistinguishable by design.** Both units run
   `--scheduled`, under which `IDLE` and `BLOCKED` are **successful** exits so
   the hourly timer can retry quietly. That is correct for a retry loop, and it
   means **unit-exit-status can never answer "is the lane producing"**. A lane
   can idle forever with every unit green.

An alert built only on (1) would have caught this outage and **missed its
successor**, because the next stall need not crash at all.

### 6.3 The invariant

> **Every registered work-producing port emits progress within its declared
> staleness deadline; absence of progress within the deadline is STALE, absence
> of registration is UNKNOWN, and both are alerting states.**

Note that this is **not a new invention** — it is `HeartbeatRegistration` +
`PortLiveness` from contracts v22, with `UNKNOWN ⇒ fail-closed HALTED`. The
research lanes simply never registered such a port, so the primitive that would
have caught this existed in the type system and had no instance.

The operational form: **progress is measured on committed receipts, never on
partitions on disk.** A partition is work in flight; a receipt is work done, and
receipts are written last precisely so their presence is unambiguous.

### 6.4 Fail-closed for a reporter vs fail-closed for a kill switch

`HaltState` latches because it gates risk. A research-lane reporter that latched
identically would need an operator reset after every transient, and would be
switched off within a week — the classic path by which fail-closed becomes
fail-ignored. This plan therefore records **two different disciplines** and
where each applies (§7.2), rather than pretending one rule fits both.

## 7. Retrofit — what is built, and how it maps

### 7.1 Built today (research-lane supervision, NOT the runtime OP plane)

| artifact | role | maps to |
|---|---|---|
| `ops/pm-collector-*.service` | process supervision, `Restart=always`, linger | no OP analogue — process liveness, below the plane |
| `ops/pm_lane_health.py` `UNITS` | unit failure | `HealthEvent{severity: FAULT}` **analogue** |
| `ops/pm_lane_health.py` `LANE_PROGRESS` | eligible day uncommitted past grace | **`PortLiveness` analogue — this is §6.3's invariant** |
| `ops/pm_lane_health.py` `COLLECTOR_PROCS`, `TAPE_FRESH` | producer liveness | `PortLiveness` analogue |
| `ops/pm_lane_health.py` `GAP_RATE` | recorded collector loss | `HealthEvent{severity: WARN}` analogue |
| `ops/pm_lane_health.py` `MONITOR_LIVENESS` | the monitor's own outage, self-announced on resumption | `PortLiveness` applied to the observer itself |
| `ops/pm_lane_health.py` `TIER1_LOCK` | single-writer lock held, by whom, how long | no OP analogue — a resource observation |
| `ops/pm_lane_health.py` `NO_PROGRESS` | running batch below 5 % CPU with >50 % full stall = livelocked in reclaim | `HealthEvent{severity: FAULT}` analogue |
| `OnFailure=pm-alert@%n.service` | immediate delivery on crash | delivery, no OP analogue |
| `ops/lane_health.jsonl` | append-only ledger incl. delivery | `HaltState.reason` analogue |

**Every row says "analogue" deliberately.** None of these consume a declared
telemetry port; they observe the lanes from outside, by reading systemd state
and the filesystem. That is legitimate for supervising a research pipeline and
it is **not** the observation model in §2, which requires modules to *declare*
health rather than have it inferred. Converting these into `OP-Monitor` would
require the lanes to publish telemetry ports — which is work nobody has demanded
yet, and this plan does not demand it either.

### 7.2 Divergences, recorded rather than smoothed over

1. **The built checker does not latch.** `ALERT.txt` is deleted when conditions
   clear. Per §3 that is wrong for a kill switch and right for a reporter: the
   append-only `lane_health.jsonl` is the durable record, and nothing gates on
   the live file. **If anything ever gates on it, it must latch first.**
2. **The built checker has no `DEGRADED` rung** — it reports OK/WARN/ALERT, which
   is not `HaltState`'s ladder and should not be renamed into it.
3. **The monitor's own death — PARTLY closed, and the residue is named.**
   If the health timer stops, the silence looks identical to health — **the
   original bug, one level up.** `MONITOR_LIVENESS` now reads the previous run's
   timestamp from the append-only ledger and alerts when the gap exceeds two
   timer periods (bar 1800 s), so an outage **self-announces on resumption**
   instead of vanishing. Coverage is mutual rather than self-referential: two
   *independent* schedules reach this code — the 15 min health timer and the
   `OnFailure` hook on the hourly batch units — so a stopped health timer is
   still caught by the next batch failure, and vice versa. Verified: a synthetic
   stale ledger alerts (gap 10,902 s > 1800 s); an absent ledger does **not**
   alert, so a first run cannot false-alarm.
   **RETRACTED IN REVISION 2 (MF7): the on-box half is NOT closed.** The
   mutual-coverage claim assumed the batch units would *fail* and pull the alert
   path. They need not: under `--scheduled`, `IDLE`/`BLOCKED` exit **green**, so
   the composite **health timer dead + batch units idling successfully = total
   silence** — `OnFailure` never fires, and `MONITOR_LIVENESS` never runs to
   notice its own absence. That is this plan's own §6.2 lesson, that success and
   no-op are indistinguishable, applied to the monitor itself: Revision 1 wrote
   the lesson and fell for it one section later. Testable today, never tested;
   **that test is now the first item of §10.**
   **Residue that IS irreducible:** if everything stops, nothing reports — a
   process cannot observe its own absence — and closing that needs an off-box
   observer this host has no channel for (item 4).
4. **No out-of-band channel exists on this host** (no MTA, no webhook, no
   credential), so nothing pages anyone when no session is open. Recorded as a
   user decision, not an OPS one.

### 7.3 Not built, and correctly so

`OP-Monitor` proper, `HaltState`, the halt port, `cancel_all`, and legs 2–4 of
the budget. All require a runtime decision system; `DE` is unbuilt and live
quoting is a programme non-goal. **Building them now would be speculative**,
which the demand-driven build order forbids. This plan exists so that when
demand arrives the design is already reviewed — not to authorize building it.

## 8. Dependencies this plane hands to other seats

0. **CONTRACTS, coordinator — two ADDITIVE changes this plane cannot work
   without, both surfaced by DE cross-review iteration 1:**
   **(a) `OP-Monitor.consumes` gains `CancelAllStatus` — DISCHARGED in v23
   (R-68).** Verified in the file: `OP-Monitor.consumes = [HealthEvent,
   HeartbeatRegistration, HeartbeatPulse, CancelAllStatus]`. `R-HALT`'s
   `Unconfirmed ⇒ HALTED` is now **evaluable in principle**. It is still not an
   active protection, because nothing implements it — but the reason changed
   from *unevaluable* to *unbuilt*, and only the first was a defect.
   **(b) `telemetry_out` on the modules that lack it — SUBSTANTIALLY DISCHARGED
   in v23.** Verified: **every `DE-*` module now carries `telemetry_out`**, so
   the acting path DE depends on is covered. The residual four are
   `BE-Competition`, `BE-CompetitionAggregator`, `BE-ScenarioProvider` and
   `DA-Settlement` — all **deferred or facts-only**, none on the acting path, and
   each correctly outside it. If any is later promoted, its port comes with it;
   until then those modules are `UNKNOWN` ⇒ `HALTED` by §2.1, which is the right
   answer for a module that does not run.

1. **`SP-Params` / `SP` plane (B1, coordinator).** The registry of expected
   telemetry ports and their `period` / `staleness_deadline`; the τ ladder rungs
   and the operative-rung provenance field; the halt-reset authority identity.
   Every one is a configuration constant, and §6 already found `SP` ownerless.
2. **DE (B2).** `DE-Constraints` must carry `halt_in` and treat `HaltState` as a
   **hard** constraint; `DE-Actuator` must carry `cancel_cmd_in` and remain the
   sole venue writer. Both are in contracts v22 already; this plan does not
   widen them.
3. **Coordinator.** Whether the §6.3 invariant becomes a **mandatory reporting
   field** (derivation lag) — already asked in `ops/STALLED_LANE_POSITION.md`,
   repeated here because it is the same invariant seen from the plane side.

## 8a. R-6 parameter character, applied to this plane

Ruling **R-6** (`COORDINATION.md` §4a, `SP_PLANE_PLAN.md` §5a) bounds the
coordinator's `SP-Params` authority by **character**: A configuration (free),
B load-bearing (change forces a re-run), C measured (adopted, never chosen),
D frozen verdict bar (never moved after the measurement). Applied here:

| OP parameter | class | who sets it | note |
|---|---|---|---|
| `tau_operative` | **C — measured** | **OPS publishes; coordinator adopts** | the ack-latency upper bound is the one Class-C value this plane produces |
| `tau_ladder` interior rungs `{0, 50, 100, 500}` | **A** | coordinator | grid resolution; adding or removing one of *these* changes precision, not a verdict |
| **`tau_decision_rung` = 250 ms** | **D — VERDICT, frozen by R-1** | frozen; nobody may move it | **MF8.** Revision 1 tabled it as an interior Class-A rung saying "no verdict turns on them" — false: the three-way `ww_v1` verdict is computed on `R(τ=250 ms)`. `SP` §4 records the identical error and its correction |
| `tau_ladder` **top rung (1000 ms)** | **D — FROZEN, Ruling R-8** | frozen; nobody may move it | falsifier threshold, not a grid point — see §5.2 |
| expected-port registry (which ports must be alive) | **A** | coordinator | a configuration fact: "which modules must run" |
| per-port `period` | **C — measured** | OPS publishes | the producer's actual cadence is observed, not chosen |
| per-port `staleness_deadline` | **A**, on a **C** base | coordinator | see the split below |
| `LANE_PROGRESS` grace window (3 h) | **A** *while nothing gates on it* | coordinator | becomes **D** if it is ever read as an admissibility threshold (§8c) |
| halt `reset_authority` | **B — load-bearing** | coordinator | changing who may clear a latch changes what every past halt meant |

**The split that keeps monitoring bars honest: the cadence is measured, the
multiple is chosen.** A staleness deadline is not one parameter but two. The
observed cadence is a Class-C fact — `collector.log` writes every ~50 s,
`markets.jsonl` every ~140 s against a 300 s window lattice — and OPS publishes
it. The **multiple** applied to it (the built bars are 600 s and 900 s, i.e.
~12× and ~6× observed) is a Class-A choice about how much quiet to tolerate
before alerting. Conflating them is how a bar gets "tuned" until it stops
firing: widen the multiple far enough and the monitor is off without anyone
recording that it was turned off. Stated separately, that move requires changing
a number that is visibly a choice.

## 8b. HISTORICAL — how the τ top rung came to be frozen (SF13)

**This section is a record, not a live argument.** R-8 froze the rung; §5.2 is
the operative statement. It is kept because the *reasoning* is the precedent for
finding a falsifier-bearing row misclassified, and retained in the past tense so
no reader can cite it as an open question and relitigate a Class-D freeze.

**Raised by OPS because it lands on an OPS-owned measurement, and the window to
fix it for free is open right now.**

Two documents currently disagree:

- `SP_PLANE_PLAN.md` §4 classifies `tau_ladder` rungs as **CHOSEN (the grid)**,
  coordinator-owned — and R-6 Class A is changeable **freely, at any time**.
- `DE_MODULE_PLAN.md` §7.3 makes the top rung a **falsifier**: *if the measured
  bound exceeds the ladder's top rung (1000 ms), the cancellation lever is dead
  at deployment **regardless of replay results***.

A threshold that can kill a whole DE lever is not a grid point. Under the Class-A
reading, the sequence R-6 exists to prevent is available and requires no bad
faith at all: OPS measures an ack bound of, say, 1,500 ms; the lever is dead; the
ladder is "configuration"; someone extends it to 2,000 ms; the lever lives.
**That is a refutation converted into a pass by fiat**, executed entirely inside
a row currently marked freely changeable.

**OPS position: split the row.** Interior rungs stay Class A — they set
resolution. The **top rung is Class D** and should be frozen now, with its
`SP-Params` record naming what it decides. Note the ladder already contains a
member treated as a bar: R-1 froze the `τ = 250 ms` decision rung for `ww_v1`,
so "some rungs are bars" is established practice, not a new category.

**Why now, and why it is free.** A Class-D value may be set only **before** its
measurement runs. `tau_operative` is measured at **deployment**, from
`DE-Actuator` ack latency; `DE-Actuator` is unbuilt and venue access is out of
scope for this programme. So the measurement **cannot** have run, no verdict
depends on the bar yet, and freezing it costs nothing and invalidates nothing.
That is exactly the condition R-6 requires, and it will not be true later.

**This is a proposal, not a decision.** Classification of an `SP-Params` row is
coordinator-owned; OPS raises the conflict and its consequence.

## 8c. What OPS commits to under the standing instruction

1. **`tau_operative` is published, never selected.** OPS reports the measured
   **upper bound** with provenance — it is an upper bound because venue acks are
   not observed and confirmation is by open-order reconciliation (§5). The
   operative rung then follows mechanically: smallest rung ≥ bound, and the
   **coarser** rung whenever reconciliation cannot separate two adjacent ones.
   OPS will not choose a rung, and specifically will not choose one because it
   keeps the cancellation lever alive.
2. **If asked to move a Class-C or Class-D value after a result is visible, OPS
   refuses and records the refusal in `COORDINATION.md`.** Including — and this
   is the case most likely to arise — a request to extend the τ ladder after the
   ack bound is known.
3. **Sweeps report ranges, not best points.** No OPS parameter currently feeds a
   verdict, so this binds prospectively rather than retroactively.
4. **No OPS artifact is invalidated by R-6.** Nothing in `pm_lane_health.py` or
   the units reads an `SP-Params` value; lane supervision is independent of the
   trading configuration. The freshness bars were set from measured cadence
   before any verdict existed and no verdict is conditioned on them.

## 8d. R-8 generalisation applied — falsifier sweep of the whole SP §4 register

**AS-OF: `SP_PLANE_PLAN.md` Revision 7 (MF11).** The sweep below was first run
against Revision 3 and is re-stated here against Rev 7 as it stands on disk. Any
reader must check this stamp before relying on a class here — the register moved
four revisions in one day.

**What Rev 7 changed, and two of them adopt this sweep's findings:**
`quote_size_pin` is now **Class D — VERDICT** (finding 2, adopted);
`verdict_coins` is **ESCALATED** as a *quantifier domain, not a value*, with
membership frozen in `FLOW_MODEL_PROTOCOL_V4` rather than R-DUAL-governed —
stronger than the Class-D I proposed, and it supersedes finding 1; `refuse_k` is
now **Class D — GUARD** (R-20), which this sweep **missed entirely** and wrongly
left in its Class-A list; `tau_decision_rung` is its own **Class D — VERDICT**
row; and a new `min_size` row records `orderMinSize = 5` MEASURED on 7,771/7,815
rows, with the load-bearing consequence that **`min_size` EQUALS
`quote_size_pin`, so the pin has zero downward headroom** — it can be raised,
never lowered.

R-8 generalised the τ finding: **any row in `SP_PLANE_PLAN.md` §4 on which a
documented falsifier turns is Class D for that value, whatever its nominal
class.** OPS swept every §4 row against every "what would falsify this" section
in the programme (DE module §7, DE placement §9, DA inventory §7, EV-Gates §7,
EV-Replay §7, SP §8, `CANCEL_POLICY_PROTOCOL` §1.4). Test applied to each row:
**could moving this value convert a refutation into a pass?**

### Finding 1 — `verdict_coins`, and a result is ALREADY VISIBLE

| | |
|---|---|
| nominal class | **C — MEASURED** (BE publishes; btc, eth by the R-DUAL micro-share rule) |
| falsifier that turns on it | **R-1's frozen branch rule**: *"family DEAD only if **both verdict coins** are DEAD"* |
| status | **`ww_v1` has FIRED — DEAD on both verdict coins** (DE placement §9.1) |

The verdict of the entire cancellation family is a function of **which coins are
in the set**. Add a third coin that is not DEAD, or drop `eth`, and a DEAD
verdict stops being DEAD — without touching a single measurement. `SP` §8 even
names this as an expected revision: *"`verdict_coins` if a thin coin's micro
share drops below the R-DUAL bar."*

**That revision is now constrained.** The result is visible, so under R-6/R-8
and the standing instruction the membership may not move in a way that changes
the `ww_v1` verdict. If the R-DUAL micro share genuinely moves a coin across the
bar, that is new information about the *coin*, not about the result — and it
requires the full Class-D amendment: made before the re-run, stating why the
information is not the result itself, and **explicitly invalidating every verdict
computed under the old set**. This is the row where the pressure will actually
come, because the honest reason to revise it and the convenient one look
identical from outside.

### Finding 2 — `quote_size_pin`

| | |
|---|---|
| nominal class | **B — load-bearing choice** (pinned to 5 shares) |
| falsifier that turns on it | DE module §7.1: venue **min size above 5 shares** contradicts the menu and reshapes `ActionSpace`/the Allocator |

Moving the pin to 10 after learning the venue floor is 8 makes the falsifier stop
firing. That is the R-8 sequence exactly. It is already Class B — a change forces
a re-run — and `SP` §6 already states every measured number in the programme is
conditional on it. R-8 adds the missing half: it is **Class D for that value**,
so the change must also be made *before* the re-run and must explicitly
invalidate what was computed under the old pin, rather than merely triggering
one. Note `ww_v1`'s DEAD verdict is itself conditioned on the 5-share pin.

### Finding 3 — the generalisation is well-formed for CHOSEN rows and ambiguous for MEASURED ones

Raised because R-8 says *"whatever its nominal class"*, and applied literally to
a Class-C row it does not have a meaning.

**You cannot freeze a measurement — you re-measure it.** `r_terminal` carries a
documented falsifier (DE placement §9.4: *"`r ≈ 60` is an artefact of the
confounded 60 s structure"*), as do `tau_operative` and `verdict_coins`. Declaring
such a row "Class D" reads as either *never re-measure it* — which is wrong, and
would freeze an error in place — or as nothing at all.

**OPS proposal: for a falsifier-bearing Class-C row, the frozen object is not the
value.** It is the pair:

1. the **comparison bar** the measurement is judged against (already Class D on
   its own terms — the 1000 ms rung is the worked example), and
2. the **measurement protocol and its re-measurement trigger**, frozen before the
   measurement runs.

Then a re-measurement is legitimate whenever the trigger fires, and any
re-measurement **after a verdict is visible** carries the same three-part
Class-D obligation, with (c) — explicit invalidation of verdicts computed under
the old value — doing the real work. Without this, a Class-C row is protected
against the coordinator *choosing* it and unprotected against it being
*re-measured into a different answer* once the answer is known.

### Rows that survive as Class A — the falsifier on them is a sweep, not a bar

`kappa_usd`, `capital_budget`, `ScenarioLossLimit`, `gamma_ladder`. **`refuse_k`
is STRUCK from this list** — Rev 7 makes it Class D — GUARD under R-20, and this
sweep missed it.
`SP` §8's falsifier — *"the replay defaults turn out to select the answer"* — is
a **robustness statement**, not a threshold: no verdict flips at a named value of
`κ_$`. That is precisely the legitimate tuning R-6 §5a describes, and it is why
sweeping these produces "the verdict holds across this range" rather than a bar
that could be moved.

### One interaction worth recording before someone misreads it later

`ww_v1` fired **DEAD on both verdict coins**, so the cancellation family is dead
**on this tape**. The 1000 ms top rung frozen by R-8 therefore now sits in front
of a lever a *different* falsifier has already killed. The freeze remains correct
and still costs nothing — it protects any future re-sample. But **a favourable τ
measurement cannot revive the family**: only a re-sample under R-ADMISS can
reopen `ww_v1`. **Its scope is DEAD on 8 of 8 coin-days across four era days
(R-9), not "one UTC day" (MF3)** — Revision 2 corrected that at §5.2 and left it
stale here, in the very paragraph warning against misreading, where it materially
UNDERSTATED the closure's strength.

## 8e. NEVER-ATTEMPTED AUDIT — what this plan never proposed

**R-45's method lesson: the review loops audit WHAT EXISTS; nobody audits WHAT
WAS NEVER ATTEMPTED, and untested is invisible to a review of the tested.** DE's
cross-review found 11 MUST-FIX in what this plan *said*. None of them could have
found a subject the plan never raised. Applied to the OP plane:

| never proposed | why it matters | status |
|---|---|---|
| **disk headroom** | the ONE irrecoverable failure here. A batch that dies re-runs; a collector that cannot write loses venue tape for ever. `TAPE_FRESH` caught "stopped writing" and nothing caught "about to be unable to write" — the same outcome with no warning | **CLOSED** — `DISK_HEADROOM`, runway derived from the measured growth rate (4.35 GB/day over 5 raw days, 1,170 GB free, 269 days) |
| **clock health** | §2.1 *names* the monitor's clock as a `HealthEvent` source and **no revision ever implemented a check** — a plan commitment with zero code, the same shape BE found in `EV_GATES_PLAN`. Worse than liveness: every row is stamped at knowledge time, so an unsynchronised clock does not announce itself, it silently mis-stamps the tape | **CLOSED** — `CLOCK_SYNC` |
| **graceful degradation** | the plane treats collectors only as things to *protect*. Never proposed: shedding coins under pressure rather than dying, so a resource squeeze costs some tape instead of all of it | OPEN |
| **capacity pre-flight** | *(R-65 note: its landing-evidence claim was `grep -c VERDICT` — a FORM check that would have passed on output reading "VERDICT: banana". Replaced by `--selftest`, which drives the verdict LOGIC. And the selftest found the `MARGINAL` censored-refusal branch is **UNREACHABLE** with the real references, since the censored reference already sits AT the cap — a branch that cannot fire.)* the exponent work measures peaks *after* a run. Never proposed: predicting whether the NEXT day fits the envelope before starting, which would convert a cap kill into a refusal-before-start and save the wasted hours | **CLOSED** — `ops/capacity_preflight.py`. Weak by construction (two reference points, one censored at `memory.max`) and it says so: three-way verdict, and it **refuses `FITS`** on any upward extrapolation from the censored reference |
| **the collectors' own envelope** | batch caps are set against 30 GB total, but **nobody has ever measured what the collectors need**, so the headroom is assumed rather than known | OPEN |
| **restart drill** | `Restart=always` was verified once, 2026-08-21. Never re-verified, never periodic — a guard demonstrated once and then trusted | OPEN |

**The pattern in the two closed rows is worth naming**: both were failure modes
whose *symptom* was already monitored while their *onset* was not. Reviewing what
the plan said could not surface them, because the plan was internally consistent
about the symptom.

## 9. Falsifiers for this plane

1. **Ack bound > 1000 ms at deployment** ⇒ the DE cancellation lever is dead
   regardless of replay (§5.1). Owned here.
2. **A registered port cannot sustain its declared period** ⇒ the staleness
   deadline is mis-set, and a monitor that alerts constantly is a monitor that
   gets muted. Deadlines must be set from measured cadence, not taste — the
   built checker's bars already are (600 s / 900 s against ~50 s / ~140 s
   observed).
3. **Any proposal to route an EV output into OP** ⇒ §1 violated; refuse and
   escalate. This is written as a falsifier because it will be proposed, and it
   will sound reasonable when it is.

**Falsifiers for THIS PLAN's own claims (SF16).** Revision 1's falsifiers tested
the world, not the document, and none of its five self-flagged weak points
carried one — a gate that cannot fire.

| claim | what would falsify it | status |
|---|---|---|
| §5's four legs are the right cut | any real cancel path decomposing into a leg that is none of observe/decide/transmit/confirm, or two legs that cannot be measured apart | **untested — no venue access** |
| §2 "health is declared, never inferred" is buildable | the first module to publish a telemetry port cannot express its health without the monitor inferring something | **untestable today: zero modules publish** |
| §7.1's "analogue" mapping is a mapping and not a category error | a built check whose OP counterpart would have to consume a type the check cannot see — `TIER1_LOCK` is the live candidate, with no OP analogue at all | **open** |
| §3's ladder needs `DEGRADED` | no `HealthEvent` sequence reaches `DEGRADED` without immediately qualifying for `HALTED`, making the rung decorative | **testable on §3.1's table, not yet run** |
| the on-box monitoring gap is closed | health timer stopped while batch units exit green ⇒ silence (MF7) | **ALREADY FALSIFIED — §7.2(3)** |

## 10. Open, and not settled here

- Whether the research lanes should publish real telemetry ports (§7.1) — no
  demand yet; would be the only way `pm_lane_health.py` becomes `OP-Monitor`
  rather than remaining its analogue.
- **FIRST ITEM, promised by §7.2(3) and missing until Revision 3 (MF1):**
  **run the composite silent-stall test.** Stop the health timer, let the batch
  units exit green on `IDLE`, and confirm what is reported. Expected: **nothing**
  — `OnFailure` never fires because nothing failed, and `MONITOR_LIVENESS` never
  runs to notice its own absence. Testable today; still untested.
- Off-box observation (§7.2 item 3 residue). **The on-box half is NOT closed** —
  Revision 2 retracted that at §7.2(3) and this section contradicted the
  retraction for a whole revision. What IS closed is the *stopped-timer* half
  (`MONITOR_LIVENESS`) and the *failing-checker* half (§7.1's `OnFailure` +
  self-watch); what is open is the **composite** above. Total-stop detection
  needs an observer that is not this host, and therefore needs item 4 first.
- The out-of-band channel (§7.2 item 4) — user decision.

## 13. R-62 rule 3 applied RETROACTIVELY to my own applied findings

R-62 requires every MUST-FIX to name **the decision it changes and whose**. SP's
iteration 12 graded **8 of 11 down** under that test. Applying it backwards to the
14 MUST-FIX already applied here:

| finding | decision it changes | whose | grade |
|---|---|---|---|
| I-1 `R-HALT` unevaluable | R-HALT may not be cited as active protection; a contract row is added | DE, coordinator | **MUST-FIX** |
| I-4 telemetry premise false | those modules are `UNKNOWN` ⇒ `HALTED`; a contract dependency is raised | coordinator | **MUST-FIX** |
| I-7 "on-box half closed" false | an OPS operational claim was withdrawn; a test item created | OPS | **MUST-FIX** |
| I-8 τ=250 Class A → Class D | whether the rung may be moved | coordinator | **MUST-FIX** |
| I-11 §8d stale vs SP Rev 7 | `refuse_k`'s class; two sweep findings superseded | coordinator | **MUST-FIX** |
| II-1 §10 contradicted §7.2(3) | whether the monitoring gap is closed | OPS | **MUST-FIX** |
| II-2 §5.2 re-planted MF8 | same as I-8, at a second site | coordinator | **MUST-FIX** |
| I-2 issuance not bound to HALTED | a rule for an **unbuilt** Actuator | DE (future) | **future-decision** |
| I-5 `STALE` has no consequence | a rule for an **unbuilt** monitor | OPS (future) | **future-decision** |
| I-6 closed-world registry | a rule for an **unbuilt** registry | OPS (future) | **future-decision** |
| I-9 no transition function | a rule for an **unbuilt** ladder | OPS (future) | **future-decision** |
| I-10 reset semantics | a rule for an **unbuilt** reset | OPS (future) | **future-decision** |
| I-3 weak "no new risk" label | nothing today — DE's plan already carries the strong semantics | — | **SHOULD-FIX** |
| II-3 §8d "one UTC day" | nothing today — understates a closure in prose | — | **SHOULD-FIX** |

**7 MUST-FIX · 5 future-decision · 2 SHOULD-FIX.**

**THE UNCOMFORTABLE PART, and it is a property of this plan rather than of the
reviewers.** §0a already marks most of this document **PROSE-ONLY** under
R-36(2) — no mechanism exists for §2, §3, §4 or legs 2–4 of §5. **A finding in a
prose-only section cannot name a decision it changes TODAY**, because nothing in
that section decides anything yet. Five of the fourteen are exactly that: correct,
carefully verified, and binding only on a system nobody has built.

**Consequence for the loop, offered for the lens-set declaration (R-62 rule 1):**
the two halves of this document should not be graded on the same bar.

- **The BUILT surface** (§7.1's supervision, §8a–8d's class assignments, §0a's
  landing evidence) is where findings change decisions today. **Grade at zero.**
- **The PROSE-ONLY surface** (§2, §3, §4, §5 legs 2–4) can only ever produce
  future-decision findings. **Grade on MARGINAL VALUE (R-62 rule 4)** — it is
  converged when a further iteration would not change what gets built when the
  time comes.

Otherwise the loop grinds indefinitely on a specification for an unbuilt runtime,
which is the shape of activity rather than knowledge.

## 12. DE cross-review iteration 2 — one method failure, and one finding declined

**Verdict `DEFECTS_FOUND`: 3 MUST-FIX + 2 SHOULD-FIX, and DE was right that they
are all ONE defect** — *the fix was applied at the finding's named site and the
same defect survived at unnamed sites.* All three verified against the file
before applying:

1. **§10 contradicted §7.2(3)'s own MF7 retraction** for an entire revision —
   §7.2(3) retracted "the on-box half is closed" and promised the test "is now
   the first item of §10"; §10 still asserted it was closed and contained **no
   test at all**. A load-bearing safety claim, internally contradictory, with the
   promised falsifier missing. Fixed at §10, and the composite silent-stall test
   is now its first item.
2. **§5.2 re-planted MF8's misclassification at its own site** — "interior rungs
   `{0,50,100,250,500}` … no verdict turns on them" — while §8a's table two
   sections later had it right. Fixed; `τ=250 ms` is named as the R-1-frozen
   Class-D decision rung at **both** sites.
3. **§8d kept the pre-R-9 "one UTC day" scope**, corrected at §5.2 and missed
   here, in the very paragraph warning against misreading, where it materially
   **understated** the closure. Fixed to DEAD on 8 of 8 coin-days over four era
   days.

**METHOD CHANGE ADOPTED, because the pattern is the finding.** Applying a review
item at its cited line is not applying it. **From Revision 3: every applied
finding is followed by a grep for its CLASS across the whole document**, and the
grep result is what closes the item, not the edit. Revision 3 did that — it found
two residual "one UTC day" hits and confirmed both are the correction quoting the
old text, not survivals. This is R-40's lesson at document scale: a fix bounds
the site, never the defect.

### One SHOULD-FIX DECLINED — superseded between the review and its application

**SF4 says §5.1/§8a/§8d are stale because they describe the τ-selection seam as
unfired, when "R-49 fired it TODAY".** **Declined: R-55 subsequently ruled the
seam has NOT fired.** `tau_operative` is Class C — the **Actuator ack latency
measured at deployment** — and OPS measured `our_feed_lag` and composed it with
an *assumed* ack, so the composite is a **lower bound on achievable τ**, not the
seam's input. R-55 upheld `Q-OPS-10` on exactly this point and recorded that the
seam has not fired and `tau_operative` remains UNMEASURED.

So the plan's existing wording — *"a deployment measurement moves the operative
rung along a pre-built ladder"* — is **correct as written** and applying SF4
would have introduced the error R-55 had just removed. DE could not have known;
the ruling landed after the review. Recorded rather than silently skipped, per
the same discipline that produced MF3's correction in iteration 1.

## 11. DE cross-review iteration 1 — what was applied, and one finding corrected

All 11 MUST-FIX **verified against the artifacts before applying**, not accepted
on report. The four heaviest were checked directly:

- **MF1** — `grep CancelAllStatus contracts.yaml` returns exactly three hits: the
  `R-HALT` rule, the type, and `DE-Actuator.produces`. **Nothing consumes it.**
  Confirmed, and it is the most serious finding in the set.
- **MF8** — Revision 1's §8a listed 250 ms among Class-A "interior rungs" saying
  "no verdict turns on them", while `CANCEL_POLICY_PROTOCOL` §1.4 computes the
  three-way verdict on `R(τ=250 ms)`. Confirmed. `SP` §4 Rev 7 records the
  identical error and its correction, so this plan repeated a mistake another
  plan had already found.
- **MF11** — `SP_PLANE_PLAN` is at **Revision 7**; the sweep was run against
  Rev 3. Confirmed and re-stated with an as-of stamp.
- **MF12** — v22 line 1799 gives `DE-Actuator` a `halt_in` port. Confirmed.

**MF3 does not hold as written, and this is recorded rather than quietly
adjusted.** The finding says the weak "no new risk" label "appears in three
places". It appears in **two** (§1's diagram and §4's table), and the §4
occurrence already carries the strong semantics — "stops NEW risk: the feasible
set collapses". So exactly **one** genuinely weak label existed. The substance is
accepted and fixed — §1's diagram now reads `FeasibleSet = ∅` — but the count is
wrong and a reviewer should know which part of their finding survived.

**SF17 was understated:** §7.1 omitted **two** built checks, not one —
`TIER1_LOCK` and `NO_PROGRESS`. Both are now in the table.

**Not yet applied, and named so they are not lost:** SF18's cross-reference and
section-ordering defects (§8a/§8b/§8d/§8c order on disk, the §8c mis-cite, the
§5.1/§5.2 swap) are cosmetic and deferred to Revision 3; the reviewer NOTES
(observation-time 26 h vs 21 h, the 47 ms best-end-of-band quote, the `ops/`
versus `data/pm_5min/ops/` ledger path, leg-2's unbound measurement source, the
heartbeat-period three-objects question) are accepted as open and itemised in
DE's reviewer outputs.

**Verdict on the review itself: it was sharper than my self-audit.** I flagged
five weak areas; DE found eleven MUST-FIX, and only one of my five (the latching
rules, MF9/MF10) overlapped. The one I was proudest of catching — that the
latching rules were copied rather than reasoned — turned out to be the *shallow*
version of the real defect, which is that `DEGRADED` had no producing rule at all
and so was a state nothing could enter.
