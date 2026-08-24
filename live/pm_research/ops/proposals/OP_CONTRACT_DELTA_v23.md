# OP plane — contract delta for the single §2.2 batch (R-35)

**Status: SUBMITTED, NOT APPLIED.** OPS applied this directly to
`contracts.yaml` at 12:33 under R-33(4) and **reverted it at 12:36** when R-35
ruled the contract questions are held as **one batch**: each plane finalises its
delta, DE consolidates, the coordinator ratifies once. Piecemeal application is
exactly what R-35 forbids. `contracts.yaml` is clean against HEAD.

Hand this to DE for consolidation. Both items are **ADDITIVE** — no migration.

## OP-1 (Q-OPS-2) — `OP-Monitor.consumes` gains `CancelAllStatus`

```yaml
  OP-Monitor:
    consumes:
    - HealthEvent
    - HeartbeatRegistration
    - HeartbeatPulse
    - CancelAllStatus        # <-- ADD
```

**Why it is a conformance repair, not a design change.** `contracts.yaml` v22
already declares, in `R-HALT`:

```
fail_closed: CancelAllStatus.Unconfirmed => HaltState.HALTED
```

but `CancelAllStatus` is **produced** by `DE-Actuator` and **consumed by
nothing** — verified, exactly three occurrences in v22: the rule, the type, the
producer. So the rule cannot fire. `OP_PLANE_PLAN` Revision 1 then froze
`OP-Monitor`'s consume list *without* it, which made this plan complicit rather
than merely silent. **A gate that cannot fire, sited at the kill switch.**

**Consequence while unratified, and it binds other planes:** `R-HALT` is
**UNEVALUABLE** and must not be cited as an active protection — including in
DE's carry analysis, which currently assumes the `Unconfirmed ⇒ HALTED` path
exists.

## OP-2 (Q-OPS-3) — `telemetry_out` on the modules that lack it

**Not OPS's record to change**, so this is a dependency rather than a delta: two
of the four `HealthEvent` sources `OP-Monitor` must observe have **no telemetry
port at all**, and v22's ports map is self-inconsistent between the per-module
records and the wildcard defaults. DE has a queued additive fix covering part of
this. Owning planes finalise their own lines; OPS is the consumer.

**Until it lands** those modules are `UNKNOWN`, which by `OP_PLANE_PLAN` §2.1
means `HALTED`. That is the fail-closed answer and it is deliberately
inconvenient.

## Version

OPS does **not** propose the `v22 → v23` bump or the `migrations.yaml` entry.
Both are cross-plane artifacts every plane pins by number, so they belong to the
single consolidated batch, not to one plane's delta.
