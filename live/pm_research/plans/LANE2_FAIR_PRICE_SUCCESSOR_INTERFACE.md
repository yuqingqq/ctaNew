# Lane 2 — fair-price successor interface (SPEC ONLY)

**Authorised by** the user's plan `d506a06`: *"write the successor timestamped
interface, preserve `Identity`, and test only the predeclared challengers"* —
**documentation only, nothing frozen or scored.** No scoring is authorised by
this document and none is implied by it.
**Status:** SPEC. **Owner:** DA. **Consumers:** BE (estimation), DE (policy).

## 1. What the interface is for

The fair-price module estimates the **unconditional** object `E[Y | state]`.
Everything downstream anchors to it. Today the only supported output is
`Identity` — the executable PM price itself — and `Identity` remains the
**mandatory baseline**: the full policy must run with it, with no challenger
present, forever. A challenger is an option, never a dependency.

## 2. The record

Every fair-price output, `Identity` included, is one record:

```
FairPrice = {
  value:                    float   # the point-in-time estimate
  source_timestamp:         float   # when the WORLD produced the input
  local_knowledge_timestamp:float   # when WE could first have known it
  freshness_s:              float   # local_knowledge_timestamp - source_timestamp
  book_admissible:          bool    # was the book in a state this may be read from
  book_inadmissible_reason: str|None
  estimator:                str     # "Identity" | a predeclared challenger id
}
```

**The two timestamps are the point of the interface, and they are separate on
purpose.** `source_timestamp` is when the world produced the input;
`local_knowledge_timestamp` is the earliest instant a live policy could have
acted on it. A single timestamp cannot distinguish them, and the gap between
them is exactly where look-ahead enters. This programme has already paid for
that class: a resync clock cost 22–162 ms of label error, and the era boundary
exists because rows stamped post-parse carried up to ~0.6 s of backlog error
concentrated in bursts — precisely when a fair price matters most.

**Consumption rule, mechanical, not advisory:** a decision at time `t` may read
a `FairPrice` only if `local_knowledge_timestamp <= t`. This is the same
estimand discipline as rule 7's latency (value only tranches after `t + L`),
applied to the input side. **A challenger that cannot produce both timestamps
is INADMISSIBLE — not degraded, not defaulted.** Absence of a timestamp must
never be read as zero freshness.

## 3. Ownership fence — the no-double-counting rule

- **Fair price owns** the unconditional `E[Y | state]`.
- **Toxicity owns** the fill-conditional residual relative to that anchor:
  `E[Y | state, FILLED] - E[Y | state]`.

**Toxicity must never re-estimate the unconditional term**, or adverse
selection is counted in both and every downstream comparison inherits the
double count. The fence is stated as a checkable predicate rather than a
convention: **the toxicity estimator's feature set must not contain the
fair-price value, and its target must be the residual, not the level.** A
mechanical check on the fitted artifact is required — a rule that lives only in
prose has been violated in this programme before, and the violation was
invisible in every per-arm number.

## 4. Challengers — predeclared, closed set, timestamped

At most **two**: PM microprice, and **at most one** cross-venue forecast.
Declared BEFORE any comparison, each carrying:

```
ChallengerDeclaration = {
  id, estimator_ref (commit), declared_utc,
  inputs: [...], point_in_time_rule: "how local_knowledge_timestamp is derived",
  admissibility: "when it must abstain",
  success_criterion: "declared BEFORE the comparison"
}
```

**A failed challenger does not block integration.** The policy runs with
`Identity`. That asymmetry is deliberate: it removes the incentive to keep
looking until a challenger passes.

## 5. What this document does NOT authorise

- **No scoring, no fitting, no comparison.** Spec only.
- **No new forward clock.** Any challenger evaluated later starts its own
  ≥5 complete-UTC-day clock on days not yet consumed. Consumed days stay
  consumed (08-20..27 for the harmful-fill line; 08-27 excluded outright).
- **No promotion path.** Whether a passing challenger is adopted is a policy
  decision with its own priced trade-offs (rule 14).

## 6. Falsifiers required of any implementation (rule 15)

1. A record missing `local_knowledge_timestamp` is **REFUSED**, not defaulted.
2. A decision at `t` reading a record with `local_knowledge_timestamp > t` is
   **REFUSED** — with a positive control that a legitimately-fresh read passes,
   or the check is untestable in the accepting direction.
3. `Identity` with **no challenger present** produces a complete, valid record —
   the baseline path must not depend on the challenger machinery existing.
4. A toxicity feature set containing the fair-price value is **REFUSED** (§3),
   with a known-bad fixture that must fire.
5. A challenger declared AFTER its comparison is **REFUSED** on `declared_utc`.
