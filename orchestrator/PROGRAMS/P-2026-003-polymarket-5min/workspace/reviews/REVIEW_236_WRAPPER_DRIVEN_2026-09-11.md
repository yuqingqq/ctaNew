# REVIEW 236 — five of six driven green at the ref; the sixth is not: **the refusal named `SOURCE_AND_LOCAL_KNOWLEDGE_COLLAPSED` admits a collapsed record**. §5 gate 3 is not satisfied, on one character

**REV, 2026-09-11T19:17Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Everything below is driven at
`origin/de-freeze-chain-v2:live/pm_research/de_fair_price_wrapper.py` (blob `5d8150e87792`,
31,426 B, 670 lines), run from the chain-ref tree.

## THE HEADLINE

```
Stamped(value=…, source_as_of=100.0, local_receipt= 99.0)  -> REFUSED SOURCE_AND_LOCAL_KNOWLEDGE_COLLAPSED
Stamped(value=…, source_as_of=100.0, local_receipt=100.0)  -> ADMITTED   transport_s = 0.0
Stamped(value=…, source_as_of=100.0, local_receipt=100.2)  -> ADMITTED   transport_s = 0.2
```

**The refusal fires on INVERSION and admits COLLAPSE.** Its condition is
`local_receipt < source_as_of`; a record whose two clocks are *the same number* — which is what
a collapsed clock is once someone writes `local_receipt = source_as_of` — passes with
`transport_s = 0.0`.

The class docstring is half right and the half matters: *"A float carries one number and a
caller must remember which clock it came from; the plan names that collapse as a blocker, so
the type makes it impossible to pass one where two are required."* The **type** does prevent
passing one number where two are required, and that is a good design. It does not prevent **the
two being the same number**, and §5 gate 3's requirement is the second thing:
*"source-event and local-knowledge timestamps remain distinct at every hop."*

A genuinely zero transport is not physical for a network feed — the event happens at the venue
and reaches this process later. `transport_s == 0.0` is a collapsed assignment or a clock
without the resolution to tell. Either way it is what the gate forbids. **The fix is one
character** (`<` → `<=`) plus the message, or a separate zero-transport cause if a true zero is
ever legitimate.

*(Method note: my first pass grepped the string literal `"SOURCE_AND_LOCAL_KNOWLEDGE_COLLAPSED"`,
found one hit, and nearly filed "defined but never raised". The **constant** `TIMESTAMPS_COLLAPSED`
has two references — definition and one use. Grep the identifier, not the message.)*

## THE SIX ITEMS, DRIVEN

**24/24 cells pass at the ref, rc 0.** Mapped to the round's six:

| # | item | cells | verdict |
|---|---|---|---|
| 1 | two-regime, both ways | partial **before** `T-60` refuses by name; partial **missing** from `T-60` refuses by the same name; a complete partial inside the terminal window **passes**; the regime is carried explicitly; *"the two regimes are DIFFERENT answers, so the guard is not moot"* (0.600625 vs 0.956385) | **PASS** — both refuse arms *and* both admit arms |
| 2 | source/local-knowledge distinct at every hop | C1 keeps them distinct at the book hop (transport 0.200 s); knowledge that **predates** its own event refuses at the hop | **FAIL** — inversion refuses, **collapse does not** |
| 3 | `X60(t0)` only after local receipt | a future-knowledge reference refuses `REFERENCE_NOT_YET_RECEIVED`, by cause, **with no value** | **PASS** |
| 4 | the estimator never substitutes Identity | *"the refusal is NOT a substitution of Identity"* (estimator stays `bn_bookticker_mid`); the **policy** falls back and **counts** it (`{"REFERENCE_NOT_YET_RECEIVED": 1}`); the share is **computed, not typed** (0.5) | **PASS** |
| 5 | DOWN mechanical, complement / token identity / sign flip | pair sums to **1.000000000000**, dev 0.00e+00; token-identity accepts the real pair; a **sign flip is detected** (sum 1.201250 vs tolerance 0.02); complementing a DOWN record refuses — one side is the source | **PASS**, all three falsifiers present and firing |
| 6 | C1's population identical to Identity's | C1 is a valid FairPrice on Identity's own admissibility; an inadmissible book gives C1 and Identity the **same status**; and `C1_ADMISSIBILITY_DIVERGED_FROM_IDENTITY` refuses if it ever diverges | **PASS** — and the divergence guard answers REVIEW 229 §1's qualification |

**Every cause constant is used, not merely defined** — I counted references per identifier:
`OK` 8, `REFERENCE_NOT_YET_RECEIVED` 5, `REGIME_CONTRADICTED` 6, `OUTCOME_MISMATCH` 4,
`PRE_ERA_EVENT` 3, `INPUT_STALE` 3, `ADMISSIBILITY_DIVERGED` 3, and `INPUT_MISSING`,
`INPUT_MALFORMED`, `ESTIMATOR_REFUSED`, `TIMESTAMPS_COLLAPSED` at 2 each (definition + one use).
No decorative names. Three of those single uses have no visible cell of their own —
`INPUT_MALFORMED`, `ESTIMATOR_REFUSED`, and the `TIMESTAMPS_COLLAPSED` arm that would catch
`==`.

## IS §5 GATE 3 SATISFIED? — **NO**

Gate 3 is *"a valid `FairPrice` for C1/C2; source-event and local-knowledge timestamps remain
distinct at every hop."* The first clause holds and is well built. **The second clause is the
one condition that is not enforced**, and it is the clause the gate exists for. One character
and a cell, and gate 3 is done — everything else about this module is ready.

## WHICH GATES REMAIN

Probed at the refs, by protocol constant, with counts rather than exit codes:

| §5 gate | at the ref | state |
|---|---|---|
| 1 settlement verifier | `FAIR_VALUE_GATE1` — 1 file | **landed**; admissibility, lane-hold and all three plan falsifiers driven green (REVIEW 232), day-slice cells green (REVIEW 235) |
| 2 sigma producer | **`be_sigma_30m.py` is ABSENT from all three refs**; commit `8fe2a2e` is on **0 refs** | **NOT LANDED** — verified 27/27 at the blob (REVIEW 231) and still not on a ref, two rounds later |
| 3 estimator wrapper | `P003_DE_FAIR_PRICE_WRAPPER_V1` — 1 file | **landed, not satisfied** (above) |
| 4 canonical forecast-action builder | `forecast_action` — **0 files on every ref** | **absent** |
| 5 policy seam | `quote anchor` / `policy seam` — **0 files on every ref** | **absent** |
| 6 replay seam | 2 files match `replay seam|baseline and challenger` on every ref | **partial at most** — matched by prose, not by a protocol constant; I did not verify it implements gate 6 |

**So: one gate satisfied (1), one landed-but-failing (3), one built-but-unlanded (2), two absent
(4, 5), one unverified (6).** Gate 4 is the one I would build next regardless of order — it
defines the unit every later score is computed on (`(coin, slug, generation_id,
decision_recv_ns)`, duplicate keys refuse), and gates 5 and 6 both consume it.

## SCOPE

Closed over: the wrapper's 24 cells run at the chain-ref tree and mapped to the six items;
`Stamped.__post_init__` read to the line and driven on all three timestamp orderings; every
cause constant counted by identifier; the six gates probed at all three refs by protocol
constant with visible counts. **Not closed over:** the wrapper's arithmetic; the three causes
with no visible cell; gate 6's two prose matches, which I did not open.

## ROUTED

1. **DE — the collapse refusal admits a collapsed record** (§headline). One character, one
   cell, and gate 3 closes.
2. **BE — the sigma producer is still on no ref** (gate 2), two rounds after REVIEW 231 §0.
3. **Coordinator — gates 4 and 5 are absent** and gate 6 is unverified; gate 4 defines the unit
   the other two consume.
