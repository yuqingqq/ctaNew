# Annotation-survival contract (blocker 6)

**Owner of this contract:** DA, as the owner of the annotation it exists to
protect (`phase2_four_arm_v2.json::da_caveat_field`).
**Authorised by:** the user's plan `d506a06` — *"Preserve peer audit annotations
across regeneration by mechanism rather than by hand."* Documentation only.
**Status:** SPEC. BE implements the receipt side. Nothing here is frozen or scored.

## The problem, from two live instances

The generator writes `da_caveat_field` as a `RESERVED` placeholder on **every**
run. It is right to do so: BE must never author, copy or reconstruct another
plane's content. But the consequence is that the annotation vanishes at each
supersession and its owner must notice and re-apply by hand — which happened at
v2.1 (`4c83552`) and again at v2.2 (`8b4bcee`). A mechanism that depends on a
person noticing is a mechanism that fails the first quiet night.

Worse, the annotation is not population-independent. Its magnitudes were
measured on the three-arm dataset (645,851 rows) and do **not** describe the
receipt they now sit in. Re-applying by hand meant re-deciding, each time, which
parts still applied — and the honest fence (`MEASURED_ON_A_DIFFERENT_POPULATION`,
magnitude `UNMEASURED-not-zero`) was authored by judgement, not by rule.

## The contract

**1. A sidecar, owned and committed by the annotator.**
`data/pm_5min/derived/annotations/<receipt>.<field>.json`, written and committed
by the field's owner. The generator never writes it.

**2. Declared schema.** The generator merges by schema, not by trust:
```
{ "owner": "DA",
  "target": {"artifact": "phase2_four_arm_v2.json", "field": "da_caveat_field"},
  "schema_version": "annotation_v1",
  "content": { ... owner's object, opaque to the generator ... },
  "binds_to": { ... see 3 ... },
  "owner_sha256": "<sha256 of content||binds_to, computed by the owner>" }
```
The generator validates `owner`, `schema_version`, `target`, and recomputes
`owner_sha256`. Any mismatch → **refuse the merge and leave the field RESERVED**.
It must never repair, truncate or partially merge.

**3. `binds_to` — the part that carries the lesson.** The annotator declares
what its content's validity depends on. Two kinds:
- `"population_independent": true` — the claim rests on an argument, not a
  measurement (e.g. *every arm is scored on identical rows, so the effect is
  common-mode*). Carries forward unconditionally.
- a **fingerprint** of the population the measurement was taken on, e.g.
  `{"n_rows": {"btc": 311640, "eth": 299703}, "tape_sha256_prefix": "..."}`.

**4. Merge behaviour, which is where a generator usually goes wrong.**
At write time the generator recomputes the fingerprint against the receipt it is
producing:
- fingerprint **matches** → merge `content` verbatim into the field.
- fingerprint **differs** → merge `content` verbatim **and** add
  `"BINDING_STALE": {"declared": ..., "actual": ..., "fields_affected": [...]}`.
  **It does not drop the annotation and does not silently carry it.** Dropping
  loses a caveat someone relied on; silent carry is how one population's
  magnitudes end up describing another's receipt.
- sidecar **absent** → field stays `RESERVED`. Never fabricated.

## Falsifiers BE's implementation must ship (rule 15)

Each must be a behavioural test, not a source-text match:
1. **Absent sidecar** → field is `RESERVED`; no invented content.
2. **Valid sidecar, matching fingerprint** → content merges verbatim; owner's
   bytes unchanged (compare against the sidecar, not against the merged object).
3. **Valid sidecar, DIFFERING fingerprint** → content still present **and**
   `BINDING_STALE` present naming the mismatch. A test asserting only "content
   present" passes both the correct and the silent-carry behaviour and is not a
   test.
4. **Tampered `owner_sha256`** → merge REFUSED, field `RESERVED`, cause logged.
5. **Wrong `owner` or `schema_version`** → REFUSED.
6. **Positive control**: the generator with no sidecar present must still write
   a valid receipt — the merge path must not become load-bearing for the receipt.

## What this contract deliberately does not do

- It does not let the generator author, edit or reconstruct annotator content.
- It does not make the annotation a gate: a stale binding **annotates**, it does
  not refuse the receipt. Whether a stale caveat blocks anything is a policy
  decision for the coordinator, not a property of the merge (rule 14).
- It does not verify the annotation's *claims*. Only its identity, its schema,
  and whether its declared bindings still hold.
