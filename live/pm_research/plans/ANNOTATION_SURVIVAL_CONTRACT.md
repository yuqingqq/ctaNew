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

### 2a. Canonicalization — AMENDMENT, ruled by the contract owner

**The first version of clause 2 said "sha256 of content||binds_to" and pinned no
canonical form.** BE refused to implement on that basis and was right to: if two
correct implementations differ on key order, whitespace, unicode escaping, float
repr, or what `||` concatenates, **every VALID sidecar mismatches on recompute,
and the refusal is indistinguishable from tampering** — the mechanism built to
preserve annotations would drop them while reporting an attack. A signature rule
that does not pin its bytes is not a signature rule.

**RULED — `canonical_form: "annotation_canon_v1"`**, declared as a field in the
sidecar, defined as exactly:

```python
payload = json.dumps({"content": content, "binds_to": binds_to},
                     sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
owner_sha256 = hashlib.sha256(payload).hexdigest()
```

One object (no `||` ambiguity), recursively sorted keys, no insignificant
whitespace, explicit UTF-8. **`allow_nan=False` is not decoration**: the default
emits bare `NaN`/`Infinity`, which is not JSON, so some readers reject the
payload and others parse it differently — a divergence that would surface as a
signature mismatch. NaN/Infinity in a signed payload REFUSES at write time.

**Floats, the residue BE's proposal does not close.** `json.dumps` serialises
floats with Python's shortest round-tripping `repr`. That is stable across
CPython 3.x but is **not** a cross-language guarantee. So: `annotation_canon_v1`
is defined **for CPython 3.12+ on both sides**, and any implementation outside
that must declare a new `canonical_form` rather than reinterpret this one.
Annotators who want to be safe from this entirely may carry numeric magnitudes
as strings; nothing in the contract requires floats.

**Unknown `canonical_form` → REFUSE with a DISTINCT cause** ("unrecognised
canonical form"), never a signature-mismatch error. This is the clause that
keeps *we disagree about the recipe* from being reported as *someone tampered*.

### 2b. Agreement must be PROVEN before first use, not assumed

The merge implementation must ship a **canonicalization agreement test** that
recomputes `owner_sha256` from **the owner's real committed sidecar bytes** and
matches the value the owner recorded. **If that test is absent or failing, the
merge path must refuse to run at all** — not fall back, not warn. Two
implementations that have never been shown to agree on the bytes are not a
signature scheme, and the failure mode this contract exists to prevent is
precisely a silent disagreement that reads as an attack.

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
4a. **Unknown `canonical_form`** → REFUSED with the *unrecognised-form* cause,
   NOT the signature-mismatch cause. A test that accepts either message cannot
   tell a recipe disagreement from an attack, which is the whole point.
4b. **Agreement test on the owner's REAL committed sidecar** → recomputes to the
   recorded `owner_sha256`. Absent or failing ⇒ merge path refuses to run.
4c. **NaN/Infinity in content** → REFUSED at write time (`allow_nan=False`).
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
