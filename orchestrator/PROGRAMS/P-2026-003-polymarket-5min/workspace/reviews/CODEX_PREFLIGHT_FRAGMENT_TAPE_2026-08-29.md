# Codex preflight review — fragment state tape — 2026-08-29

**Reviewed branch tip:** `a888af1e0f676965dfd31ec06698c085fab5e5e1`

**Artifact inspected:**
`data/pm_5min/derived/be_fragment_state_tape_v1.json`

**Artifact identity:** 861,493,945 bytes; SHA-256
`816be39dccae9527bb1722af4b528a9018b3f950f1802ed735e219716d991b9d`.

## Verdict

**STOP BEFORE GATE OR DIAGNOSTIC SCORE. PRESERVE THE ARTIFACT.** The file is
closed at EOF and the protected live state tape was not replaced, but this
artifact cannot truthfully bind itself to the code that produced it and it
certifies an embargo over an empty training population using non-standard JSON
infinities.

This is a preflight finding, not an Iteration 011 or O1 hold decision. The
fragment line is explicitly `DIAGNOSTIC_NEVER_EVIDENCE`; that limitation does
not make false provenance or a vacuous certificate reproducible.

## What passed

- The artifact ends with the expected `]}`. This specific file is not an
  instance of the truncated-array false-green found in Batch 3.
- It declares 472,413 rows and the ruled split shape:
  `train.slugs=0`, `score.slugs=253`.
- The protected live tape remains 3,170,987,711 bytes with SHA-256
  `c7ab02ebcf27d2fc837c57a9d1a10cca86b331a1aa26a02df51ba835950422d6`,
  matching its established identity.
- The fragment artifact pins the intended ledger as
  `e1dcd4eb8a85a0b5b2f86ed0bf4f5d43ec40bf6b9ced713201b13240e639a2ae`.

## F1 — `builder_ref` does not contain the producing builder

The header declares:

```text
builder_ref = 8a76b271339e9453cd5a0614c7f6e103acf87508
```

That commit changes only `COORDINATION.md`. The builder at that ref hashes:

```text
c3419b0f135e71a68b2a2586368173b81a4ada14eeef81a9025f909f7087f9b5
```

The builder actually present during the run is a dirty, uncommitted file and
hashes:

```text
eb65cf859a78291eb9b937af0ce430292402ebe3bc2e0954e63eb2180a29c9f4
```

It differs from the claimed ref by 29 added and 6 removed lines, including the
path parameters and overwrite guard that made this fragment build possible.
`git diff --exit-code <builder_ref> -- build_state_tape_v2.py` returns 1.

This is not a cosmetic dirty flag: checking out the recorded commit does not
provide a builder capable of reproducing the recorded invocation. It is the
same class the `BUILD_REF` mechanism exists to refuse. Reading a 40-hex string
from the environment proves syntax, not that the executing bytes are at that
ref.

## F2 — empty train produces a vacuous embargo certificate and invalid JSON

The artifact correctly reports the ruled empty train split, then reports:

```text
embargo.state                 = CERTIFIED
embargo.detail.gap_s          = Infinity
embargo.detail.embargo_s      = 60.0
embargo.detail.last_train...  = -Infinity
embargo.detail.first_score... = 1787897340.9817157
```

The builder initializes the train maximum to `-inf` and score minimum to
`inf`, then computes their difference without requiring both split populations
to be non-empty. `Infinity >= 60` therefore certifies the embargo without a
single training row. The total-row guard does not see this because the score
split is populated.

A strict JSON parse of the 3,240-byte header refuses at `Infinity`. Python's
default decoder accepts these implementation-specific constants, which can
make in-repo readers look green while standards-compliant readers reject the
artifact.

R-303 explicitly ruled: if the empty train split trips a legitimate guard,
stop and report the guard rather than papering it. The correct artifact state
is a declared **NOT APPLICABLE / NO TRAIN POPULATION** diagnostic status, not
`CERTIFIED`; whether the downstream gate admits that diagnostic-only status is
a coordinator ruling and must not be inferred from infinity arithmetic.

## Minimum closure before rerun

1. Commit the exact producing builder first. Launch from that commit and bind
   the artifact to it; additionally recording the builder content hash would
   make the claim directly checkable.
2. Make per-split emptiness explicit before embargo arithmetic. Never serialize
   NaN or infinities; use strict JSON (`allow_nan=False`) at the artifact
   boundary.
3. Add a red-first case with non-empty score and empty train. It must not report
   `CERTIFIED`, and its serialized header must pass a strict parser.
4. Apply R-303's promised stop-and-rule at the gate: decide explicitly whether
   an embargo marked not applicable is admissible for this diagnostic-only
   frozen-model run. Do not reinterpret it as a passed fit/score embargo.
5. Rerun to a fresh output path. Do not overwrite either the protected live
   tape or this failed-provenance artifact.

## Executed checks

- artifact `stat`, header and tail inspection;
- full SHA-256 of fragment and protected live tapes;
- strict header parse with non-finite constants refused;
- committed-versus-executing builder SHA-256 comparison;
- `git diff --exit-code` against the artifact's claimed `builder_ref`.
