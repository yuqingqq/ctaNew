# CONTRACTS_BATCH_v24 — accumulator (R-35: held as one batch, never ad-hoc)

**Status: DRAFT — ACCUMULATING.** Opened 2026-08-23 to carry v23→v24
changes as they are ruled; nothing here is applied until the batch is
submitted as one §2.2 request and ratified. The R-57 condition discipline
(declared before READY) applies to this batch as precedent.

## §1 — NON-ADDITIVE

| # | change | source | status |
|---|---|---|---|
| M-1 | `CapitalOpCommand.op`: `enum:DEPOSIT\|WITHDRAW` → `enum:MINT\|MERGE` | Ruling R-72 (Q-DE-13): the applied v23 literal was APPLIER-CHOSEN where the ratified v23 batch named no enum, so it has no ratification to defend; MINT/MERGE are the venue's actual operations (mint pairs from $1 / merge pairs to $1) | RULED — record drafted below |

**M-1's migration record, verbatim-ready for `migrations.yaml` at
application:**
```yaml
- from_version: 23
  to_version: 24
  operation: change
  key: field:CapitalOpCommand.op
  old: "enum:DEPOSIT|WITHDRAW"
  new: "enum:MINT|MERGE"
  reason: 'R-72 (Q-DE-13): the v23 literal was applier-chosen where the
    ratified batch named no enum; MINT/MERGE are the venue''s actual
    operations and de_constraints.CAPITAL_OPS vocabulary'
```

**Standing guard until this batch lands (R-72, binding):**
`de_actionspace.py`'s v23-conformance selftest matches NEITHER
`DEPOSIT|WITHDRAW` nor `MINT|MERGE` — it is the only thing holding the
discrepancy visible, and it must NOT be turned green early.

## §2 — ADDITIVE

(none yet)

## §3 — Known candidates NOT yet ruled into this batch

- DA's `Provenance` de-collision naming + authority axis (residue named in
  the v23 type's own notes) — enters when DA names it.
- DA's refinement of the four skeleton-loose SP record types.
- Q-DA-19/Q-DA-20 carriers, if and when ruled.
