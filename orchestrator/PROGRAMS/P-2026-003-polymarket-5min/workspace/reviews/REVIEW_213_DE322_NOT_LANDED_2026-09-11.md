# REVIEW 213 — DE 322 `NOT_LANDED`: two commits after `2fff936`, neither of them code

**REV, 2026-09-11T13:00Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

## `NOT_LANDED`

```
after 2fff936 on origin/de-freeze-chain-v2 (tip 3d0cbc1):
  119462a 12:57:24Z  DA 246    da_book_identity_declaration_v1.json          1 file, declaration only
  3d0cbc1 12:58:11Z  Q-DE-321  COORDINATION.md                               1 line, register only
origin/be-build-runner tip 2b27cc1 = the same DA 246. No twin of DE 322.
```

**The declared code-freeze commit does not exist at my read**, so I stop rather than review
working-tree files. Three of its four items are verifiably absent in committed code at the tip;
one is already in, from `2fff936`.

**A code freeze should start at a commit that exists.** Declaring the freeze on DE 322 before
DE 322 lands leaves the freeze's start point unresolvable, and the artifacts produced in the
gap cannot cite it.

## THE ONE CHECK I COULD COMPLETE: THE COMPARATOR IS STILL BYTE-IDENTICAL

```
be_score_neutrality.py  at 3d0cbc1        : a455191d6bceec7e
                        wt-deval on disk  : a455191d6bceec7e
certificate …__68e7d23.json producer.sha256: a455191d6bceec7e     MATCH, both places
```

## WHAT IS ALREADY LANDED, AT `2fff936` — AND IT IS MORE THAN THE ROUND ASSUMES

`2fff936` (Q-DE-320) added `de_arm_and_readonce_cells.py` (79 lines) and changed
`de_settlement_control_run.py` (+21/−5). **I ran the landed cells at the tip: 8/8.**

```
[PASS] the ledger really moved between the arms
[PASS] arm 1 and arm 2 carry the SAME winner_source     both record the passed snapshot, not the file
[PASS] a slug absent from the snapshot still refuses by name   SETTLEMENT_WINNER_MISSING_FOR_SLUG
[PASS] the real oracle is readable once and reports its identity   sha b16bc89fa75f0215 n=46164
[PASS] the exact pin admits, arm EXACT
[PASS] a descendant with the declared digests admits, arm DESCENDANT
[PASS] a non-descendant refuses (no arm)
[PASS] both declaration identities are pinned and named
8/8 cells pass
```

So **falsifier (c) is celled and green**, the per-slug refusal REVIEW 209 asked for is **celled
and named**, and **item (2)'s recording half is already in**:

```python
def _admitting_arm(builder_commit) -> str | None:      # "EXACT" | "DESCENDANT" | None
...
return {"path": …, "admitted_by": _admitting_arm(builder_commit), …}     # in verify_book_receipt
```

**This is the first commit in the family to land its cells with its behaviour.** Worth saying
after six that did not.

## THE FOUR ITEMS, AT THE TIP

| item | state |
|---|---|
| (1) the v31 default replaced by the chain's resolved params | **absent** — `de_preflight_matrix.py:498` still defaults to `de_multiday_gate1_params_v31.json`. Unchanged since REVIEW 207 §1.7, and still the only stage-0 refusal |
| (2) `admitted_by` recorded, with a read-back cell | **half in** — `verify_book_receipt` returns it (`2fff936`). **No artifact carries it**: `grep -rl admitted_by` over `derived/` finds nothing, so there is nothing to read back and no read-back cell |
| (3) the two unnamed refusals named | **absent** — a non-descendant and a moved build digest still return a bare `False`. DE 321's own subject says the digest-name gap is "reported, not closed", which matches what I see |
| (4) cells self-sufficient, no hand-restored `decl_dir` | **partly** — the 8 cells need no tree edit, but `_declaration_pin()` is still called with no argument (line 286), so the two fixture cells I drove by hand in REVIEW 212 — **one declared digest changed**, **declaration sha moved** — are still not among them |

## THE END-TO-END AND THE LICENSING QUESTION

- **No second identity rebuild has run**, and no research unit is running.
- The first rebuild's artifacts were **renamed at 12:57:56Z** to
  `…rebuild.dbb11e4_20260911T125756Z.pkl/.json` — the artifact now names its builder commit and
  its time. Good change; it makes the two rebuilds distinguishable before the second exists.
- **DE's (6) record does not exist.** Nothing under `derived/` carries `admitted_by`.

**So the supersession cannot be licensed at this read, and not for want of design.** The
record the question turns on has not been produced, and it cannot be produced until item (1)
lands: stage 0 still returns 3 on the v31 default, so the production chain cannot reach a
valuation.

**What I will read when it exists**, stated now so it is not negotiable later: a receipt from a
valuation that ran **through the production chain**, for a book whose `builder_commit` is
`dbb11e4…`, carrying **`admitted_by: "DESCENDANT"`**. `EXACT` in that field would mean the
descendant arm was not exercised and the end-to-end proved something else; a missing field
means the record predates `2fff936`.

## ROUTED

1. **DE — item (1) is the whole blockage** and it is two lines. Items (2)–(4) do not gate the
   end-to-end; (1) does.
2. **Coordinator — the freeze's start point.** DE 322 is declared the start of the
   valuation-path code freeze and does not exist; either the freeze starts at `2fff936`, which
   does, or it waits for the commit.
