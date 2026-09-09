# REVIEW 127 — BE 119's consumer contract (`6463711`): it verifies

**REV, 2026-09-09T09:10Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. Short round, one commit, three questions driven.

**IT VERIFIES ON ALL THREE, with one nuance on (1) and one gap on (3) that is DE's half, not
BE's.**

**(1) The set is genuinely DERIVED, driven both ways.** Feeding `derive()` the real
49-module recording from `be_daybook_receipt_20260907_btc__L250ms.json` gives
`scoring.n = 8`; **dropping `de_score_stream.py` from the RECORDING gives `n = 7` with that
module absent from the result.** The set tracks the recording, so it is not a second typed
list wearing a derivation's name. **The nuance worth stating:** the *seeds* are typed —
`SCORING_ENTRY_POINTS = (("de_phase4_diag_runner","assemble_streaming"),
("de_phase4_diag_runner","build_tape_index"))` — so membership is *"reachable from these
entry points within the recorded closure"*. That IS an operation (rule 32), and it degrades
the right way: a missed *module* cannot happen, but a missed *entry point* would silently
shrink the set. Worth a line in the contract; not a defect in it.

**(2) The user's case is covered, named as the acceptance criterion, and driven — and BE
found a stronger instance than the user's.** The contract's `EXPECTED_SET_NOT_FULLY_PRESENT`
says in terms *"**THIS IS THE USER'S DEFECT**… the current predicate returns MATCH with
n_checked 4, 3, 2 and 1. A subset must never be accepted as the set"*. And it is not prose
only: the battery drives the membership-varies claim on the real 09-05 receipt pair — **two
runs of the SAME builder recorded 48 and 49 modules, differing by `da_root.py`** — which is
better than the user's case, because it shows a partial expected set arises from **ordinary
run-to-run variation**, not only from a crafted receipt. Closure membership is a property of
the run.

**(3) BE has made refusal POSSIBLE and correct; it has not made it ENFORCED — and that is
DE's half.** The module exports `ClosureRefused` and the contract, and **no `assert_` or
`check_` helper at all**: nothing on BE's surface makes a consumer that ignores
`derived_closures` fail. That is rule 28's shape in the abstract — but BE cannot refuse on
DE's behalf, and the contract does the one thing that makes DE's refusal writable: it ships
the expected set **with its own count**, so `n_checked == n` is answerable without a typed
list, and it says in the same block that a count proves cardinality and not identity, so the
consumer must compare the KEY SET. **The remaining risk is entirely that DE's predicate reads
the key and still accepts a subset — which is the defect under repair, and I will drive it
from the entry point when it lands.**

**One number DE needs:** the derived scoring set is **8**, not the typed 5 — it includes
`de_multiday_gate1_runner.py` and `de_data_root.py`, which the typed list does not. So the
completeness test becomes `n_checked == 8` for a 09-07-shaped recording, and it will differ
per run by construction, which is the whole point.

**Battery: 16 cells, 0 failures.** *My first run reported "15 cells, 7 failures" — that was
my scratch clone with no `data/` mirror, and the failing cells are the ones that read real
receipts. I mirrored and re-ran before reporting; recording it, because reporting another
seat's battery as red when it is my environment is the same class I keep flagging in
instruments.*

**Standing by for DE's repairs.**
