# REVIEW 235 — I grepped a tree 205 commits stale: the fork detector exists and drives 36/36, reachability is now complete on both days, the wrappers land 24/24, and the undeclared-tree row is not spoofable by an env var

**REV, 2026-09-11T19:12Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. All four items verified **at the refs**.

## 0. MY ERROR, AND IT IS NOT THE ONE I THOUGHT I HAD FIXED

REVIEW 234 §3 reported `SUPERSESSION_CHAIN_FORKED` absent, counted three ways. The counts were
honest; **the subject was wrong**. I grepped `/home/yuqing/ctaNew`, which is **205 commits
behind `origin/mm-research`**. At the refs:

```
origin/de-freeze-chain-v2  live/pm_research/de_day_record.py   PRESENT
origin/be-build-runner     live/pm_research/de_day_record.py   PRESENT
origin/mm-research         live/pm_research/de_day_record.py   PRESENT
```

It is on **all three**, including the shared branch I could have grepped at any time. Last
round I fixed the *form* of the search so its failure would be visible; **I did not fix its
subject.** "Does this exist" has two halves — *in a form whose failure is visible* and *in the
tree that is the artifact* — and I had already written the second half myself, twice
(REVIEW 206, REVIEW 220: the shared tree is not the artifact). **From here every existence
check I run is `git grep <ref> -- <pathspec>`, never a working-tree grep.**

*(The same error would have cost two more this round: `de_fair_price_wrapper.py` and
`EXECUTED_FROM_AN_UNDECLARED_TREE` are both on the chain refs and **not** in the stale shared
tree. Grepping the working tree would have produced two more false `NOT_LANDED`s.)*

## 1. THE FORK DETECTOR — EXISTS, AND THE SCOPING IS EXACTLY REVIEW 233's CONDITION

`de_day_record.py` (1,455 lines, identical on both chain refs; `origin/mm-research` carries an
older blob). Run by me from the chain-ref tree: **36/36 cells pass, rc 0.**

```python
lineage_key(doc) = book_lineage.this_book_sha256   # "two records of one book are one lineage"
# TOTALITY IS PER LINEAGE …
# a fork is TWO RECORDS OF ONE LINEAGE NAMING THE SAME PARENT. Two lineages sharing a
# root is the normal case here -- `_v3` (landed) and `_v2` (freeze-built) both supersede
# the same v1 -- and a detector without this scoping would have refused this family on
# its first run.
```

Both directions are in the cells, and DE added a distinction I did not ask for and would not
have thought of: **a historical fork is REPORTED and must be disclosed by the newest record;
only a fork the newest record is part of REFUSES** — because that one is being created now and
can still be avoided, while rule 13 forbids editing the landed records of an old one. That is
the right asymmetry.

**And the two brittle cells are fixed the right way**: the stage-0 reader now returns a typed
verdict — `structured` / verbatim `log_lines` / `REFUSED NO_STAGE0: a gate whose verdict is
recorded nowhere is …` — where a `KeyError` used to escape.

## 2. REACHABILITY — WALKED BY ME, NOW COMPLETE ON BOTH DAYS

```
2026-09-07   newest _v9   11 day-result files   REACHED all   NOT REACHED: none
             link kinds present: other_books_for_this_day, other_records_for_this_day,
                                 residue_records_for_this_day
2026-09-09   newest _v5    5 day-result files   REACHED all   NOT REACHED: none
             link kinds: none needed -- one lineage
```

**REVIEW 234 §2's 8-of-10 gap is closed**, and by the better of the two available fixes: the
two `…_reproduction_at_the_freeze*` records were never day results — they were **mis-named into
the day-result glob**. A reproduction emitted now is `p003_de_reproduction_at_the_freeze_20260907.json`,
outside the glob, and the two old files are carried as **`residue_records_for_this_day`** and
reported by the cells as `[RESIDUE] 2 record(s) under the old name, not deleted`. Naming fixed
forward, residue disclosed rather than deleted — which is rule 13's shape.

## 3. THE FAIR-PRICE WRAPPERS — 24/24, AND THE TWO-REGIME CONTRACT IS THERE

`de_fair_price_wrapper.py`, on both chain refs, run by me: **24/24, rc 0.** Against
REVIEW 192's contract:

```
[PASS] a MISSING partial after T-60 refuses by the same name  PARTIAL_CONTRADICTS_THE_DECISION_TIME_REGIME
[PASS] the two regimes are DIFFERENT answers, so the guard is not moot  0.600625 vs 0.956385
[PASS] DOWN is the MECHANICAL complement and the pair prices to 1  sum 1.000000000000  dev 0.00e+00
[PASS] the token/outcome identity check accepts the real pair
[PASS] a UP/DOWN SIGN FLIP is DETECTED by the complement check  sum 1.201250 vs tolerance 0.02
[PASS] complementing a DOWN record refuses -- one side is the source
[PASS] knowledge that PREDATES its own event refuses at the hop
[PASS] the POLICY falls back to Identity and COUNTS it  {"REFERENCE_NOT_YET_RECEIVED": 1}
[PASS]   and the fallback share is computed, not typed  0.5
[PASS] C1 REFUSES if its admissibility ever diverges from Identity's  C1_ADMISSIBILITY_DIVERGED_FROM_IDENTITY
```

All three mandated falsifiers are present (complement, token identity, sign flip); the
source-event/local-knowledge distinction is enforced **at the hop**; the policy — not the
estimator — performs the fallback **and counts it**; and the second cell is the one that makes
the regime guard non-vacuous. The C1 divergence guard answers REVIEW 229 §1's qualification
directly.

**Not closed over**: I verified the contract through the cells, not by reading all 24 or the
wrapper's arithmetic. The `partial`-present-**before**-`T-60` direction is not among the ten I
saw printed; the named refusal covers both directions by construction, and I did not confirm
the second arm has its own cell.

## 4. THE UNDECLARED-TREE ROW — **NOT SPOOFABLE BY AN ENV VAR**, AND CELLED BOTH WAYS

```python
def executing_tree_of(module=None, cwd=None) -> Path:
    """The tree a payload WILL import from -- from the artifact, not the env."""
    if module is not None and getattr(module, "__file__", None):
        return Path(module.__file__).resolve().parents[2]
    return Path(cwd if cwd is not None else Path.cwd()).resolve()
```

It reads the **resolved `__file__` of a module the payload actually imported**, falling back to
the **process's own cwd**. Neither is an environment variable, and the cells prove it:

```
[PASS] it reads a MODULE'S RESOLVED __file__, not an env var -> any environment variable
[PASS] ...and an env var claiming the shared tree cannot change the answer
[PASS] the SHARED tree REFUSES by name, saying it is NON-EXECUTING
[PASS] an unrelated tree REFUSES by name
[PASS] the RIGHT tree for the WRONG lane REFUSES
[PASS] a DECLARED executing tree passes and names its lane  (build / valuation / emit)
```

**The reasoning in the comment is the best part**, and it is measured rather than argued:
*"An env var says what someone INTENDED; the resolved `__file__` … and the process's own cwd,
say what it WILL IMPORT"* — with the running 09-10 build as the case in point, whose argv names
the **shared-tree** launcher with a relative payload path while its cwd is `wt-fwd`. **The
launcher path is decoration; the cwd is the fact.**

**The obvious objection, answered:** `PYTHONPATH` *can* move where `__file__` resolves — but if
it does, the payload genuinely will import from there, so the check reports the fact rather
than being deceived by it. The only way to make the row lie is to make the payload actually run
from the tree the row then names.

**One residual, and it is the programme's recurring class:** `EXECUTING_TREES` and
`NON_EXECUTING_TREE` are **hardcoded literals** (lines 262–267) while
`declarations/da_shared_tree_non_executing_v1.json` declares the **same three trees** under
`THE_EXECUTING_TREES`. **Two sources of truth for one fact.** It is fail-closed (a fourth tree
refuses), and the declaration itself honestly records `NOT_PUSHED_TO_ORIGIN_MM_RESEARCH` — but
the code should read the declaration, as hunk C and the freeze pin were both made to do.

## 5. FOUND WHILE RUNNING: THE GUARD REGISTER'S COMPLETENESS JOIN EXISTS, AND IT IS REFUSING

`da_population_freeze_verify.py --falsify` exits **1**, on a real verdict:

```
REFUSED GUARD_REGISTER_INCOMPLETE:de_forward_value_day.py:51:VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT
 -- 102 enumerated refusal site(s) have NO ROW. A register that cannot see a new guard cannot gate a run on it.
```

**That is REVIEW 222 §2.4's routed join, built** — and its first act is to refuse with 102
unregistered sites. The instrument is working; the work it names is outstanding. Worth the
coordinator's attention because the verifier's own falsifier is red until the register catches
up, and a red falsifier is easy to mistake for a broken instrument.

## SCOPE

Closed over: the refusal located at all three refs by `git grep <ref>`; `de_day_record.py`'s
fork logic read and its 36 cells run from the chain-ref tree; both days walked programmatically
from their newest records across all link kinds; the wrapper's 24 cells run; the
executing-tree reader read to the line and its 8 cells run; the declaration compared against
the code's literals. **Not closed over:** the wrapper's arithmetic and the 14 cells I did not
see printed; `de_day_record.py`'s other 20 cells beyond their names and results; whether the
102 unregistered guard sites are real or an enumeration artefact.

## ROUTED

1. **Coordinator — REVIEW 234 §3 is withdrawn** (§0). The detector exists and is correct.
2. **DA — the executing trees are declared twice** (§4), once in code and once in a
   declaration. One should read the other.
3. **DA — the guard register is 102 sites short** (§5), and the verifier's falsifier is red
   until it is not.
4. **Me — every existence check at a ref, never at the working tree** (§0).
