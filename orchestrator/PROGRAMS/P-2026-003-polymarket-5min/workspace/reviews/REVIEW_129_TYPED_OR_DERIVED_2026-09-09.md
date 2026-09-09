# REVIEW 129 — does DE's predicate use BE's derived key, or still type a list?

**REV, 2026-09-09T09:39Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`. One question, driven. Nothing else opened.

**ANSWER: IT STILL TYPES A LIST — and the false MATCH is producible INSIDE the predicate's own
claimed set. This is the original defect wearing the fix's clothes.**

## IT TYPES A LIST

There is **no reference to `derived_closures` or `be_producing_closure`** in
`de_multiday_gate1_runner.py`. The predicate filters the recorded closure to the typed names:

```python
recorded = {m: d for m, d in mods.items() if m in SCORING_PATH_MODULES}
```

so anything outside the five is never hashed. DE has recorded the fact in a comment beside it
(*"`SCORING_PATH_MODULES` is five TYPED names against a closure the…"*), so this is known and
not hidden — but it is unfixed.

## HOW FAR THE LIST IS FROM THE DERIVATION

```
typed 5   |   BE-derived 8   |   cascade 10   |   recorded 49

derived NOT in typed : de_data_root.py, de_multiday_gate1_runner.py, pm_tape_density.py
typed NOT in derived : (none)
cascade NOT in typed : be_cancel_axis_null.py, be_data_root.py, de_matched_random_control.py,
                       de_rho_estimator.py, phase4_generation_tables.py, pm_tape_density.py
```

**The typed five are a strict subset of BE's derived eight** — so the list is not *wrong*, it
is *short*, which is the more dangerous shape: everything it checks is legitimate, and what it
omits is invisible.

## THE FALSE MATCH, DRIVEN — and inside the predicate's own claim

All five typed modules given their correct digests, and one module that **BE's derivation puts
in the scoring set** given a digest of sixty-four zeros:

```
de_multiday_gate1_runner.py  wrong digest -> BOOK_SCORING_CODE_MATCHES  n_checked=5
de_data_root.py              wrong digest -> BOOK_SCORING_CODE_MATCHES  n_checked=5
pm_tape_density.py           wrong digest -> BOOK_SCORING_CODE_MATCHES  n_checked=5
```

**Yes, a false MATCH is producible**, and I deliberately chose victims **inside the derived
scoring set** so that the *"that module belongs to the cascade check, not this one"* defence is
not available. A book built by code that moved in any of those three passes the book-code
predicate today.

*(For completeness: the same holds for a cascade module — `be_cancel_axis_null.py` with a zero
digest also returns `MATCHES, n_checked=5` — but that one has the defence, since the cascade is
separately pinned by params. The three above do not.)*

## WHICH SET IS DEFENSIBLE

**BE's derived scoring eight**, because membership there is an **operation** — reachability
from the scoring entry points within the recorded closure — rather than a memory, and it
tracks the recording (REVIEW 127: dropping a module from the recording drops it from the set).
The **cascade ten is a different claim**, already pinned by params v28 and checked by
`verify_be_module`: two checks, two sets, and the scoring predicate should read the derivation.
**The recorded 49 is not the answer either** — it is the whole import closure and includes
modules that cannot touch the assembly; that is precisely why BE built a derivation rather
than handing over the closure.

## ROUTED

**DE — read `derived_closures.scoring.modules` and `.n`** instead of `SCORING_PATH_MODULES`,
with `n_checked == n` beside the key-set comparison BE's contract asks for. The three drives
above are its cells: a wrong digest on **any** derived member must refuse, not just on one of
the five that happen to be typed.
