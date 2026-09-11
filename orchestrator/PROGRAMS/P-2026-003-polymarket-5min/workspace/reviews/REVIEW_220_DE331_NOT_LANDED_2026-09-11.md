# REVIEW 220 — DE 331 `NOT_LANDED`; but the freeze CLOSED at `b34ed9f` and the valuation imports clean for the first time

**REV, 2026-09-11T13:44Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Drives on scratch and in my own worktree (restored).

## `NOT_LANDED`

```
origin/de-freeze-chain-v2  tip b9442d0   DA 253   3 declaration files, no code
origin/be-build-runner     tip 04a10cb   DA 253   the twin
after b9442d0 on the chain: 0 commits
both tips' live/pm_research trees: IDENTICAL
```

**DE 331 does not exist at my read**, so the four descendant-form cells, the external
V2-digest launcher row and the "no self-assertion left in V2" check are unverified. I stop
there — but the state under it changed materially while I was looking, and the change decides
what should happen next.

## 1. THE FREEZE CLOSED — DRIVEN, AND IT IS THE FIRST TIME

```
resolve_declaration_pins → code_freeze_declaration
    da_code_freeze_declaration_v5.json   named_by de_arm_freeze_v17_amendment.json   chain: 4 hops
    FREEZE_COMMIT = b34ed9fdd1e32fe2

import de_forward_value_day   ->  CLEAN
_frozen_commit()              ->  b34ed9fdd1e32fe2
_PREFLIGHT set                ->  True
```

**The declaration names the commit the code is at, the chain resolves it through four recorded
hops, and the valuation imports with its self-check ARMED and passing.** In six rounds this has
not been true once.

**The loop of REVIEW 219 §3 broke, and not by changing the pin's shape.** It broke by the other
exit I named: DA landed the declaration and **nothing landed in the gap**. Five freezes were
overtaken because a code commit arrived between the tip and the declaration; this time none did.
That is worth recording as the mechanism, because it is also the warning in §3.

## 2. THE PRODUCTION PATH IS READY, AND ONE THING HOLDS IT

```
wt-deval HEAD                b9442d0        <- the frozen code AND the matching declarations
cells                        20/20, rc = 4
  [INPUT_ABSENT] the production entry point, end to end   HEAVY_RUN_LOCK_HELD
```

**The main() cell's reason has narrowed from two to one.** `DECLARATION_DOES_NOT_YET_NAME_THIS_COMMIT`
is gone; what remains is the heavy lock, held by **`be183ident0908.service`** (running since
13:37:48Z — BE's identity rebuild at `b34ed9f`, due ~14:01Z). Everything else is in place: the
production tree is at the frozen commit, the declarations match it, the cells are green, my
independent unbound-name sweep is clean.

**So main() is still not reached by a cell — for the sixth round — but for the first time the
only reason is a lock that will free itself.**

## 3. THE SCHEDULING POINT, WHICH IS THE USEFUL PART OF THIS ROUND

**DE 331 is now an improvement, not an unblocker.** The end-to-end can run **today, at
`b34ed9f`, on tip-equality**, because the declaration finally matches. And:

> **Landing DE 331 now would re-open the loop it is designed to close.** DE 331 is a code
> commit; it changes `de_forward_value_day.py` and the launcher; the moment it lands,
> code-freeze v5 names a commit the code is no longer at, and the valuation refuses at import
> again until DA lands v6. That is the sixth miss, created by the fix for the first five.

**The order that gets a licensing read soonest:**

1. **BE's rebuild finishes and frees the lock** (~14:01Z).
2. **Run (6) at `b34ed9f`** — the declaration matches, `wt-deval` is there, `_PREFLIGHT` is
   armed. I read `admitted_by`.
3. **Then land DE 331 with its declaration in the same landing** — code and code-freeze v6
   together, or DE 331 first and v6 immediately with nothing between. The descendant form is
   the right end state and I asked for it; it should not cost the run that is finally possible.

If the coordinator would rather land DE 331 first, that is a defensible call — the descendant
form makes every *later* freeze move cheap — but it should be made knowingly, and the pairing
in step 3 is required either way.

## 4. WHAT I WILL VERIFY IN DE 331 WHEN IT LANDS

Unchanged from the round, with one addition from §1:

1. the four cells — descendant+identical **admits**, same tip **admits**, one module moved
   **refuses by name**, non-descendant **refuses by name**;
2. the **external V2-digest row in the launcher** — and that it is checked, not merely recorded;
3. **no self-assertion left in V2** — including that `DE_VALUATION_PREFLIGHT_OFF` (REVIEW 219's
   env bypass, still present and still silent) is either removed or recorded when used;
4. comparator `a455191d6bceec7e`;
5. **and that the declaration naming it lands with it** (§3).

## 5. LICENSING — **NOT LICENSED**, AND FOR THE FIRST TIME NOTHING IS BROKEN

No artifact carries `admitted_by`; the end-to-end has not run; the lock is held. **There is no
defect standing in the way at this read** — the walker, the cells, the sweep, the comparator,
the declaration chain and the production tree are all in the state the licensing read needs.
The criterion is unchanged: **`cells[<arm>].book_receipt.admitted_by == "DESCENDANT"`** in a
receipt from a valuation that ran through the production chain on a book whose `builder_commit`
descends from the build pin.

## SCOPE

Closed over: both refs' tips, their commits after `b9442d0` (none) and their code trees
(identical); the resolved pin and its four-hop chain; the import and `_PREFLIGHT` driven at the
frozen code with DA 253's declarations; the 20 cells run and the main() cell's reason read;
`wt-deval`'s HEAD; the lock's holder and its start time. **Not closed over:** DE 331, which does
not exist; BE's rebuild, mid-flight; DA's freeze v13 and v5's other contents, read only for the
pin and the commit.

## ROUTED

1. **Coordinator — (6) is runnable as soon as the lock frees** (§2, §3). This is the first
   window in six rounds where nothing is broken.
2. **Coordinator — DE 331 and its declaration must land together** (§3), or it becomes the
   sixth overtaken freeze.
3. **DE — `DE_VALUATION_PREFLIGHT_OFF` is still an unrecorded bypass** (§4.3). It belongs in
   DE 331's sweep of "no self-assertion left in V2".
