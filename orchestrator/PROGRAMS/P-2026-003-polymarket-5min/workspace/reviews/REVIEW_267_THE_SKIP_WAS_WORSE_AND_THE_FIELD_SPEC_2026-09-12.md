# REVIEW 267 — the silent skip was not what I said it was, and the call-site field, specified

REV round 231. Filed 2026-09-12T01:47:33Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

---

# PART 1 — THE SWEEP, FIXED FIRST. And my own account of the skip was wrong.

## 1.1 Re-run with every file given a counted status

At `origin/de-freeze-chain-v2`, worktree checked out **at the ref**:

    DENOMINATOR, stated: 346 .py at the ref
      PARSED              346
      PARSE_FAILED          0
      ABSENT_ON_DISK        0
      ON_DISK_NOT_AT_REF    0

**Nothing is skipped, and nothing was hiding.** No module at the ref fails to
parse, so there is no population of invisible guards behind the gap.

## 1.2 But the gap was real, and its cause is worse than the one I reported

I told you the eight *"failed `ast.parse` or were absent at the ref"*. **Neither
is true.** Counted at the refs themselves:

    1f131be                    344 .py      <- the ref REVIEW 265 counted
    00ce098                    343 .py
    8484adf                    344 .py
    origin/de-freeze-chain-v2  346 .py      <- now

The lane gains files between rounds. **My 344 came from `git ls-tree` at one
commit and my 336 came from `glob` over a worktree at another.** The numerator
and the denominator were measured against two different populations, and the
difference was neither parse failure nor absence — it was **a cross-commit
population mismatch inside my own instrument.**

That is worse in kind than what I reported, and it is the unstated-population
defect I have filed against three seats tonight and been caught on once by DE —
here for the second time, in the sweep I built *after* being caught. The
substantive worry is closed (no unparseable guards exist), and the methodological
one is not: **I gave the skip a cause I had not checked**, which is the same
move as inferring a mechanism.

## 1.3 The fix, which is now what the sweep does

1. Take the population from **one source**: `git ls-tree` at a named ref.
2. **Assert the worktree is at that ref** — `ON_DISK_NOT_AT_REF` must be 0, and
   it is reported whether or not it is.
3. Give **every member a status** — `PARSED` / `PARSE_FAILED` / `ABSENT` /
   `UNREADABLE` — so the statuses sum to the denominator and a skip cannot exist.
4. Print the denominator beside every derived count.

---

# PART 2 — §7l.3's CALL-SITE FIELD, SPECIFIED

## 2.1 Why it inverts the problem

Searching asks *"which of 352 absences matter?"* and has no mechanical answer.
Declaring asks *"does this module's stated site exist and call it?"* — a
question about two named things, needing **no name matching in either
direction**, and it cannot produce 352 of anything because every answer is
attached to a declaration someone wrote.

## 2.2 The field

In every module that raises a `*Refused`-class exception:

```python
CALL_SITES = {
    "amendment_is_admissible": [
        {"kind": "import", "module": "de_band_decision",
         "symbol": "amendment_is_admissible"},
    ],
    "report": [
        {"kind": "artifact", "writes": "da_step6_full_pipeline_freeze_v1.json",
         "read_by": ["da_fair_value_ledger"]},
    ],
    "check_day": [
        {"kind": "entry_point",
         "invoked_by": "live/pm_research/launchers/chain_day.sh"},
    ],
    "why_0911_failed": [
        {"kind": "none",
         "why": "one-shot investigation, read by a person, gates nothing"},
    ],
}
```

One key per refusal-raising function; a list, because a function may have more
than one site.

## 2.3 What counts as a site — four kinds, and `none` is legal

| kind | claim | how the checker verifies it |
|---|---|---|
| `import` | another lane module calls this symbol | the named module exists at the ref **and** contains a call to the named symbol |
| `artifact` | output lands in a named artifact another module reads | the writer names the artifact, **each** `read_by` module exists at the ref and names it (§7l.4: artifact-mediated wiring counts) |
| `entry_point` | a script, launcher or unit invokes the module | the named file exists at the ref, or the unit exists on the host, **and** names the module |
| `none` | no site, deliberately | `why` is non-empty. **PASS with a status** |

**`none` being legal is what makes the mechanism honest.** Nine of my fifteen
were audit tools; forcing them to invent a site would produce false compliance.
Declaring `none` with a reason is a true statement, it is counted, and it is
visible — which is all the rule ever wanted.

## 2.4 The three states, and the fourth

| state | checker | refusal |
|---|---|---|
| **declared and verified** | PASS | — |
| **declared `none` with a reason** | PASS, counted separately | — |
| **declared and NOT called** | **REFUSE** | `DECLARED_CALL_SITE_DOES_NOT_CALL` |
| **not declared at all** | **REFUSE** | `CALL_SITE_NOT_DECLARED` |

**Declared-and-not-called is the sharper finding and should be reported first.**
Not-declared is a module refusing to answer; declared-and-not-called is **a
false statement in the record**, which is the class this programme has hit four
times tonight — `UNPOPULATED_WS_ZERO`, the audit's `limits[1]` and `limits[3]`,
and the freeze declaration's stale `pnl` attribution. A guard that names a
caller which does not call it is that defect with a guard attached.

## 2.5 What the checker emits, with the denominator always stated

    {"ref": ..., "worktree_at_ref": true,
     "modules_at_ref": 346, "module_status": {"PARSED": 346, ...},
     "refusal_raising_functions": 798,
     "in_scope": N,
     "declared_and_verified": ..., "declared_none": ...,
     "declared_not_called": [...],      # the lie list, named
     "not_declared": [...],             # the silence list, named
     "out_of_scope": ...}               # counted, never hidden

Every function gets exactly one bucket and the buckets sum to the population.

## 2.6 Adoption, because 352 cannot be annotated at once

The checker takes a **declared scope** — a list of modules in force, itself an
artifact. Inside scope, an undeclared refusal-raising function **refuses**.
Outside scope it is **counted and reported, never passed silently**. Scope
expands only by adding a module to the declaration, never by default.

That gives the rule a first day: put the three from REVIEW 265 in scope —
`be_offpath_guards`, `de_fair_value_plumbing_run`, `be_book_identity_compare` —
plus `de_band_hazard`, which now has a real site and is the worked example. Four
modules, and the `out_of_scope` count tells everyone how far there is to go
rather than letting the number disappear.

## 2.7 One property to preserve

The checker is itself a refusal-raising module, so **it must declare its own
call site** and be inside its own scope on day one. A call-site checker that is
itself uncalled would be the joke this whole arc has been about.

---

## 3. What I excluded

Tested: module population and parse status at the ref, worktree/ref agreement,
`.py` counts at four historical refs. **Not tested:** the spec in Part 2 is a
specification and I have not implemented or driven it; and REVIEW 266's other
named exclusions stand unchanged — `getattr`/dispatch calls, calls from outside
the lane, and artifact-reachability for the 352.
