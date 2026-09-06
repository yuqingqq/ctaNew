# REVIEW — DE 74's data-root resolver: no second resolver, the shell trap is closed, and `fixture=True` is the one promise left unchecked

**Filed** 2026-09-06T04:47Z (clock read before composing) · reviewer seat
(pm-codex) · tip `fade621` · no code fixed · no write under `data/` · nothing
sealed opened.

**ROUTING — CHECKED**, everything driven by me.

## ONE FINDING

**`fixture=True` short-circuits both checks and nothing verifies the run is a
fixture.** It is a caller-supplied boolean. The mitigation is visibility — the
block records `fixture: true` and `refusal: NOT_APPLICABLE_FIXTURE_RUN` — but a
result-bearing emission that passed `fixture=True` would be admitted off any root
and would say so only in a field nobody is required to read. **The runner already
ships the instrument that would make it a check** (§3).

**Everything else in this batch verifies, and the practical payoff is large.**

---

## 0. The payoff, driven first because it is the point

**My `data/` was a SHELL when I ran this** (`drwxrwxr-x`, holding only `derived/`),
and:

```
de_multiday_design_declaration --selftest   PASS -- 53 checks
de_data_root                   --selftest   PASS -- 12 checks
```

**The trap that bit me three times in three rounds no longer poisons a read.** With
`PM_DATA_ROOT` unset the resolver falls to **branch 3 (`3_canonical`)** because
`<worktree>/data/pm_5min/raw` does not exist in a shell — so the shell *selects
itself out*. With the env set to the repo root it takes branch 1 and resolves the
same place. **Either way the shell is no longer silently readable.**

## 1. The env matrix — driven, four cases

| `PM_DATA_ROOT` | mode | result |
|---|---|---|
| a scratch dir | **real emission** | **REFUSED** |
| a scratch dir | **fixture** | **ADMITTED**, `branch 1_env_PM_DATA_ROOT`, `is_canonical: false`, `refusal: NOT_APPLICABLE_FIXTURE_RUN` |
| `/tmp/nope` | real emission | **REFUSED** |
| **unset** | real emission | **ADMITTED**, `branch 3_canonical`, resolves `/home/yuqing/ctaNew/data` |

**And the arms emitter refuses at IMPORT**, driven both ways:

```
PM_DATA_ROOT=/tmp/nope  -> DataRootRefused at de_data_root.py:90
   "the section-8.1 arms emission is a result-bearing emission and the resolved
    data root is None ... A worktree shell resolves, reads and holds a DIFFERENT
    ledger -- which is why this is a refusal and not a warning."
PM_DATA_ROOT=/home/yuqing/ctaNew  -> imported OK
```

**Refusing at import rather than at emit is the right placement** — a module that
cannot legitimately emit should not be importable into a run that intends to.

**Env unset falls to branch 2/3 with no DE-side rule of its own**: the branch
predicate is `TD.CODE_ROOT / "data" / "pm_5min" / "raw"` — **pm_tape_density's**
condition, evaluated here rather than reimplemented. ✓

## 2. **Is there a second resolver? NO — and I checked every `de_*.py`**

Four modules import `de_data_root`: the resolver itself, the runner, the design
declaration and `de_section81_arms`. Two literal `data/` path expressions survive
elsewhere, and **neither is a resolver**:

* **`de_multiday_design_declaration.py:337–338`** — the `base / "data" / "data"`
  nested-symlink expression is **only in the `else:` branch**, taken when a caller
  passes an explicit `root` argument, and it is labelled
  `branch: "0_explicit_root_argument"`. The `if root is None:` branch calls
  `DR.resolve()`. **And both branches funnel into the same gate three lines later**
  — `if str(resolved) != DR.CANONICAL_DATA_ROOT: raise DesignRefused`. So the test
  hook cannot reach a different ledger than the production path can. **Not a second
  resolver; a parameterised entry with the same refusal.**
* **`de_phase4_diag_runner.py:7603`** — `data = (ROOT / "data").resolve()` is **not
  a data-root resolution at all**. It is the *write-prohibition* guard: it tests
  whether a proposed output path lies under `data/` and refuses, *"this seat is
  READ-ONLY under `data/`."* **Using the LOCAL root there is more conservative than
  the canonical one**, because it also catches a write under the worktree's own
  shell. Correct as it stands.

**So the precedence lives in exactly one place, and it is imported rather than
copied** — `resolution_owner: "pm_tape_density._resolve_data_root (imported, not
reimplemented)"`, with the same branch names.

## 3. **FINDING — does `require_canonical` distinguish "fixture" from "offline" honestly?**

**No. `fixture=True` is a promise, not a check.** Reading
`de_data_root.py:77–104`: the fixture branch returns **before** both the
`is_canonical` test and the `tape_present` test:

```python
if fixture:
    r["refusal"] = "NOT_APPLICABLE_FIXTURE_RUN"
    return r
```

**Nothing verifies that the caller is a fixture run.** A result-bearing emission
that passed `fixture=True` would be admitted against any root — including a shell —
and the only trace would be `"fixture": true` in the emitted block.

**In its favour:** the flag *is* recorded, and I confirmed it reaches the artifact —
fixture v5 carries `fixture: true`, `refusal: "NOT_APPLICABLE_FIXTURE_RUN"`,
`purpose: "the fixture run"`. So a reader who looks can tell. **That is
disclosure, not enforcement**, and this programme's own standing lesson is that a
disclosure in a field nobody resolves is where defects live.

**The fix is cheap and the instrument already exists.** The runner emits
`no_path_under_data_was_opened` and proves it by instrumenting `open`/`read_bytes`/
`read_text`. **Bind the two**: `require_canonical(..., fixture=True)` should refuse
to record `NOT_APPLICABLE_FIXTURE_RUN` unless the data-free instrument reports that
no path under `data/` was opened. That converts the caller's promise into a
measured property, using a probe that already ships. **A fixture run that touched
the ledger is not a fixture run, and today only its author knows.**

## 4. The receipts' root/branch fields — **produced at emit time, not copied from env**

Design v6 (`…043936Z`) and fixture v5 carry the resolver's whole output, not an
echo of the environment:

```
branch                        "1_env_PM_DATA_ROOT"
data_root                     "/home/yuqing/ctaNew/data"
data_root_resolved            "/home/yuqing/ctaNew/data"      <- a .resolve() call
is_canonical                  true
PM_DATA_ROOT_env              "/home/yuqing/ctaNew"
resolution_owner              "pm_tape_density._resolve_data_root (imported, ...)"
PM_DATA_ROOT_names_the_REPO_root_not_the_data_dir   true
```

**`data_root_resolved` is a symlink resolution — the environment cannot supply it**,
and `is_canonical` is a comparison against the canonical constant rather than a
restatement of the input. The env value appears as **one field among seven**, beside
the resolution rather than in place of it. ✓

And design v6 carries the block **twice** — at top level and inside
`R7_the_day_set.root_resolution` — so the day-set derivation names the root it read
at the point of use, not only in a header. ✓

**The `PM_DATA_ROOT_names_the_REPO_root_not_the_data_dir: true` field is the right
response to the round's own error** (the dispatched value was the data dir; DE
measured it as the repo root and corrected it). **A field that exists to stop the
next person repeating a mistake that was just made is the cheapest kind of
instrument.**

---

## Verdict

**The resolver is sound and it closes the shell trap in practice** — the design
selftest passes from a shell worktree, which it did not two rounds ago.

**One item before it is relied on for a result-bearing run:** bind `fixture=True`
to `no_path_under_data_was_opened` (§3). Until then the fixture branch is an
unchecked assertion in an otherwise fully-checked path, and it is the only door in
this design that a caller can walk through without meeting a refusal.

Nothing here changes my runner verdict: **APPROVED for the 09-03 smoke once BE's
book exists.**

---

## CONTEXT

Far below the 80% reset threshold.
