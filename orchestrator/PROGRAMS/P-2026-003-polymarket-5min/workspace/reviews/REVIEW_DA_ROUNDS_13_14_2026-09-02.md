# Review — DA rounds 13 + 14 (the ruled state and a block that cannot contradict itself; twinning in both directions)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `801eb31`** — HELD in `~/ctaNew-wt-da`, unpushed; chain `3a89e6c` →
`e292439` → `636a455` (released at `852b9aa`) → `e384792` (round 13) → `801eb31` (round 14).
**Request of record:** `REQUEST_DA_ROUNDS_13_14_2026-09-02.md`.
**Composed 2026-09-02T15:51:24Z.** One filing, per R-377.

**Constraints observed.** Read-only under `data/`; the `derived/` listing (184 entries) is
**identical** before and after. `DA_MIDNIGHT_MODE` never set; `da_midnight_verify.sh` never run;
no timer, service or unit touched; `DA_MIDNIGHT_OUTDIR` + `DA_MIDNIGHT_LOG` at scratch.
`__pycache__` cleared **before** every mutant execution (R-446); worktree clean after each.
**The full gate runner was run once — in my own worktree copy, never against the canonical tree —
under `systemd-run --user --scope -p MemoryMax=8G`**, as the request permits, with a scratch
`PM_DATA_ROOT`.

Scope confirmed: round 13 = `da_governed_verdict_preflight.py` only (+122/−22); round 14 =
`v5_deploy_gates.py` (+174/−19) and one assertion in `da_blackout_mask.py` (+21/−5, selftest
region). Suites: preflight **39**, gates' own selftest **6**, mask **30** — all under both
launchers, rc 0.

---

## Verdict

### RELEASE for both `e384792` and `801eb31`. The dispositions do not differ.

Round 13's wiring proof is the real thing — deleting the production call fails **by name**, which
is rule 17 closed rather than asserted. Round 14 twins in both directions and prints every
exclusion with its reason; the roster line reads `23 declared + 15 derived twins + 1 injected
canary = 39` and only the canary is red.

Three findings, all small, and two of them are the same shape the programme has now closed
elsewhere: a control that encodes today's arrangement (**DA13-R1**, **DA14-R2**) and a control
that cannot fail (**DA14-R1**). Item 8's ruling is the one that matters: with the fixture as
built, **a producer hardcoded to `True` passes the suite**.

---

## 1. The ruled state — five ruled, nothing open

At the artifact: `ruled` = `R-408(2)`, `R-408(3)`, `R-411(i)`, `R-411(ii)`, **`freeze_disposition`**;
`still_open == {}` (an empty dict, not a placeholder) with a note saying *"`still_open` is what
remains the USER's, and it is EMPTY: nothing remains the USER's"*. No `-- USER` label survives
anywhere in the block (JSON search). Each entry carries the entry that settled it — R-424 §2/§3/§4
for the four, **R-442** for the freeze disposition, transcribed as *"race on the frozen bytes at
`1b53929` (R-424 §6 adopted verbatim); no re-freeze"*.

## 2. The block cannot contradict itself — and the wiring proof is genuine

`_assert_decisions_coherent` (`:121`) refuses on (1) a key in both halves and (2) any pre-ruling
phrase in the block's JSON. The production call is at `:428`.

**Driven:**

| mutant | result |
|---|---|
| the production call at `:428` **deleted** | **red by name** — *"FAIL: the coherence guard is NOT called by preflight()"* |
| a ruled key added to `still_open` | **refused** — *"['R-411(i)'] appear in BOTH `ruled` and `still_open`"* |

**Ruling on the poisoning.** The technique is right and it is the only honest way to prove the
call: the suite temporarily replaces `STALE_DECISION_PHRASES` with `("RULED at R-442",)` — a
string the real block carries — so the production path is the only thing that can raise, then
restores it in a `finally` and asserts the restoration by identity. That is the same lesson BE
learned when a `require_verified_called: True` flag survived deleting the call: a flag beside a
call is forgeable, a behaviour through the call is not.

**But the poison is a specific citation, and that is DA13-R1.** It works because
`freeze_disposition` says *"RULED at R-442"* today. Every ruled entry carries the citation **form**
(`"RULED at "` — I checked all five), which is the block's actual discipline; the specific ref is
an accident of which entry ruled last.

## 3. Six mutants

Two reproduced above (the deleted call; a key in both halves), each red by name.

## 4. Tonight is untouched — by hash

| | |
|---|---|
| the shared tree's `da_governed_verdict_preflight.py` | sha `6a15ed5dd25513b7` — **equal to `fadc986`'s** (round-9 vintage), and the file is unmodified in that tree |
| the held chain's version | sha `f2c728a937399630` — a different file, in a worktree, on no branch |
| the unit | `ExecStart=/home/yuqing/ctaNew/live/pm_research/da_midnight_verify.sh` |
| the timers | `da-midnight-verify` **Thu 2026-09-03 00:06:00 UTC**; `co-preflight-…` **00:14:00 UTC** |

Nothing in the chain can reach either read.

## 5. DA12-R1 — twinning in both directions, every exclusion named

`_launch_twins` (`:157`) derives a `-m` twin from a path gate **and a path twin from an `-m`
gate**. Measured on the real roster: **23 declared, 15 twins, 8 excluded**, and every gate lands
in exactly one list (15 + 8 = 23).

**The exclusion's reason is true at the pin**, reproduced rather than taken on report:

```
python3 live/pm_research/tier1_pipeline.py --selftest  -> rc 1  ModuleNotFoundError: No module named 'live'
python3 -m live.pm_research.tier1_pipeline --selftest  -> rc 0
tier1_pipeline.py:55 -> from live.pm_research.coverage_ledger import (
```

**Named, not repaired:** `tier1_pipeline.py` does not move anywhere in the chain
(`git diff 3a89e6c..801eb31` on that path: no files), which is the coordinator's ruling honoured.

*MEM's caveat, stated as the property it is:* `_launch_twins` anchors path gates on
`Path(argv[1]).parent == HERE`, so a **relocated copy** of the runner derives fewer twins — and
its own count assertion then refuses the run. That is the runner declining to certify a roster it
cannot fully twin, not a defect.

## 6. The runner reads its own output

Every exclusion is printed, both kinds:

```
1 gate(s) EXCLUDED from twinning:
  - tier1 normalisation: already `-m`; the path launch FAILS with ModuleNotFoundError … named, not repaired.
7 gate(s) have no second launcher to derive (not --selftest module gates): v5 heartbeat behaviour, …
roster: 23 declared + 15 derived twins + 1 injected canary = 39
```

The accounting line is computed from the roster, not written down. Driven: **an exclusion computed
but not printed → red by name** (*"a NAMED exclusion is returned with its reason, not silently
skipped"*).

**The full roster, run once in my worktree under an 8 G scope:** `--falsify` → **39 gates, 1 FAIL
— the injected canary — "falsifier fired"**, with 15 twin gates executed. 23 + 15 + 1 = 39, end
to end.

## 7. The synthetic roster — right falsifier, and the self-reference is stopped by construction

**Ruled: yes, a synthetic roster is the right falsifier here.** The property DA12-R1 asked for —
the reverse direction — cannot be exhibited by the real roster, because its only `-m` gate is the
excluded one. A fixture that contains a path gate, an `-m` gate, a named exclusion and a
behavioural gate is the smallest input on which both directions and both exclusion kinds can fire,
and it is the same idiom the repo uses elsewhere (`v5_chain_equivalence_test`'s one fixture, two
consumers). Driven: removing the `-m` → path branch turns the runner's selftest **red by name**.

**The self-reference is stopped by construction, not by a special case.** The runner's own gate
invokes `--selftest`, and `main()` returns on `--selftest` **before the roster is built** (the
comment at `:94-96` says so and the code does it). So the gate that runs the runner runs six
checks on a synthetic roster; it never re-enters the roster, and twinning it produces a `-m` twin
of the same `--selftest` entry point. No recursion, and nothing had to be excluded to achieve it.

One check inside that selftest does not do what its label says — **DA14-R1**.

## 8. Ruling — the scope deviation, and what the control now discriminates

The hunk is the RR12-1 control's expectation only (`:853-873`), in the selftest region, and the
property is right: the flag must agree with **the child tree's own measured state**, computed by
`git status --porcelain` on the four producing files rather than assumed. Accepting it in-batch
was the right call — a red control cannot land.

**What it discriminates, measured rather than argued.** With the fixture as the control builds it,
`_exp_dirty` is **True**, so:

| the producer's flag | suite |
|---|---|
| `_tree_dirty()` (real) | 30 checks, rc 0 |
| hardcoded **`True`** | **30 checks, rc 0 — SURVIVES** |
| hardcoded `False` | red |
| `not _tree_dirty()` (DA's inversion) | red |

DA's "inverting the flag still fails" is true and is the weaker claim: inversion is caught because
it moves the value away from the arrangement. **The constant that matches the arrangement is not
caught — and it is `True`, the exact literal the control used to assert before this fix.** So the
control now discriminates the flag's *value* but not the fact that the producer *measured* it.

**Ruled: drive both arrangements — a finding for round 15, not a hold.** The fixture already
creates the child worktree and copies files into it; committing them there (or copying nothing)
gives the clean arrangement, and with both, each constant is red somewhere. That is the difference
between "the expectation is computed" — true now — and "the producer's answer is discriminated",
which is what the control's name claims. **DA14-R2.**

## 9. Nothing else moves

| | |
|---|---|
| the shared tree, nine DA files | **all byte-identical to `b75c9fe`** |
| `derived/` | listing identical before and after |
| the unit and timers | unchanged; next elapse Thu 2026-09-03 00:06:00 UTC |
| `--require-no-skips` | **not wired** — zero occurrences in the runner, the launcher or the unit |
| `~/ctaNew-wt-da` | clean at `801eb31`; my worktree clean after every mutant |

Every new message interpolates what it evaluated (the counts, the excluded labels and reasons, the
measured flag beside the expectation, the keys in both halves). No emission carries a boolean that
encodes an entitlement.

---

## Findings

### DA13-R1 — LOW — the wiring control's poison is a specific citation, so a legitimate re-ruling turns it red

The suite proves the production call by poisoning `STALE_DECISION_PHRASES` with `("RULED at
R-442",)` and requiring `preflight()` to raise. It works today because the block says exactly
that. The moment any decision is re-ruled in band — a superseding entry, rule 13 — the citation
becomes `RULED at R-4xx`, the poisoned run stops raising, and the control goes **red for a change
that is not a defect**.

That is the DA10-R5 class, now in its fourth place in this seat's suites, and the direction is
safe (a false red, not a false green) — which is exactly what makes it easy to fix in advance.

**Closure:** poison with the citation **form** the block must always carry — `"RULED at "` — which
I confirmed on all five entries. It proves the same call, and it survives every future ruling
because the block's own discipline is that a ruled entry cites the entry that settled it.

### DA14-R1 — LOW-MEDIUM — the "the equation can fail" check cannot fail

`_selftest`'s last check drops one element from the returned `twins` and asserts
`len(_t2) + len(excluded) != len(roster)`, labelled *"the equation the runner asserts is able to
be false, which is what makes asserting it a check"*.

Given the function's own invariant — every entry lands in exactly one list, which the check two
lines above asserts — the expression is **arithmetic, not behaviour**. Evaluated across four
arrangements (3+1, 2+2, 0+4, 7+1 over rosters of 4 and 8): **True in every one**. No change to
`_launch_twins` can make it false, so it tests nothing about the runner.

It is the DE16-R4 shape, which this programme has now closed twice elsewhere — DE replaced three
such lines with harness hooks in round 22, and BE5-R2 named the same thing last round.

**Closure:** either delete it and let the message say what is true — the equation is *structural*,
and asserting it documents an invariant rather than testing one — or give `_launch_twins` a test
hook (`_drop=…`) that can actually return an entry reaching neither list, which is what DE did
after the same finding.

### DA14-R2 — LOW — the RR12-1 control does not discriminate the constant that matches its fixture

Measured above: with the control's single arrangement (`_exp_dirty` True), a producer whose flag
is hardcoded **`True` passes the suite**; `False` and the inversion are caught. The expectation is
computed — the DA10-R5 defect this hunk fixed — but one constant still survives, and it is the
one the control previously asserted literally.

**Closure (round 15, as the request frames it):** drive both arrangements — child files equal to
HEAD, and child files differing — so each constant is red somewhere. The fixture already builds
the tree; the clean arrangement is one commit in it.

---

## Executed evidence

At `801eb31` (chain from `636a455`), 2026-09-02T15:45–15:51Z:

| check | result |
|---|---|
| scope | round 13 preflight only; round 14 runner + one mask assertion |
| suites | preflight **39**, gates selftest **6**, mask **30**, both launchers, rc 0 |
| the ruled state | five ruled incl. `freeze_disposition` (R-442), `still_open {}`, no `-- USER` label |
| the production call deleted | **red by name**: *"the coherence guard is NOT called by preflight()"* |
| a key in both halves | refused, naming it |
| tonight | shared preflight sha `6a15ed5d…` = `fadc986`; the chain's is `f2c728a9…`; timers 00:06 / 00:14 |
| twinning | 23 declared / **15 twins** / 8 excluded; 15 + 8 = 23 |
| the tier1 reason | path rc **1** (`No module named 'live'`, `:55`), `-m` rc **0**; the file moves nowhere in the chain |
| exclusions printed | the named one with its reason + the seven enumerated + `23 + 15 + 1 = 39` |
| exclusion computed but not printed | **red by name** |
| the reverse direction removed | **red by name** on the synthetic roster |
| self-reference | `--selftest` returns before the roster is built — no recursion by construction |
| **"the equation can fail"** | **True under every arrangement** — DA14-R1 |
| **the RR12-1 flag** | hardcoded `True` **survives**; `False` and the inversion are caught — DA14-R2 |
| full roster (`--falsify`, my worktree, 8 G scope) | **39 gates, 1 FAIL — the canary**; falsifier fired |
| nothing moved | nine shared files byte-identical to `b75c9fe`; `derived/` identical; unit unchanged; `--require-no-skips` unwired |
| worktrees | `~/ctaNew-wt-da` clean at `801eb31`; mine clean |

---

## Disposition

- **RELEASE** for **`e384792`** and **`801eb31`** — the dispositions do not differ. Round 13
  closes R-442's state with a wiring proof that fails by name when the call is removed, which is
  rule 17 met rather than claimed; round 14 closes DA12-R1 in both directions with every exclusion
  named and printed, and the reason it names is true at the pin. **No hold.**
- **RULED (item 7):** a synthetic roster is the right falsifier for a property the real roster
  cannot exhibit, and the runner's self-reference terminates by construction (`--selftest` returns
  before the roster is built) rather than by a special case.
- **RULED (item 8):** the recomputed expectation is the right property, and it is not yet
  sufficient — a producer hardcoded to the arrangement's value passes. Drive both arrangements;
  a finding for round 15, not a hold.
- **FILED:** **DA13-R1** (the poison is a specific citation), **DA14-R1** (a check that cannot
  fail), **DA14-R2** (the surviving constant).
