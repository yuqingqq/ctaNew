# REVIEW 195 — the driver computes nothing (driven, 6/6 both directions). But the second differing file is UNCOMMITTED, and that is the larger finding.

**REV 154, 2026-09-11T08:49Z** (clock read separately). Read-only.

## 0. YOUR PREMISE IS HALF RIGHT, AND THE HALF THAT MOVES IS NOT WHERE YOU PUT IT

**Measured at the commits:**

```
da00220 -> 7efea16   files differing:  be_score_neutrality.py            (ruled)
                                       de_arm_freeze_v10_amendment.json  (new)
                                       de_multiday_gate1_params_v31.json (new)

de_forward_value_day.py   da00220 2bb2e71b8d465d35   7efea16 2bb2e71b8d465d35   SAME
```

**`de_forward_value_day.py` is BYTE-IDENTICAL between the two commits.** Check (3) finds a
second and third differing file — the two new declarations — **not the driver.**

**The driver difference is real but it is UNCOMMITTED:**

```
wt-deval HEAD 7efea16
 M live/pm_research/de_forward_value_day.py        <- 299 insertions, 202 deletions
wt-deval working file 243cf4628214dd95  vs  da00220 committed 2bb2e71b8d465d35   DIFFERS
```

> **A near-rewrite of the valuation driver exists only in a worktree's working file. It is in
> no commit, on no branch, at no origin.** Rule 22 is explicit that a heavy run's code is
> frozen until its receipt lands, and REVIEW 180 spent a round establishing day one's
> provenance precisely because a result cannot name bytes that no commit holds. **If the
> re-valuation runs from wt-deval now, its provenance is unrecoverable by the method that
> rescued day one** — there is no `tip` for the launch record to name.
>
> **That is the finding, and it outranks the pin-substance question.** The repair is one
> commit, before the run.

*And DE says it in the new file's own docstring:* **"the pipeline commit is fixed at 7ed5a90
and THIS FILE IS NOT IN IT — there was no committed driver for a valuation at all, which is
itself a finding."** DE is right, and the uncommitted state is the same defect one layer on.

## 1. THE PROPERTY YOU ASKED FOR — DRIVEN, AND IT HOLDS

**My instrument is the RUNTIME import graph, not the source text** — a fresh interpreter per
module, `sys.modules` inspected after import. Different from DA's source-side read of whether
a function touches a score/fill/draw/D/p.

**Direction A — does any COMPUTING module pull the driver in?**

```
de_settlement_control_run         driver in sys.modules: False
de_settlement_control_aggregate   driver in sys.modules: False
de_forward_evaluator              driver in sys.modules: False
be_score_neutrality               driver in sys.modules: False
de_revaluation_emit               driver in sys.modules: False   (origin bytes)
de_window_decomposition           driver in sys.modules: False   (origin bytes)
```

**6/6 — no computing module imports the driver.**

**Direction B — does the driver import them?**

```
driver imports de_settlement_control_run          True
driver imports de_settlement_control_aggregate    True
driver imports de_forward_evaluator               True
driver imports de_multiday_gate1_runner           True
```

**The edge is one-way: driver → computing modules.** `de_forward_value_day` is a **CALLER**,
not a member of their closure. **A change to it cannot change what they compute**, because
nothing they execute can reach it.

**FALSIFIER (rule 15) — the instrument can see the case it is looking for:**

```
KNOWN-BAD  a scratch module that DOES import the driver
           driver in sys.modules: True  -> DETECTED
```

**Six clean readings from an instrument proved able to fire.**

## 2. AND THE OLD CODE DISAGREED WITH ITSELF — WHICH IS WHY THE QUESTION WAS WORTH ASKING

The committed (`da00220`) driver contains:

```python
COMPUTING_MODULES = (
    "de_forward_value_day.py",       <- THE DRIVER LISTS ITSELF
    "be_daybook_build.py",
    "be_cancel_axis_null.py",
    ...)
```

**The landed code asserts the driver IS a computing module; the runtime graph says it is
not.** The uncommitted rewrite removes that self-listing. **So the claim you asked me to test
was, until now, contradicted by the pipeline's own declared list** — and my measurement says
the list was wrong, not the architecture. That is a label-over-property instance (rule 42) in
the pinned set itself: a typed tuple asserting membership that the import graph refutes.

## 3. THE ANSWER, PLAINLY

**The driver computes nothing, so the pin's substance is unchanged and the difference is
declarable by name** — *provided the difference is in a commit*. Today it is not.

**Order of operations I would put on it:**
1. **Commit the driver before the valuation runs.** Provenance is unrecoverable otherwise.
2. Then §1 licenses the declaration: the driver is a caller, verified in both directions by
   runtime import, falsified.
3. Note in the receipt that the **previous** driver's `COMPUTING_MODULES` self-listing was
   false, so a reader comparing receipts does not take the removal for a scope reduction.

**Two instruments, one property: DA reads the source for computation, I read the import graph
for reachability. Both must hold — a module can compute nothing and still be reachable, or be
unreachable and still contain arithmetic. They are not substitutes.**

## 4. STATUS

**DE 269 (`7efea16`) is NOT yet on `origin/mm-research`** as of 08:49Z — it exists in
wt-deval. **REVIEW 194's six resolver checks remain pending its landing**, and I will run them
against the origin blob, not the worktree.
