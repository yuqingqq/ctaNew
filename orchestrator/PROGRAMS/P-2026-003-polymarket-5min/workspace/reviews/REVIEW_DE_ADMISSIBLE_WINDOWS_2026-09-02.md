# Review — DE rounds 4 + 5 + 6 (registry applier, admissible-window supplier, launch-invariance)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `21f4edf`** (rounds 4/5/6: `f1c8f88`, `f625abe`, `21f4edf`).
**Request of record:** `REQUEST_DE_ADMISSIBLE_WINDOWS_2026-09-02.md` (at `32fe80f`).
**Composed 2026-09-02T10:48:30Z.** One filing, per R-377.
`da_midnight_verify.sh` was **not** run, per the dispatch.

Executed in `~/ctaNew-wt-rev` at `--detach 21f4edf`; filing lands in the shared tree by
pathspec.

**Input identity, recorded as asked:** `da_blackout_mask_20260901.json` sha
**`74a67074654e43f5`** (mtime 10:34:15Z — DA's `770e5ee` regeneration with the
`producer{}` block), read at 10:43Z and unchanged at 10:48Z.
`da_blackout_mask_20260827.json` is **absent from disk**, so the 08-27 case below was
built by calling **DA's own producer in memory** — nothing was written.

---

## Verdict

### RELEASED. All seven scopes verified by execution. No hold from this seat.

One low finding (**RR11-1**) on the two-module-object question the request asked me to
weigh, plus a judgement answer on the two-readings clause.

---

## 1. RR4-4 — the applier refuses non-additive shapes it does not implement

| check | result |
|---|---|
| `load_baseline()` | reads **v24** |
| `load_baseline(ref='HEAD')` | **REFUSED** — *"'HEAD' carries version 26, not the baseline 24 the proposal was drafted against"* |
| `apply_amendments(v24, A–E)` vs the live v26 | **exactly equal** on types, modules and config_supplied — `{}` diff, recomputed by me |

**The shape RR4-4 asked for**, tested by injecting an operation the applier does not
implement into E's parsed record:

| operation | result |
|---|---|
| `change` | **REFUSED** (`MigrationRefused`) — *"is not one this reference applier implements (('remove',)); refusing rather than…"* |
| `rename` | **REFUSED** |
| `replace` | **REFUSED** |

That is the closure: a skipped record would have reproduced a subset and the equality
would have passed on it. It refuses instead.

**DE's own catch is fixed.** The withheld-E control now asserts
`d_no_e.get("config_supplied_only_in_first") == ["ActionSet"] and set(d_no_e) ==
{"config_supplied_only_in_first"}` — the exact difference and nothing else, with the
reason in place (*"A disjunctive predicate here would be satisfied by almost any
difference"*). That is the right repair for a control that was previously satisfiable
by a coincidence.

## 2. The supplier supplies and stamps, never selects

`de_admissible_windows`'s parsed imports are `__future__, argparse, ast, hashlib, json,
pathlib, sys, typing, da_blackout_mask, da_content_liveness_rule` — **no verdict
producer**, and `reads_no_verdict(...)` is True.

**The predicate is over the AST import list, not a grep** — verified in both directions:

- a verdict producer named only **inside a string** leaves the import set unchanged and
  the predicate still passes (a grep would have failed here);
- `import da_forward_day_verify` → **False**;
- `from da_forward_day_verify import verify_day` → **False**.

DE's replacement of its own grep is real, and the known-bad discriminates. The emission
also carries `_g_no_decision_field` as a post-condition on its own output, and
`supplies_not_selects` is stated in the emission.

## 3. On the REAL artifact

`supply('20260901', PRESENT = the full 288-window grid × 7 coins)`:

| coin | supplied | artifact `n_windows_total − n_masked` |
|---|---|---|
| bnb | 266 | 288 − 22 |
| **btc** | **265** | 288 − 23 |
| doge | 266 | 288 − 22 |
| eth | 266 | 288 − 22 |
| **hype** | **279** | 288 − 9 |
| sol | 265 | 288 − 23 |
| xrp | 268 | 288 − 20 |

**Every per-coin identity holds against the artifact's own numbers** (not literals), and
`n_supplied_total = 1875 = 288×7 − 141`.

**Hash sensitivity.** Unmasking one btc window in a copy of the artifact:

- `mask_identity_hash` moves (`27d97942…` → `229ca2bb…`);
- total goes 1875 → 1876, and the unmasked window is now supplied;
- **every one of the 1,875 windows common to both runs has a different `inputs_hash`.**
  No window's stamp survives a change to the mask, which is what makes the stamp mean
  anything.

**08-27**, built from **DA's own producer** (`build_mask('20260827')`, in memory):
`total_masked_windows: 0`, and the supplier emits **2,016 of 2,016** present windows —
the full list — with `mask_consumed: true`, `governed: false`. An empty mask is
consumed and says so.

## 4. Refusals, both directions

| case | result |
|---|---|
| baseline (real artifact) | SUPPLIED 1875, `mask_consumed: true` |
| **governed day + no mask** | **REFUSED** — names the day, the governed threshold and that it was read from the frozen rule |
| **pre-governed + no mask** | **SUPPLIED**, `mask_consumed: **false**` — absent ≠ empty, and the emission says which |
| `day_closed_calendar: false` | REFUSED — partial mask |
| envelope: `artifact` renamed | REFUSED |
| envelope: `coins` → `per_coin` | REFUSED — *"the PRODUCER's committed artifact is the contract"* |
| mask for another day | REFUSED |
| producer count vs list disagreement | REFUSED |
| a PRESENT coin absent from `coins` | REFUSED |
| a masked window absent from the calendar | REFUSED |
| empty calendar | REFUSED — *"this module never derives it — deriving the calendar is selecting"* |
| duplicate starts | REFUSED |

**The two-readings clause, and my answer.** The dispatch's *"a supplied window that the
mask masks"* is implemented as two guards: `_g_masked_subset_of_present` (a masked
window the calendar does not contain — the inputs disagree about what exists) and
`_g_no_masked_window_emitted` (a masked window that survived into the output). **They
are not alternatives and the right answer is that both are needed.** The second is the
post-condition the subtraction itself requires — it is the only one that can catch an
arithmetic slip in `present − masked`. The first is a precondition on the inputs, and
without it a mask/calendar mismatch silently under-subtracts while the post-condition
still passes. Splitting them into one job each is also what let the mutation audit
surface a guard that crashed (`KeyError`) instead of refusing — *"a crash is not a
refusal"*, which is the right standard.

## 5. The mutation harness — live and disabled are now distinct calls

`mutation_audit` on the real artifact reports:

- **3 INPUT guards**, each with a visibly separate `(live, disabled)` pair, each
  `refuses_when_live: true` / `refuses_when_disabled: false` → `load_bearing: true`;
- **2 POST_CONDITIONS**, reported as `kind: POST_CONDITION` with
  `fires_on_its_known_bad: true` — **reported as post-conditions, not counted as
  mutant kills**;
- `survivors: []`, `all_load_bearing: true`.

**Rule 16, tested:** disabling one input guard's refusal makes the suite **FAIL by
name** (*"KNOWN-BAD: the mask masking a window the calendar does not contain
REFUSES"*). The controls can fail.

DE's self-found defect — the "live" run passing `skip_guard` too, so every guard was
measured with itself disabled — is genuinely closed: the two lambdas differ in exactly
that argument, and the audit's own output distinguishes the two runs.

## 6. `rule_policy_v1`'s re-anchored known-bad

`code_binding()` reports **five files, all BOUND**: `rule_policy_v1.py`,
`ev_replay_seam.py`, `harmful_stateful_policy.py`, `de_constraints.py`,
`de_actionspace.py` — the round-3 hash artifact still binds every engine file.

The anchor now resolves to **`13bae317…^` = `e252b8e`**, the parent of the commit that
added the file:

- the file **does not exist** there — so it is a valid known-bad;
- `verify_binding(...)` at the anchor → **False**; at its own carrying commit →
  **True**. Both directions.

The repair is the right kind: the anchor is **located** (parent of the introducing
commit) rather than **counted backwards from the tip**, so it cannot drift as the branch
advances. I note that `HEAD~50` happens still to predate the file today — the point is
not that the old anchor is broken this hour, it is that its correctness was a function
of branch length and now is not.

## 7. CO-2 / CO-3

**Both launchers, all modules, rc 0:**

| module | checks |
|---|---|
| `de_registry_amendment_check` | 44 |
| `de_admissible_windows` | **39** |
| `rule_policy_v1` | **37** |
| `de_actionspace` | **17** |
| `de_lane4_real_parity` | **35** |
| `ev_replay_seam` | **41** |

matching the coordinator's 39/17/35/41/37 on the five it measured, under both
`python3 -m live.pm_research.<m>` and the script-directory launch.

`MASK_GOVERNED_FROM_DAY` is gone; `is_governed` reads `RULE_MODULE.EFFECTIVE_FROM_DAY`
at call time — patching `RULE_MODULE`'s value to `20260905` makes `is_governed('20260902')`
**False**, and the unreadable case raises `GoverningRuleUnreadable`.

### RR11-1 — LOW — the two module objects are a non-issue in production and a live hazard in controls

Executed: `da_content_liveness_rule` and `live.pm_research.da_content_liveness_rule` are
**two distinct module objects** for the **same file**, with the **same value**
(`20260902`) at import. Patching one does **not** reach the other.

- **For a consumer comparing BE's governing day with DE's: not a hazard.** Both derive
  the value from the same frozen file at import, and nothing writes to it at runtime, so
  the two objects cannot disagree in production. If one ever could, that would mean the
  file changed under a running process — a different problem with a different remedy.
- **For controls: a real hazard, and the same shape as CO-1.** DE's `RULE_MODULE` is
  the **bare** object; a control that patched the `live.pm_research.…` spelling instead
  would change nothing and pass vacuously. I confirmed the asymmetry by patching the
  bare object and watching `is_governed('20260902')` flip while the package object kept
  `20260902`.

**Closure (cheap):** in any control that patches the rule, assert first that the object
being patched **is** the one the code under test bound — e.g.
`assert RULE_MODULE is sys.modules['da_content_liveness_rule']` — so a future test that
patches the other spelling fails loudly instead of silently testing nothing.

---

## Executed evidence

At `21f4edf`, 2026-09-02T10:43–10:48Z:

| check | result |
|---|---|
| six modules × two launchers | **rc 0 everywhere**; 44/39/37/17/35/41 checks |
| `load_baseline` v24 / HEAD refusal | v24 / **REFUSED by version** |
| A–E on v24 vs live v26 | **exactly equal** (types, modules, config_supplied) |
| unimplemented ops `change`/`rename`/`replace` | **all three REFUSED**, not skipped |
| import-list predicate | string mention ignored; `import` and `from…import` both caught |
| real 09-01 supply | per-coin identity holds for all 7 coins; **total 1875** |
| unmask one window | identity hash moves; **all 1,875 `inputs_hash` move**; total 1876 |
| 08-27 from DA's producer (in memory) | **2,016 of 2,016** supplied, `mask_consumed: true` |
| refusal matrix | **10 refusals by name**, plus the pre-governed permit with `mask_consumed: false` |
| mutation audit | 3 input guards load-bearing, **0 survivors**, 2 post-conditions reported as such |
| audit control mutant | **KILLED** |
| `rule_policy_v1` bindings | **5 of 5 BOUND**; anchor `13bae31^`, False there / True at the carrying commit |
| two module objects | distinct objects, same file, same value; patch does not cross — RR11-1 |
| code mutants executed | 1 (the audit control); the remaining 20+ probes were API- and data-level |
| worktree after the review | clean |

---

## Disposition

- **RELEASED:** DE rounds 4, 5 and 6. RR4-4 is closed by an applier that refuses the
  shapes it does not implement; the supplier demonstrably supplies and stamps without
  selecting, and its identity stamp is sensitive to every input it claims to depend on;
  the refusal set is complete in both directions; the mutation harness measures live
  guards against disabled ones; the drifted anchor is now located rather than counted;
  and five modules are launch-invariant under both launchers. **No hold.**
- **FILED, not holding:** RR11-1 — assert same-object before patching the frozen rule in
  any control.
- **Answered, as asked:** the two readings of the mask/calendar clause are a
  precondition and a post-condition, not alternatives; both are required, and the
  post-condition is the one the subtraction itself needs.
