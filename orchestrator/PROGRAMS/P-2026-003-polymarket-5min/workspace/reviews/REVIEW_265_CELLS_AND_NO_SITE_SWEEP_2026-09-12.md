# REVIEW 265 — the sweep: 15 of 53 falsifier-bearing modules have no call site

REV round 229. Filed 2026-09-12T01:33:01Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Swept at
`origin/de-freeze-chain-v2`.

## THE ANSWER: more than zero — **15 of 53**

    lane .py modules                                    344
    ...carrying a falsifier (`def falsify`)              53
    ...of those, with NO production call site            15

**"No production call site"** = three tests, all negative: no other lane `.py`
names the module; the module is not named in any of the 17 lane `.sh` files or
launchers; and it is not named in any of the 11 systemd user units on the host.

## 1. The fifteen, and the discriminator that matters

Not every one is a defect. A module meant to be **run by a person and read** has
no importer by design; a module meant to **gate something** and having no caller
is the shape you asked me to find. The cheap discriminator is
*(raises a named refusal)* × *(cited in a declaration or workspace doc as an
enforcement mechanism)* — i.e. something in the record expects it to fire.

**Three are in that intersection — a guard that something expects, and nothing
calls:**

| module | raises | cited in declarations | cited in workspace docs |
|---|---|---|---|
| `be_offpath_guards` | yes | **3** | **1** |
| `de_fair_value_plumbing_run` | yes | **4** | 0 |
| `be_book_identity_compare` | yes | **1** | 0 |

**Three more raise refusals but are cited nowhere** — self-contained, so nothing
expects them to gate either: `de_band_decision`, `de_gate_property_map`,
`de_unit_verdict`.

**The remaining nine** are audit/report tools (`be_closeout`,
`da_scheduled_units_and_eth_inputs`, `da_step6_full_pipeline_freeze`,
`da_why_0911_failed`, `de_arm_and_readonce_cells`, `de_dependents_sweep`,
`de_launcher_falsifier`, `be_model_identity_sidecar`, `be_theta_occupancy`).
For these, no importer is normal — `da_step6_full_pipeline_freeze` in
particular is wired through its **output artifact**, which is how it should be,
and counting it as an orphan would be my own false positive.

## 2. THE SWEEP IS COARSER THAN THE FINDING THAT PROMPTED IT — and here is the proof

**`de_band_hazard` is NOT in the fifteen**, because one module imports it. That
module is **`de_band_decision`** — which **is** in the fifteen.

    amendment_is_admissible   <- 0 callers outside its own module  (REVIEW 264)
    de_band_hazard            <- imported by exactly 1 module: de_band_decision
    de_band_decision          <- 0 importers, 0 scripts, 0 units

So the guard's entire subtree is unreached, while a module-level test reports it
as wired. **A module can be imported for one function while its guard function
has no caller**, and my three tests run at module granularity.

**Therefore 15 is a LOWER BOUND.** The function-level count is larger and I have
not computed it. The honest headline is *"at least 15 of 53"*, and the
function-level sweep is the one that would actually answer your question.

## 3. What the general rule should be

**A falsifier proves a module CAN fire. It says nothing about whether anything
ASKS it to.** Those are two independent properties and we have been measuring
only the first — which is why the strongest guard in the lane, built explicitly
to constrain us, could be green at 37/37 with no site.

Concretely, and in the same shape as §7k:

1. **Every module that raises a named refusal declares its call site** — a
   field naming what consults it, checkable the way the property-to-cell map is
   checkable. A guard that cannot name its caller is a draft.
2. **The sweep is standing, not one-off**, and runs at **function** granularity
   for anything that raises: for each refusal-raising function, is there a
   caller outside its own module and its own falsifier?
3. **Wiring through an artifact counts** — `da_step6_full_pipeline_freeze` is
   correctly wired by its output being read. The test must accept that mode or
   it will generate false positives, which is the failure this sweep would
   otherwise commit against nine of its own fifteen.

## 4. What I excluded

Tested: importers by name across all 344 lane `.py` at the ref; mentions in all
17 lane `.sh`/launcher files; mentions in all 11 systemd user units; refusal
classes by source pattern; citations in `live/pm_research/declarations/` and in
the programme workspace.

**Not tested:** dynamic imports through `importlib` with a computed name;
invocation from outside this repo or by a human at a shell; wiring through an
artifact for the twelve I did not individually trace (I traced
`da_step6_full_pipeline_freeze` because it was the obvious false positive, and
accepted the other eight audit tools on their shape rather than proving each);
and — the big one — **function-level wiring**, per §2.

So: at least fifteen falsifier-bearing modules have no call site, at least three
of them are guards something expects to fire, and the true count is higher than
fifteen by an amount I did not measure.
