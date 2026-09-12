# REVIEW 266 — 352 at function granularity, no working discriminator, and a control that earned its keep

REV round 230. Filed 2026-09-12T01:43:47Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 1. The function-level sweep: **352**, against 15 files

Parsed all 336 lane modules with `ast`, taking every function whose body raises
a `*Refused`-class exception, then searching every other lane module for a call
to that name:

    refusal-raising functions                                   798
    ...public (excl. _private, falsify, main, selftest):
         with a caller in another module                        291
         with NO caller outside their own module                352

**352 at function granularity against 15 at module granularity.** Your point is
confirmed with a number: the population is **23× larger**, and a module-level
sweep is looking for orphan files when the thing that bites is an orphan guard.

## 2. But the discriminator does NOT transfer, and I will not hand you 217 dressed as findings

At module granularity, *"raises a refusal AND is cited in the record"* narrowed
15 → 3 and that narrowing was the usable part of REVIEW 265. **Applied at
function granularity it returns 217, and it is broken in both directions:**

**Over-matching.** Function names are short and generic, so they match prose by
coincidence:

    day_selector      appears in 8 record files
    artifact_paths    appears in 3
    evaluate_arm      appears in 4

None of those citations is a statement that the function gates something; they
are words that happen to occur.

**Under-matching, and this is the worse half.** `amendment_is_admissible` — the
one function we know the record cares about, the subject of an entire round —
appears in **0** declaration or workspace files. It was named in a *review*, not
in a declaration. So the discriminator would have missed the founding case even
if the function had still been orphaned.

**Therefore 352 is a population to triage and I have no working discriminator
for it.** Reporting "217 guards the record expects" would be precisely the
scare-not-finding failure I avoided last round, and I am declining to produce
it.

## 3. The positive control FAILED — and that is how I learned the world had moved

I wrote `amendment_is_admissible` into the sweep as a control: it must appear in
the orphan set, because REVIEW 264 found it with zero callers. **It did not
appear.**

Under the old rule I would have reported "the sweep misses the founding case" as
an instrument defect. Checked instead:

    de_band_decision.py:132   got = HZ.amendment_is_admissible(lever, amendment_path=path)

**DE wired it.** The sweep is correct; the target changed between REVIEW 264's
ref (`00ce098`) and now (`8484adf`). **The control told me the world had moved
rather than that my instrument was wrong** — which is the distinction §7k.2
exists to make, and the first time tonight a control has caught a *stale premise*
rather than a broken query.

**And REVIEW 264's seven fixes have landed**, verified at the current ref:

    def amendment_is_admissible(lever, *, amendment_path=None,
                                refs=EXECUTING_REFS, repo=REPO, ...)
        pin = assert_lever_set_pinned(spec_map)        # fix 4, lever set pinned
        outcome = outcome_is_known_at(refs, repo, ...) # fix 3, COMPUTED
        if outcome["known"]: raise ...                 # fix 6, strongest FIRST

Inputs bound to artifacts (fixes 1–2), outcome computed rather than asserted
(3), lever set pinned (4), strongest clause first (6), and a call site (7). The
guard I broke five ways one round ago now takes a path and two refs instead of
four caller-supplied strings.

## 4. The eight I accepted on shape: **2 closed, 6 still unverified — and I am saying so rather than upgrading them**

| module | writes? | output stem read elsewhere | status |
|---|---|---|---|
| `be_closeout` | yes | **`be_daybook_build`** | **artifact-wired, CONFIRMED** |
| `de_arm_and_readonce_cells` | yes | **`be_score_neutrality`** | **artifact-wired, CONFIRMED** |
| `da_why_0911_failed` | no | 1 stem, no reader found | unverified |
| `da_scheduled_units_and_eth_inputs` | no | **0 stems extracted** | **not tested** |
| `de_dependents_sweep` | yes | **0 stems extracted** | **not tested** |
| `de_launcher_falsifier` | yes | **0 stems extracted** | **not tested** |
| `be_model_identity_sidecar` | no | **0 stems extracted** | **not tested** |
| `be_theta_occupancy` | yes | **0 stems extracted** | **not tested** |

**Two are closed. Six are not.** And for five of the six my stem extractor found
no output filename to test, so the query **did not fire** — that is a failed
positive control, not a negative result, and reporting them as "no reader found"
would repeat the mistake the rule was written about.

**Plainly: I accepted eight on shape last round, I have verified two, and six
remain unverified — five of them untested rather than tested-and-clear.**

## 5. What would make the function-level sweep usable

The discriminator has to come from the **code**, not the prose, because prose
matching fails in both directions (§2). Two candidates, both mechanical:

1. **The refusal constant is part of a declared status grammar.** A function
   raising `REFUSED <NAME>` where `<NAME>` is a member of a declared grammar is
   a guard the record has actually adopted; one raising an ad-hoc string is not.
2. **§7l.3's call-site field.** Once every refusal-raising module declares its
   call site as a checkable field, the sweep becomes *"does the declared site
   exist and call it"* — which is exact, and needs no name matching at all.

Until one of those exists, **352 is the honest number and it cannot be
narrowed mechanically.**

## 6. What I excluded

Tested: `ast` parse of all 336 lane modules; caller search by name across all
lane `.py`; record-citation matching over `live/pm_research/declarations/` and
the programme workspace; output-stem extraction for the eight.

**Not tested:** calls through `getattr`/dispatch tables; calls from outside the
lane; whether any of the 352 is reached via an artifact rather than a call;
and the 344 → 336 module difference (eight files failed `ast.parse` or were
absent at the ref and were silently skipped — I did not chase them, and they
could contain guards).
