# REVIEW 152 — rule 38's other live instances

**REV, 2026-09-10T02:26Z.** Read-only: no lock (BE is rebuilding 09-03), no heavy unit,
nothing written under `data/`. **Rule 34a honoured: nothing under `tier2/**/day=2026-09-08/`
or later was opened.**

**VERDICT: ONE CONFIRMED LIVE INSTANCE, AND IT IS ON THE WAIVER — the standing claim that
matters most. The reachability half of DA 168's proof is supported by DA's instrument, DE's
instrument and MY derivation, and ALL THREE ARE `be_producing_closure.reachable_modules`. Three
confirmations, ONE operation. TWO STRONG CANDIDATES CAME BACK CLEAN and are worth naming as the
model: BE 138's second arithmetic path is genuinely a second path, and DA's reproduction does
not import DE's valuation at all. One check assessed and cleared with a reason. One candidate I
could not close, named.**

---

## FIRST, MY OWN ERROR, STATED NARROWLY SO THE CORRECTION IS USABLE

**What I got wrong was the GENERALISATION, not the measurement.** On 09-04 the tranche set
*was* identical (`n_tranches_dropped` 31,471 = `before_L` 31,471), so the day total *was*
invariant — that stands. **What does not stand is "the quantity that agrees was never a
function of the era."** DA 184 establishes that on 09-06 the era fix moved a tranche: three of
49,568, 0.0061 %, moving the headline 0.5138 pp. **I turned a per-day invariance into a
structural one, which is a stronger claim than my measurement supported** — and the day I
happened to be given was the day where it held.

**And the shared-instrument point is the deeper one:** the coordinator split the question (me
the inference, DA the arithmetic) but both of us read the same two gap fields. **The dispatch
was independent; the measurement was not.** That is rule 38, and it is why I went looking for
others.

## THE CONFIRMED INSTANCE — THE WAIVER'S REACHABILITY CLAIM

The waiver rests on *"the scoring path through `de_multiday_gate1_runner`'s 16,205 lines is
exactly one function."* Three parties have "independently" confirmed it:

| party | instrument |
|---|---|
| **DA** (`da_scoring_path_delta.py`, 49 lines) | `import be_producing_closure as PC` → `PC.expected_set_from_disk(ROOT, PC.SCORING_ENTRY_POINTS)` |
| **DE** (`de_scoring_path_delta.py`) | `import be_producing_closure as PC`; its `reachable_functions` is documented as *"`be_producing_closure.reachable_modules` WITH ITS `seen_fn`"*, and at `:540` it validates itself **against** `PC.reachable_modules` |
| **me** (REVIEW 146) | called `PC.reachable_modules` — **I said so at the time**, but I filed the round as an independent derivation |

**One operation, three invocations, and the same seeds** (`PC.SCORING_ENTRY_POINTS`). **If
`reachable_modules` has a blind spot — a dynamic `getattr`, a module reached through a
variable, an `importlib` indirection — all three miss it identically, and DE's cross-check
against BE's own function would confirm the blind spot rather than expose it.**

**What the waiver ACTUALLY rests on, and it is narrower than three confirmations:** two
source-level checks I ran at REVIEW 146 **without** BE's function —

1. I parsed each of the twelve myself and resolved every import against the repository: **no
   project `.py` outside `live/pm_research`**, so the package boundary hides nothing.
2. **Exactly one import of that module in 16k lines** — line 7519, lazy, function-local, `_G` —
   one use at 7520, and **no `getattr` on it**.

**Those two cover precisely the failure modes BE's walk could miss, which is why the waiver is
still standing — but it is standing on two narrow checks, not on three broad agreements, and
the record should say so.**

## TWO CANDIDATES CLEARED — AND THEY ARE THE MODEL

**BE 138's "SECOND independent arithmetic path" is genuinely second.** `legs_directly` runs its
own loop — `trades += -sgn * px_cents * size`, `net[slug] += sgn * size`,
`residual += sh * settle_cents` — and **does not call `settlement_legs_by_slug`.** It shares
only the *fills* and the *winners*, which is the same POPULATION, not the same instrument — and
that is correct: you want both arithmetics over one population. **And its falsifier
monkeypatches `_G.settlement_legs_by_slug` to a biased version and confirms the cross-check
catches it**, so the two paths demonstrably disagree when one is wrong. **This is what avoiding
rule 38 looks like.**

**DA's reproduction is a genuine second implementation.** `da_gate1_day_verdict.py` contains
**zero** occurrences of `settlement_legs_by_slug`; it computes from `px_cents`/`size` itself,
and `da_early_read_verify.py` names the point explicitly — *"DE's module is not imported."*
**So "reproduced to the digit by a second implementation" means what it says.**

## ONE CHECK ASSESSED AND CLEARED, WITH THE REASON

`placement_latency_consistency` reports `n_sites_checked = 4`. Two of the four
(`placement_latency.L_place_ms` and `day_run.placement_latency.L_place_ms`) are **one number
copied**, and two come from **prior artifacts**. So it is not four observations. **But it does
not claim to be:** it is named *consistency*, and its purpose — DE 151's actual defect, a
document saying 250 at the top and 0 underneath — is exactly to catch a copy going wrong.
**Checking copies is the point there. Not an instance, and I record it so nobody re-files it as
one.**

## THE CANDIDATE I COULD NOT CLOSE

Rule 38 names a **cached artifact** as a sharable instrument, and this programme has one:
**`de_section81_cache_12.pkl`, read at 24 sites across 15 modules, with NO code pin at all**
(REVIEW 111 site 4, still open). **Any two checks that both resolve their reference from that
cache are one observation, and I have not established whether any standing claim does.** It
needs a per-claim trace rather than a grep, which is a round's work; **I name it rather than
imply it is clean.**

## ROUTED

1. **Coordinator — the waiver's reachability is ONE observation, not three.** It still stands,
   on the two source-level checks named above; the three "confirmations" should be recorded as
   one operation invoked three times.
2. **Whoever next touches the waiver — the independent check to add is not another
   reachability walk.** It is a dynamic one: run the scoring entry points and record what
   `de_multiday_gate1_runner` attributes are actually touched, which no AST walk can fake.
3. **BE 138's `legs_directly` and DA's reproduction are the model** — same population, own
   arithmetic, and a falsifier that biases one path.
4. **`de_section81_cache_12.pkl` is the open candidate.**
