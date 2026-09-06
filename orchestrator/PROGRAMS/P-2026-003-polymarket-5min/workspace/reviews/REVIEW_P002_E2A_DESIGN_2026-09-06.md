# REVIEW — E2-A is APPROVED to run once the two escalations are ruled; the resolver adoption reproduces; and the closed forms are pinned at the boundaries but not in the interior

**Filed** 2026-09-06T05:12Z · reviewer seat (pm-codex) · tip `a751dfe` · **no tape
opened** · no data written · filed in P-003's review dir, subject P-002.

**ROUTING — CHECKED**, everything driven by me under the rule-20 wrapper.

## VERDICT

**(A) E2-A is APPROVED to run — conditional on the USER/coordinator ruling the two
escalations, which is exactly what DA asked for and did not absorb.** One item for
v3 (§A.4). **(B) The resolver adoption reproduces**, and all three of my E2.0
findings are now driven checks with the names I recommended.

---

# (A) The E2-A declaration

## A.1 Population, bracket, gate — sound, and two guards I did not expect

**Population is NOT simply E2.0's 16 days**, and the difference is declared: day
admission now needs **all three streams** — bookTicker, trade **and depth20** —
because *"E2.0 needed two streams, E2-A needs three for depth-aware sizing."* The
admissible set is again **an OUTPUT** with the structural count as-of and the
gap-fraction leg explicitly not yet evaluated. ✓

**The 12 symbols are INHERITED BY NAME, not recomputed**, with the right reason:
*"Recomputing the top-40 ADV universe as-of a new date would change the population
and the number together, and then E2-A would not supersede E1-A — it would measure
something else."* ✓

**And the guard I did not expect:** `window_is_not_E1A_s` — E1-A ran 07-18..08-17
on Vision, the L2 collector started 08-19, **the windows do not intersect**, so
E2-A supersedes as the operative number **without re-measuring**, and *"a reader
must not difference the two and call it a queue-model effect."* **That closes a
misreading before anyone can make it.**

**The gate**: `eff_RT ≤ 8.0` under **RiskAverse** (pessimistic, binding),
ProbQueue-f3 reported as the optimistic end and never gated. Full T_p ladder
reported, gated only at 600 s — *"no patience shopping."* ✓

**The settle table has five states and the interval binds only on PASS:**
PASS needs point **and** CI-upper ≤ 8; FAIL needs neither;
`FAIL_BRACKET_DISAGREES` for a straddle, **never averaged**; two INCONCLUSIVE
states. And `why_the_interval_binds_only_on_PASS` credits it: *"This is E2.0's
reviewer finding 3 applied before it had to be filed twice."* ✓

**`REFUTES_THE_BRACKET` is a genuine self-falsification**: if ProbQueue-f3 costs
MORE than RiskAverse, *"the optimistic model cannot cost more than the pessimistic
one, so the INSTRUMENT is refuted and no overlay verdict is read."* **An instrument
that can declare itself broken rather than the symbol.**

**ICP**: skip-rate bar **0.50, declared now and applied to all twelve** — *"not an
ICP-shaped hole… if ICP clears it, ICP is read on the same footing"* — with the
aggregate reported **both ways**, as E1-A did (6.15 excl / 6.26 incl). ✓

## A.2 **ESCALATION (a) — hftbacktest absent. My recommendation: ACCEPT the direct implementation, with one addition.**

DA checked it (`ModuleNotFoundError`, timestamped), flagged the deviation **before
any run** with three named options, and refused to absorb it. **The cost is named
honestly:** *"the models' CORRECTNESS becomes mine rather than a library's."*

**Attacking the closed forms:**

* **ProbQueue-f3's ORIENTATION is pinned**, and I checked how: the positive control
  *"a resting order ALONE at the level fills on the first opposite-side trade under
  BOTH models"* discriminates, because at `front = 0` the declared form
  `f(front)/(f(front)+f(back))` gives **0** while the inverted reading gives **1**.
  A model that never fills an unqueued order fails that control. ✓
* **RiskAverse is pinned in the never-fills direction** by *"a resting order behind
  depth larger than all subsequent volume must NOT fill."* ✓
* **FINDING — nothing pins either model in the INTERIOR.** Both controls are
  boundary cases where any plausible implementation agrees, and the ordering
  predicate is a **relative** check between two implementations by the same author,
  so it cannot catch a common-mode error. **The exponent is the specific exposure:**
  n = 3, n = 2 and n = 1 all agree at `front = 0` and at never-fills, and differ in
  between. **One hand-computed interior case pins it** — e.g. `front = 1, back = 2`
  gives `1/(1+8) = 1/9` under x³ and `1/(1+4) = 1/5` under x². That is one line and
  it converts "the plan's exponent" from a comment into a check.
* **And one deviation to state rather than fix:** DA's RiskAverse requires
  cumulative volume to exceed *"the depth resting at L at placement, **PLUS q**"* —
  your own size must also trade. **Whether hftbacktest's model includes that term I
  cannot verify here, because the library is absent** — but the direction is
  unambiguous: including it makes fills rarer and `eff_RT` higher, so it is
  **conservative for a PASS gate**. A PASS under DA's form is a PASS under a
  library that omits it; a FAIL might not be. **The declaration asserts "this is
  hftbacktest's RiskAverse"; it should say which part is the library's and which is
  DA's tightening.**

**Recommendation: ACCEPT.** Installing a dependency mid-programme is the larger
risk, the models are declared in closed form before any data, the ordering property
is computed, and the gate is the pessimistic end. **Ask only for the interior
control and the "+q" attribution.**

## A.3 **ESCALATION (b) — the XS rebalance notional. My recommendation: SOURCE IT, and if it cannot be sourced, take DA's refusal.**

DA's question: the per-symbol rebalance **notional** for the XS book, which *"must
come from the P-2026-001 capstone book's own construction, not from this tape and
not from a round number chosen here."*

**What each choice does to the reading:**

| choice | what E2-A then measures |
|---|---|
| **source it from P-001's book** | the section-2 question as asked: depth-aware sizes at the book's real rebalance notionals. Larger notional ⇒ worse fills ⇒ higher `eff_RT` ⇒ **harder to pass**. **The only choice that yields an E2-A verdict.** |
| **choose a round number here** | the gate becomes a function of a number picked in this seat — rule 11's shape exactly. **The declaration already refuses this**, correctly. |
| **refuse the size-aware arm** | E2-A reports **min-size only**, which is *"exactly the E1-A answer with a better fill model"* — a real result about the queue model, **silent on size**, and correctly labelled NOT the E2-A gate. |

**DA's fallback is the right one and is already wired** (`an absent rebalance
notional must REFUSE the size-aware arm rather than defaulting to min-size and
calling it E2-A`). **My recommendation: ask P-001 for the number before the run;
if it is not available, run the min-size arm under DA's label and do not call the
overlay resolved.**

## A.4 The one v3 item

**The interior control for the queue models (§A.2), and the "+q" attribution.**
Everything else — falsifiers both directions at every state (8 known-bads, 5
positive controls, including the declaration-sha refusal, the canonical-root
refusal and the E1-A reproduction miss), the inherited reproduction control pinned
to `e1a_gate_summary.csv` row `tp_s=600` with all four numbers and both CIs, and
resources under the wrapper with the never-raised cap — is present.

---

# (B) The resolver adoption — reproduced

**`e2_0_true_mid.py` now resolves through `de_data_root.require_canonical`,
imported** (`:1141-1142`), and `ROOT = _resolve_root()` at `:74`. **The gap I filed
in the E2.0 result review is closed.**

**I reproduced the partial-root falsifier independently.** I built a scratch root
with **every directory present** and **36 real bookTicker files — 2 of the ledger's
19 day-prefixes** — symlinked to the real tape, and ran `--run --symbols ADAUSDT`
under `PM_DATA_ROOT`:

```
rc: 1
DataRootRefused: REFUSED: P-2026-002 E2.0 result-bearing emission ... the resolved
data root is None, not /home/yuqing/ctaNew/data (branch 1_env_PM_DATA_ROOT,
PM_DATA_ROOT='/tmp/tmpg0wk45r9')
```

raised at `run():1167 -> require_canonical_root():1142 -> DR.require_canonical():173`
— **the first thing `run()` does, before any day is read.** ✓ And DA's own
partial-census falsifier at `:1021-1050` carries the reason in my words: *"a PARTIAL
shell would pass that and yield a silently smaller population."*

**And all three of my E2.0 findings are wired as driven checks**, with the names I
asked for:

```
FINDING 1 WIRED: Delta_rs takes the INTERSECTION pair -- "so the population
                 difference cannot leak into the mid difference"
FINDING 2 WIRED: 2.0 bps reads NOT_KILLED_PENDING_GATE_1, not ALIVE -- the 1.8-2.3
                 band "has its own name end to end"
FINDING 3 WIRED: a point of 2.5 with CI-lo 1.0 reads INCONCLUSIVE -- and the
                 POSITIVE CONTROL shows the interval rule can ADMIT
```

**Both directions on finding 3**, which is what makes it a rule rather than a veto.

**The supersession of the earlier smoke receipt** is a **sidecar**,
`…044234Z.superseded_by.json`; the receipt itself carries no `superseded_by` field.
**That is the correct shape** — rule 13 forbids editing the superseded artifact, and
a conventionally-named sidecar is resolvable programmatically without touching it.
**The residual is that the convention is not written down anywhere I have seen.**
A reader resolving `044234Z` learns nothing from within it and must know to look for
`<artifact>.superseded_by.json`. **One line in SEAT_PROTOCOL would make the
convention findable**; it is a documentation gap, not a defect.

**And the sidecar's CONTENT is better than a pointer — it is a computed
supersession.** It carries `all_gate_bearing_fields_identical: true` with a
per-field breakdown — `admissible_days`, `cells`, `delta_rs_bps`,
`population_at_tau_star`, `primary_ci95`, `tau_star_s`, `verdict`, all true —
plus both paths, both sha256s and the carrying commit. **That independently
confirms what I computed myself in the result review**: the two receipts differ
only on the excluded in-progress day, and no gate-bearing number moved. A
supersession that PROVES nothing moved is a different object from one that
asserts it.

---

## Verdict

**(A) E2-A: APPROVED to run once the two escalations are ruled.** My
recommendations: **accept the direct queue-model implementation** (ask for one
interior control and the "+q" attribution), and **source the rebalance notional
from P-001 before the run**, falling back to DA's labelled min-size arm if it
cannot be had. **v3 needs only the interior control.**

**(B) The resolver adoption is complete and reproduces.** The convention for
supersession sidecars should be written down.

---

## CONTEXT

Far below the 80% reset threshold.
