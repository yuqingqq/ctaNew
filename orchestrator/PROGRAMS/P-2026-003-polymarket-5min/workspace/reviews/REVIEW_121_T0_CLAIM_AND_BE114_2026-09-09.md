# REVIEW 121 — the `t0` claim: verifying the measurement, not the claim; and BE 114's three

**REV, 2026-09-09T08:35Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, and **no book unpickled** — see §4.

**THE ANSWER, IN THREE PARTS.**

1. **What DE measured is NOT what the claim says**, and the two figures are not in tension —
   which corrects the dispatch's framing **and my own REVIEW 120**.
2. **The claim is not the whole predicate.** Even if it is perfectly true, the three
   consumers are still exposed, by a second mechanism I drove.
3. **It is NOT settleable without a corrected book**, and here is precisely why — so I
   stopped rather than take the lock or unpickle 450 MB, and I name the one number the first
   corrected book should carry so it closes by measurement.

**And BE 114's three rule-28 fixes all verify, each with a green control.**

---

## 1. THE MEASUREMENT, VERBATIM, AGAINST THE CLAIM IT IS CITED FOR

**Measured** (`day_run.scoring_timing.measured_incidence_on_2026-09-04`):

> "15,867 of 40,000 **sampled rows** (39.7 %) begin strictly after their generation's start,
> across 6,586 of 24,133 **generations** (27 %); max 60 rows in one generation"

**Claimed** (`de_phase4_diag_runner`, the fixture comment):

> "`t_start` and `gen_t0` are ONE clock, a generation's **FIRST row** starts exactly at its
> `t0`, and NO row precedes its generation."

**These are different quantities, and neither figure bears on the claim.** The measurement
counts *rows that begin after their generation's start* and *generations having at least one
such row*. The claim is about *first* rows — `min(t_start) == t0`.

**They are also not in tension, and this is the part worth fixing in the record.** A
generation whose first row is at `t0` and whose second and third rows come later **satisfies
the claim and contributes to both percentages**. In fact, if the claim is true, then *every*
multi-row generation contributes to the 27 %. **So the 27 % is the count of multi-row
generations, not the exposure** — and my own REVIEW 120 sentence, "up to the 27 % of
generations DE measured as having rows after their start", **overstates it. I withdraw that
number as an exposure estimate.**

**So: the measurement DE cites does not support the claim it is cited for.** That is not an
accusation that the claim is false — it is that the evidence offered is about something
else, and the claim is still unverified, exactly as REVIEW 105 said.

## 2. AND THE CLAIM IS NOT THE WHOLE PREDICATE — a second mechanism, driven

The three consumers need `(slug, side, float(g["t0"])) in gs`, and `gs` is keyed by the
**kept** rows' `t_start`. Two independent things break that key:

- **(i) the first row is late** — what DE's claim addresses;
- **(ii) the first row is at `t0` and the FEATURE PASS DROPPED IT.**

Driven:

```
all three rows kept : keys [100.0, 103.0, 106.0]  ->  (s1, B, 100.0) in gs = True
the t0 row DROPPED  : keys [103.0, 106.0]         ->  (s1, B, 100.0) in gs = False
```

**The generation counts as UNSCORED although its first row was at `t0` exactly as claimed.**
Mechanism (ii) is independent of DE's claim and is **guaranteed to occur at some rate**,
because dropping rows for named reasons (`pm`, `fine`, `state_join_failed`) is what the
feature pass does and what `PARTIAL_ROWS` exists to count.

**Therefore verifying DE's claim, even if it came back perfect, would not close the
exposure.** That is the substantive answer to why this cannot be settled by checking one
sentence.

## 3. WHAT WOULD SETTLE IT, AND WHY I STOPPED

The exposure is one number: **for each head, how many reference generations have NO key at
their own `t0` in the corrected assembly.** It covers both mechanisms at once.

Every source for it is on the far side of the boundary DA flagged twice tonight:

- the **feature pass** — the heavy path, and the lock;
- the **tape index** (`PA.tape_index`) — the same order of work;
- a **book's `rows`** — a ~450 MB unpickle, the boundary DA declined twice and was right to.

And the cheap sources cannot answer it **in principle, not merely in practice**: *every book
on disk predates `c501824`* (newest Sep 8; the causal scoring landed 09-09T04:02), so its
assembly is keyed at `t0` **by construction**. Asking a pre-causal assembly whether rows
start at `t0` is asking a table that discarded the row times. **The decision ledger cannot
answer it either**: its `DECISION` rows are the above-theta stream, a filtered population, so
`min(decision t)` is an upper bound on `min(row t_start)` and can only fail to falsify.

**So: not settleable without a corrected book, and I stopped.** No lock taken, nothing
unpickled.

**THE ONE NUMBER TO DEMAND OF THE FIRST CORRECTED BUILD**, so this closes by measurement and
not by another review:

```
per head:  n_generations_with_a_key_at_their_own_t0   vs   n_covered
```

The gap **is** the exposure for `da_elementwise`, `da_elem_grid` and `da_elementwise_hz`, it
covers mechanisms (i) and (ii) together, and it is one line at build time beside the
coverage block that already exists. **This makes the first corrected book the artifact that
settles it, exactly as REVIEW 120 argued.**

## 4. BE 114's THREE — ALL VERIFY, EACH WITH A GREEN CONTROL

| fix | driven |
|---|---|
| **(1)** the supply must name its day | green (supply names 20260903, used for it) → **OK**; supply for another day → **`SUPPLY_IS_FOR_A_DIFFERENT_DAY`**; supply naming no day → **`SUPPLY_DOES_NOT_NAME_ITS_DAY`**; and through `mask_block`, **`MASK_SUPPLY_IS_FOR_A_DIFFERENT_DAY`** |
| **(2)** `MASK_COUNTS_ABSENT` | green (`n_present`/`n_masked_applied`/`n_supplied` all present) → **OK**; one count missing → **`MASK_COUNTS_ABSENT`** (previously `closes = None` and three nulls were emitted); arithmetic not closing → **`MASK_ARITHMETIC_DOES_NOT_CLOSE`** — **the two failure modes are distinguished by name, which is the whole point of the polarity fix** |
| **(3)** the selector's second return | **zero** discarded-selector-return sites remain in `be_daybook_build` |

*My first pass at (2) had no green control — I built the `counts` dict with the wrong key
names and every case refused, which is exactly the "refusal that always fires" I would flag
in someone else. I found the expected shape and re-drove it; the green above is that
re-drive. Recording it, because a verification without a green control is not one.*

## 5. ROUTED

1. **BE / DE — the build should emit
   `n_generations_with_a_key_at_their_own_t0` per head**, beside `n_covered`. One line, and
   it retires this question permanently.
2. **DE — the receipt's `measured_incidence_on_2026-09-04` should not be cited for the
   first-row claim.** Either measure `min(t_start) == t0` and report it, or drop the claim
   from the fixture comment; today the comment asserts more than the receipt measures.
3. **Coordinator — my REVIEW 120's "up to 27 %" is withdrawn as an exposure estimate.** The
   exposure is unmeasured, not 27 %, and §3 says what would measure it.
4. **BE 114 verifies on all three.**
