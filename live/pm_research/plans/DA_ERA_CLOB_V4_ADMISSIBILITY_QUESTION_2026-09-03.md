# DRAFT FOR USER RULING — `ERA_ADMISSIBLE["clob_v4"]`

**Status: DRAFT-FOR-USER. No seat may resolve this.** Drafted by DA
2026-09-03, routed by the coordinator per R-497 (F)(1)'s implementation round.
Nothing in this file changes any value; the table is untouched pending the
ruling.

**The question, in one sentence.** `ERA_ADMISSIBLE["clob_v4"] = False` carries
the cite `R-340`. R-340 exists. It does not mention `clob_v4`. Should the entry
keep its value with a cite that names something else, be re-cited to a ruling
that does name it, or be re-ruled under R-497 (F)(1)?

This is the same defect the USER already ruled on one entry over. `clob_v3_1`
read `# pre-O1` — no cite at all — and was quoted in this programme's own
register as R-232-ruled; R-232 contains zero occurrences of `clob_v3_1`. The
repair shipped in round 22 was the invariant, not the value: every entry now
carries its authority as DATA, an entry with none REFUSES, and the cite travels
into the emitted verdict. Applying that invariant uniformly is what surfaced
this entry and `clob_v5` (which has no ruling at all and now refuses).

---

## 1. What the cite SAYS

R-340 is 389 characters. It is quoted here in full, because its brevity is part
of the finding:

> **R-340 — 2026-08-31T03:16Z — coordinator — USER RULING (structured answer):
> v5 deploys MID-DAY TODAY, after the three blockers close and a Codex pre-arm
> review clears** — declared instant **2026-08-31T07:00:00Z**, recorded BEFORE
> execution per the filing's own requirement; work split: DA the era-admission
> guard AS CODE, coordinator the identity-bound v5 preflight/postflight/runbook

It is a real USER ruling, and it is the ruling that authorised DA to build the
era-admission guard at all.

## 2. What the cite DOES NOT say — counted, not characterised

Measured over R-340's full text:

| token | occurrences in R-340 |
|---|---|
| `clob_v4` | **0** |
| `admissib` | **0** |
| `accru` | **0** |
| `clob_v3_1` | **0** |

R-340 rules a **deploy instant** and a **work split**. The table's comment —
*"O1 package; ruled never admissible post-O1 (R-340)"* — states a conclusion the
cited entry does not carry. The entry is not unattributed the way `clob_v3_1`
was; it is **mis-attributed**, which is a different failure and arguably a
harder one to see, because the cite resolves.

This is computed and re-computable: `pm_tape_density`-style, the checker is
`da_forward_day_verify.era_authority_audit()`, which reports per era whether
each cited `R-<n>` **resolves** in the register and whether the cited entry
**names that era**. For `clob_v4` it returns `entries_resolve: {R-340: true}`
and `entry_names_this_era: {R-340: false}`.

**That audit REPORTS and does not enforce, deliberately.** The enforced
predicate (`era_authority_for`) requires a cite to be PRESENT and to name a
decision; `clob_v4` passes it. Making the audit enforcing would move an
admissibility value on a seat's reading of prose, which is the exact defect
this whole repair is about. A check (DA22-A7c) pins the disagreement so that if
anyone later makes the audit enforcing, the suite goes red rather than
absorbing the change.

## 3. What the runner would DO under each answer — computed on the real ledger

**The decisive fact first, because it makes this ruling cheap.** The `clob_v4`
era ran **2026-08-30T05:30:02.114727Z → 2026-08-31T22:00:02.274534Z** — 40.5
hours — and **no complete UTC day lies inside it**. Both days that touch it are
era-MIXED and fail the purity conjunct before admissibility is consulted:

| day | eras touched | pure | `race_admissible_by_era` with `clob_v4: False` | with `clob_v4: True` |
|---|---|---|---|---|
| 2026-08-30 | `clob_v3_1` + `clob_v4` | no | **False** | **False** |
| 2026-08-31 | `clob_v4` + `clob_v4_1` | no | **False** | **False** |

**Days whose verdict the answer changes: 0.** Not "0 so far" — the era is
CLOSED, its successor boundary is stamped, and no new `clob_v4` day can be
created. Verified at `data/pm_5min/collector_runs.jsonl` and recomputed through
`day_era_admission` for every day on the tape, n = 16 days, as-of
2026-09-03T06:46Z.

So all three answers below produce **identical day verdicts**. What differs is
what the table says about itself, and what the next reader is entitled to
conclude from it.

### (a) KEEP `False`, RE-CITED

The entry stays `False` and its authority becomes a cite that actually rules
it — either an existing entry that does, or a new USER ruling recorded now.

* **Runner behaviour:** unchanged. 08-30 and 08-31 still refuse on purity.
* **Table state:** `entry_names_this_era` becomes true; the audit's
  `eras_whose_cite_does_not_name_them` empties.
* **What it asserts:** that `clob_v4` is inadmissible *on its merits* — the O1
  package changed gap DETECTION and LABELLING (O1b/O1c/O1d), and R496-R7
  measured that `clob_v4` is the **outlier BETWEEN** `clob_v3_1` and
  `clob_v4_1`, which are identical in fields, stamping and keepalive.
* **Cost:** none mechanically. It records a judgement about a 40-hour era that
  can never carry a day.

### (b) ADMIT, under R-497 (F)(1)

*"We check the data quality and only use qualifiable data"* is read as reaching
every era, so `clob_v4` becomes `True` with R-497 (F)(1) as its cite, exactly
as `clob_v3_1` now is.

* **Runner behaviour:** unchanged. 08-30 and 08-31 still refuse on purity —
  driven, not assumed (DA22-A3b drives 08-30 with BOTH eras admitted and it
  still refuses on `boundaries_inside_day`).
* **Table state:** the admissible set becomes *"quality decides, era is an
  interlock against an UNRULED collector"* with no exception, which is what
  `ACCRUAL_RULE`'s own text already says.
* **What it asserts:** that era carries no fidelity claim at all, and the only
  thing the era conjunct still does is refuse a collector version nobody has
  ruled. `clob_v5` would remain refused, which is the interlock working.
* **Cost:** none on this tape. The stated risk is prospective: if a future
  collector version is later judged to have damaged the data, the precedent
  says quality alone decides, and the argument would have to be made on
  quality.

### (c) NO RULING (the status quo, stated so it is a choice and not a default)

* **Runner behaviour:** unchanged.
* **Table state:** the entry keeps a cite that resolves and does not name it.
  `era_authority_for` admits it, the audit reports it, and the next reader who
  follows the cite finds a deploy ruling.
* **What it asserts:** nothing — and that is the problem the round-22 invariant
  exists to make visible rather than to fix by fiat. It is a legitimate answer
  as long as it is CHOSEN; the defect was never the value, it was a value
  standing where a decision should be.

---

## 4. What DA has done and has deliberately not done

* **Done:** reproduced the defect at the artifacts (R-340's full text and its
  four zero counts, above); left `clob_v4`'s value and cite EXACTLY as shipped;
  built the audit that computes the discrepancy; pinned the enforced/reported
  disagreement with a check so it cannot be quietly resolved in code.
* **Not done, and not mine:** moving the value. The dispatch that implemented
  R-497 (F)(1) named `clob_v3_1`. Extending a ruling to a second era on a
  seat's reading of the ruling's wording is precisely the act that produced the
  `# pre-O1` entry in the first place (rule 14, SEAT_PROTOCOL 9).

**Recommendation, offered as a recommendation and nothing more:** (a) or (b),
because both end the state where a cite exists and names something else, and
the choice between them is a judgement about `clob_v4`'s merits that costs no
day either way. DA does not prefer one; the merits belong to whoever weighs
R496-R7's measurement of the O1 package.
