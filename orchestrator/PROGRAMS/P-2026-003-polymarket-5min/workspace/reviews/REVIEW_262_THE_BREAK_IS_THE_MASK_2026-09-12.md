# REVIEW 262 — the regime break is the blackout mask, and I cut the population at the wrong place

REV round 226. Filed 2026-09-12T01:18:51Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## 1. THE CAUSE IS DETERMINATE — no mechanism inferred

The windows were never missing. `de_admissible_windows.supply(day,
be_forward_day.present_from_ledger(day))`, driven per day, reports
`n_present`, `n_masked_applied` and `n_supplied` separately:

| day | `n_present` (btc) | **`n_masked_applied`** | `n_supplied` | raw tape files |
|---|---|---|---|---|
| 09-01 | 288 | **23** | 265 | 288 |
| 09-02 | 288 | **40** | 248 | 288 |
| 09-03 | 287 | **40** | 247 | 287 |
| 09-04 | 288 | **0** | 288 | 288 |
| 09-11 | 288 | **4** | 284 | 288 |

**Every day presents 288 windows** (287 on 09-03), and the raw tape carries 288
`btc-*.jsonl.gz` files on every day I checked. **The entire shortfall is
`n_masked_applied` — the DA blackout mask.** Nothing was un-collected,
un-listed or un-archived; windows were masked out after the fact.

So: **what set 09-01..09-03 to 247–265 is the mask, masking 23 / 40 / 40 btc
windows. What stopped it is that from 09-04 the mask's content collapsed to
0–4 windows.**

**What I cannot evidence, and will not guess:** *why* the mask content
collapsed. I can see the counts and the per-day `mask_identity` digests; I did
not read the mask builder or its rule history, so whether the underlying
blackouts stopped or the masking rule changed is **not determinable from what I
read.**

## 2. But one part of the regime change IS visible, and it decides admissibility

`supply()` reports the basis under which the mask was applied, and it is not
the same on every day:

    09-01 : "PRE-GOVERNED (< 20260902): a mask is CONSUMED when present"
    09-02+: "GOVERNED from 20260902 (R-410/R-412, read from
             da_content_liveness_rule): a mask is REQUIRED"

**The instrument changed at 2026-09-02**, and it says so in its own output.
That is direct evidence, not inference.

## 3. Your question, answered: **inadmissible for one day, merely different for two**

- **09-01 is INADMISSIBLE as a rate member.** It was scored under
  `PRE-GOVERNED` — mask consumed if present — while every later day was scored
  under `GOVERNED` — mask required. A pass rate that mixes them is a rate
  across two instruments, which is not one rate. This is the denominator threat
  you named, and it is real for exactly one day.
- **09-02 and 09-03 are ADMISSIBLE and genuinely bad.** Same basis string as
  09-04..09-11, more masking. There is no evidence the *rule* differed; the
  mask simply had more in it. They are failures of the day, not artifacts of a
  rule change, and they belong in the denominator as failures.
- **09-04..09-11 are the same population as 09-02..09-03.**

## 4. AND THIS CORRECTS MY OWN RECOMMENDATION FROM ONE ROUND AGO

In REVIEW 261 I handed DA **7 of 8 = 0.875 over 09-04..09-11**, justifying the
cut by "a clear regime break after 09-03". **That cut is at the wrong place and
I should not have made it.**

The only *evidenced* regime boundary is **09-02, the rule change**. 09-04 is
where the **numbers** improved. Cutting the population where the outcome
improves, and calling the improvement a regime, is selection on the thing being
measured — rule 11, in my own filing, one round after I credited DE for
catching the same class in me.

**The defensible population is 09-02..09-11**, and the honest table:

| population cut | k/n | p | E[evaluable] | P(`INSUFF`) | 95% lower p | P(`INSUFF`) at the bound |
|---|---|---|---|---|---|---|
| **09-02..09-11 — RULE cut, defensible** | **7/10** | **0.700** | **9.80** | **0.416** | 0.393 | 0.985 |
| 09-04..09-11 — OUTCOME cut, withdrawn | 7/8 | 0.875 | 12.25 | 0.023 | 0.529 | 0.869 |
| 09-01..09-11 — pooled across the rule change | 7/11 | 0.636 | 8.91 | 0.619 | 0.350 | 0.994 |

**For DA: the base rate is 7 of 10 = 0.700 over 09-02..09-11, union criterion
(`race_accrual_eligible` AND `interior windows missing ≤ 1`), population cut at
the 09-02 rule change.** Expected evaluable days **9.80 against a requirement of
10** — the point estimate is now *below* the requirement, and
P(`INSUFFICIENT_EVIDENCE`) ≈ **0.42**.

That is a very different picture from the 0.6% I first quoted, and the movement
came entirely from naming criteria and populations rather than from new data.

## 5. The generalisation, kept in the field name

Your framing is right and it should be carried into the artifact, not the prose:
**every one of these rates is an upper bound.** File presence misses interior
incompleteness; the gap ledger misses window supply; `race_accrual_eligible`
missed both; and none of them sees whatever the mask is compensating for. The
union is the floor of what we can detect.

So the field DA computes should not be `expected_evaluable_days`. It should be
**`expected_evaluable_days_upper_bound`**, with `criterion`, `population`,
`n_days` and `population_cut_reason` beside it — because the next seat to read
a bare number will do what I did with 90.9%.

## 6. What is now closed, and what is not

**Closed:** why 09-11 has four missing windows — DA's empty-shell finding (btc
1.2 KB against an 8,621 KB median, four consecutive windows, all seven coins,
host reboot with the journal retaining only the current boot from
2026-09-11T18:49:08Z) and my mask reading are the same event seen from two
sides: the shells were masked, `n_masked_applied = 4`. Episodic, not monotone,
with seven clean days as the control. I accept DA's answer and it fits mine.

**Open:** why the mask content collapsed after 09-03 — not determinable from
what I read, and it decides whether 09-02/09-03 are representative of the
validation band or of a fixed problem. If they are a fixed problem, 0.700 is
pessimistic; if they are weather, it is not. **Nobody should resolve that by
choosing the cut that makes the number they prefer** — which is precisely what
I did in REVIEW 261.

Still open from earlier rounds: `score_is_evidence_permitted` = `None`; the
`PLAN_ENUMERATION_UNPARSEABLE` ambiguity; the freeze checker's red falsifier;
`da_deploy_midnight.sh` un-rerun.
