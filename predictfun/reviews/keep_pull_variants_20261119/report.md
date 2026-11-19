# One-day test: the five declared score variants against A+0 through the scenario bank — Oct 23, 2026-11-19

**USER:** "Delete void outputs, can we run the one day test now". Asked what the test should compare, the USER chose "Score variants vs
A+0". This is review a159116e's next priority: "a predeclared baseline-versus-candidate research comparison through the same consumed-day
protocol and scenario bank".

**What is tested.** Raw threshold rules on the causal fair price, not a learned model.
- The baseline is A+0, the declared fair-edge rule: pull when the order's fair edge falls below 0 c.
- The candidates are the five variants declared beside it on 11-17 (`pf_keep_pull.POLICIES`, multiplicity 6):
  - A+1 / A-1: the same rule with the threshold at +1 / -1 c;
  - B0.5 / B1 / B2: pull when the fair price has moved 0.5 / 1 / 2 c against the order since posting.
- The orders are the reference run's: one 30-sh JOIN order at the known best per counted state (every 5 s per window-side), held to
  END = +5 s, pulls decided every 50 ms from +100 ms.

**Values are local marks at END, in c, on one consumed day (G = 1: point values, no interval).** They are not P&L.

## Instrument

- `predictfun/research/pf_keep_pull_variants.py` (new). The scenario replay's day outputs do not depend on the policy: a pull only removes
  the keep branch's fills after its landing. So every policy is valued from the committed v6 day outputs with
  `pf_keep_pull_scenarios.outcome(..., policy=p)`, with no kernel re-simulation. Readings and controls were declared in the module
  before any variant was valued in the bank.
- `pf_keep_pull_scenarios.py` v6.1: `outcome()` takes the policy as a parameter (default A+0; an undeclared one is refused). Day outputs,
  the report and A+0's values are unchanged (control V0a below).

| Gate | Result |
|---|---|
| Scenario replay v6.1: selftest; mutation suite (every mutant caught by a check) | pass; **46 / 46** (two new: the policy parameter ignored, an undeclared policy accepted) |
| Variants module: selftest; mutation suite | pass; **30 / 30** |
| V0a: A+0 reproduces the committed Oct 23 scenario report (v6, as of Nov 18 20:46:32Z) in every value of every scenario | YES |
| V0b: at B every policy's pulls and improvement equal the committed keep / pull run (`keep_pull_20261117`, M50 / 37 ms, Oct 23) | YES |
| V0c: at B every policy's E, improvement and pulls equal the clock audit's per-fill values (a second code path) | YES |
| V0d: KEEP, PULL-ALL, fills and eligible orders are the same for every policy; KEEP = the sum of its fills | YES |
| V0e: every day output matches its receipt; the scenario day was built on those keep / pull outputs | YES |
| Memory | peak 734 MiB (run 14.9 s) |

Every hash recorded by the run matches the committed sources and inputs ([sources.sha256](sources.sha256)).

## Results (Oct 23; 28,799 eligible orders at B)

Counts computed by [counts.py](counts.py) from the saved tables ([counts.txt](counts.txt)).

| Policy | Pulls at B | E at B | E at M250 | Scenarios with E > 0 | V at B | Scenarios with V > V(A+0) |
|---|---:|---:|---:|---:|---:|---:|
| A+0 (baseline) | 14,155 | +34,948 | -13,342 | 16 / 19 | -174,961 | — |
| A+1 | 16,187 | +29,847 | -10,786 | 16 / 19 | -153,379 | 18 / 19 (all but EVhi) |
| A-1 | 11,738 | +41,921 | -7,599 | 17 / 19 | -198,680 | 2 / 19 |
| B0.5 | 4,782 | +101,575 | -6,740 | 16 / 19 | -233,504 | 2 / 19 |
| B1 | 3,420 | +102,767 | -4,199 | 16 / 19 | -244,914 | 2 / 19 |
| B2 | 2,217 | +94,977 | -381 | 16 / 19 | -267,883 | 2 / 19 |

Columns:
- **E** is each policy's improvement over its own matched STATES null.
- **V** is KEEP plus the policy's improvement over KEEP (the rule's own local value).

The full tables are in [variants_20261023.txt](variants_20261023.txt):
- every scenario;
- without fills on orders < 100 ms old;
- C1 by age and B1a-weighted;
- C3 cancel-pending at B / L100 / the slow mixture;
- C12 by trade-anchor source.

The CSV tables are [variants_20261023_policies.csv](variants_20261023_policies.csv) and
[variants_20261023_compare.csv](variants_20261023_compare.csv).

**Readings (declared before any variant was valued in the bank):**

| Reading | Value |
|---|---|
| SURVIVES-TIMING (a policy whose E is positive in every tested scenario) | **none**: every policy's E is negative at S1lo and at M250; all but A-1 are also negative at S2lo |
| ABOVE THE BASELINE IN EVERY TESTED SCENARIO (D1) | **none** |
| D1 for A+1 | NOT-STABLE: V above A+0's in 18 of 19 scenarios, below at EVhi (-2,325) |
| D2 (E above A+0's in every scenario) | ABOVE-BASELINE for A-1 only; its V is below A+0's in 17 of 19 |
| P4 V - KEEP | ABOVE-KEEP in every scenario for B1 and B2; the other four policies are below KEEP at EVhi |
| P5 V - V(PULL-ALL) | NOT-STABLE for every policy; V is below PULL-ALL's in 17 of 19 scenarios for each |
| C2 E AT M250 vs M50 | SIGN-DIFFERS-AT-M250 for every policy |
| C1 B1a-WEIGHTED E AT B | HOLDS-SIGN for every policy |

**What these numbers support (protocol section 4).**
- On Oct 23, no declared variant survives the timing scenarios where A+0 changes sign. Every policy's E is negative at S1lo and
  M250, and all but A-1 are also negative at S2lo.
- The B rules have a much larger E than A+0 at B but a lower V, because they pull far less (2,217-4,782 pulls against 14,155). E
  compares a rule with its own matched null; V compares the rules' values.
- Pulling every order at its first live decision (PULL-ALL) has a higher local value than every policy in 17 of 19 scenarios.
- One consumed day, multiplicity 6: a candidate reading is a candidate only. None of this is P&L, order-level validated (C10) or robust to
  the queue rule (C5).

## Files

- **Outputs:** [variants_20261023.json](variants_20261023.json) (readings, every value, inputs and hashes), the text report and the two
  CSV tables, copied from `data/predict_fun/research/keep_pull_variants_20261119/`.
- **Logs:** [logs/](logs/): scenario v6.1 selftest and mutants, variants selftest and mutants, the Oct 23 run.
- **Protocol:** [test_run_protocol.md](../simulator_readiness_20261118/test_run_protocol.md) was updated for policy comparisons (gate V0a-V0e,
  the strongest comparison claim). Its order description was corrected: it is 30-sh JOIN orders every 5 s, not "one 5-sh order per side".
