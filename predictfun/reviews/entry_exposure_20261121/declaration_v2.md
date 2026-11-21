# ENTRY-EXPOSURE-v1 — freeze 2b: the override readings superseded in band (committed before any v2 output)

Session Claude aea5bfb9, 2026-11-21. It amends [declaration.md](declaration.md) (freeze `6ca42d25`), which stays in force for everything
not named here. The frozen outputs stay as provenance:
- `exposure.json`, `forks.csv`, `summary.csv`, `reasons.csv`;
- `checks.json` and `checks_falsifier.json`.

## What the frozen checks found

Check C (`exposure_checks.py`) failed on 424 of the 756 ok forks, and every failure carried one message: "KEEP: the shadow log is not its
overrides". The other checks (A, B, D, E, F, G) were true. The falsifier's C cases had a false clean case, so they could not pass.

## Diagnosis (evidence in the logs)

- **The defect.** `pf_entry_exposure.KeepInitial.decide` increments `overrides` on each override. The causal probe calls `decide` again on
  its hidden and delivered copies, every 35th callback; the decision log records only the branch's own callbacks. So the counter also
  counts the probe's calls.
- **The bound holds on every row** (`logs/exposure_c_diag.log`): 0 ≤ counter − log ≤ 2 × the KEEP arm's probe-checked calls on all 756
  forks, and the counter exceeds the log on exactly the 424 failing ones.
- **The rerun** (`tools/probe_counter_check.py`, `logs/exposure_probe_counter_check.log`) took one failing fork, Oct 24 m 3695609 side 0.
  - Probe off: the counter is 98, which equals the logged 98.
  - Probe on: the counter is 104, which is the saved value.
  - Both ways, the KEEP arm's fills, decisions and value are identical to the saved arm. The probe never changes the branch.

## What it changes, and what it cannot

**Affected readings:**
- the per-fork `overrides`;
- `overrides_forks` in the summary;
- the `keep_baseline_equal` shortcut, which skipped the equality test whenever the counter was positive. The log-based prefix rule covered
  every fork without a logged override: all 756 `prefix_ok` are true, meaning identical decisions and fills.

**Unaffected, because none of them reads the counter:**
- values, fills, decisions and components;
- the contrasts, the predicates and the diagnosis;
- every status except a possible `keep_baseline_mismatch` under the v2 flag.

**Disclosure.** The frozen outputs, including the diagnosis, were read before this amendment. Check H (below) therefore requires that the
supersession changed nothing else:
- the frozen and v2 `forks.csv` must be equal in every other column and row, with the frozen counter kept as `overrides_counter_v1`;
- the v2 predicates must equal the frozen ones exactly.

## The supersession

| File | Role |
|---|---|
| `predictfun/research/pf_entry_exposure_summary.py` | summary v2 |
| `predictfun/research/pf_entry_exposure_summary_selftest.py` | its selftest |
| `predictfun/research/pf_entry_exposure_summary_mutants.py` | its mutants: 11 / 11 caught by a check (`logs/mutants_sum`) |
| `exposure_checks_v2.py` (this directory) | the v2 checks; their selftest passes (`logs/exposure_checks_v2_selftest.log`) |

**Summary v2**, computed from the saved partitions:
- `overrides` is the logged override decisions;
- `keep_baseline_equal` comes from the saved ledgers when no override was logged;
- the frozen flat, summaries, reasons and predicates are otherwise unchanged.

Its outputs go to `data/.../entry_exposure_v1_20261121/v2/`, under the frozen file names. That directory also holds the manifest's byte
copy and the linked partitions. An existing v2 summary is refused.

**Checks v2:**
- the frozen checks A–G run on the v2 outputs, except that check C's shadow rule is split:
  - (i) every logged override is of the initial gid with a shadow PULL;
  - (ii) the v2 count equals the log;
  - (iii) the frozen counter exceeds the log by between 0 and 2 × probe-checked;
- added: H (the override readings, the supersession fields, and nothing else changed) and G2 (the v2 suite);
- the falsifier adds an overrides corruption (H), a counter below the log (C) and a copied v2 mutant record (G2);
- outputs: `checks_v2.json`, `checks_falsifier_v2.json`.

The receipt reports the v2 outputs. The frozen module, checks and manifest are unchanged. The budget stays 2 core-hours (check F, recursive).
