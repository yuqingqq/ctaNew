# ENTRY-EXPOSURE-v1 — freeze 2c: check H's counter comparison made NaN-aware (committed before any v3 output)

Session Claude aea5bfb9, 2026-11-21. It amends [declaration_v2.md](declaration_v2.md) (freeze 2b `5ece0c98`) for one comparison only.
`checks_v2.json` and `checks_falsifier_v2.json` stay as provenance.

## What the v2 checks found

On the v2 outputs, checks A–G and G2 were true. Check C passed under the split shadow rule: 0 failing forks, and the delivered-copy
control fired. Check H was false with one message: "the v2 forks.csv differs from the frozen one outside the override readings". Because
H's clean case was false, the v2 falsifier could not pass.

## Diagnosis

The fault is in the checker, not the data. `logs/exposure_h_diag.log` shows:
- no column differs between the frozen and v2 `forks.csv`;
- the 12 elementwise mismatches are exactly the 12 world rows, NaN in both tables;
- the NaN-aware comparison is equal.

H compared the frozen counter with its v2 copy (`overrides_counter_v1`) through `astype(str)` and `==`. Under pandas 3.0.1, `astype(str)`
keeps NaN as missing, so each NaN pair compares unequal.

## The fix

[exposure_checks_v3.py](exposure_checks_v3.py) differs from v2 in one comparison:
- **H's counter comparison** becomes `counter_equal`, which treats NaN as equal only to NaN in the same row. Everything else is
  `exposure_checks_v2` unchanged.
- **Selftest:** five hand cases, plus a positive control that the v2 failure reproduces under this pandas.
- **Falsifier:** the v2 cases, plus H_counter (one frozen-counter copy changed).
- **Outputs:** `checks_v3.json`, `checks_falsifier_v3.json`.

No output, summary or module changes.
