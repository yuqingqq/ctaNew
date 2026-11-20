# R2-C v2: the pi0_50_v1 in-sample value diagnostics, superseding report.md's value columns (review 8cc1dc49 P3) — 2026-11-20

**Supersedes** the in-sample value columns of [report.md](report.md):
- "V predicted / V realized (c, weighted mean)";
- `V_pred_wmean_c` / `V_real_wmean_c` in `checkpoint/diag.json` and `checkpoint/diag_horizon.csv`.

Those columns divided the coherent rows' weighted sums by the weight of **all** rows. That is neither the coherent-conditional mean nor the
all-row mean, because the excluded rows count as zero (review 8cc1dc49 P3: `pf_r2_teacher.py:495, 501, 504–510`).

**Unchanged:** every other reading of report.md and its checks; the freeze `769dff77...`; the 16 models; the fits; the rollout; the policy.
The original receipt stays as provenance.

**Wording correction (same review):** the complement of the short bin (0, 100] is "> 100 ms", not "≥ 100 ms". A decision exactly 100 ms
before END belongs to the short bin.

## The corrected estimands ([`pf_r2_teacher_diag2.py`](../../research/pf_r2_teacher_diag2.py), outside the frozen closure)

`diag2` verifies the freeze and loads the frozen models. Per side, head and horizon bin, weighted by the normalized census weights, it reports:
- **The coherent weight** W_c, beside the total weight W.
- **The coherent-conditional means:** predicted Σ_c w·V_pred / W_c and realized Σ_c w·V_real / W_c.
- **The full population's realized mean,** Σ w·V_real / W. This includes the rows where the teacher's forecast is incoherent and it falls
  back; that fallback is not treated as an economic zero.

**Its checks:**
- a known-answer selftest, in which the v1 mixed form demonstrably differs ([log](diag_v2/logs/diag2_selftest.log));
- **4 / 4** planted estimand errors caught ([log](diag_v2/logs/diag2_mutants.log)).

The run ([log](diag_v2/logs/diag2_run.log)): peak 1,140 MiB, 130 s. Its fill, conditional quantity / movement and severe columns, row counts and
incoherent counts equal v1's to within 3e-14, float round-off (computed over all 28 cells).

| Side / head (all horizons) | Rows | Coherent rows | W_c / W | V predicted, coherent mean (c) | V realized, coherent mean (c) | V realized, all rows (c) |
|---|---|---|---|---|---|---|
| bid POST | 327,341 | 327,228 | 327,225.0 / 327,341.0 | −9.176 | −9.351 | −9.348 |
| bid RETAIN | 2,353,692 | 2,352,355 | 2,352,273.4 / 2,353,692.0 | −9.848 | −10.127 | −10.122 |
| ask POST | 327,264 | 326,968 | 326,956.8 / 327,264.0 | −9.625 | −9.771 | −9.762 |
| ask RETAIN | 2,375,718 | 2,371,483 | 2,372,204.8 / 2,375,718.0 | −10.505 | −10.796 | −10.785 |

**The short bin (0, 100]** (coherent means, predicted / realized; all-row realized):

| Side / head | Predicted | Realized | All rows |
|---|---|---|---|
| bid POST | −0.354 | +0.073 | +0.073 |
| bid RETAIN | −0.212 | +0.016 | +0.016 |
| ask POST | −0.505 | +0.031 | +0.031 |
| ask RETAIN | −0.245 | +0.023 | +0.023 |

**The size of the correction:** at most 0.020 c in any cell between v2's coherent means and v1's mixed fields. The coherent weight is
≥ 99.85% of the total in every group.

These remain **in-sample training diagnostics** on consumed Oct 23: not skill, not a gate, no profitability claim.

## Files

- **Outputs:** [diag_v2/diag2.json](diag_v2/diag2.json), [diag_v2/diag2_horizon.csv](diag_v2/diag2_horizon.csv) (per side / head / horizon,
  plus the `all` rows).
- **Logs:** [diag_v2/logs/](diag_v2/logs/).
