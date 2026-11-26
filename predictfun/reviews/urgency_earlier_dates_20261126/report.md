# Earlier-date replay of the retained urgency controller

These three dates were selected before scoring from the existing unreserved tapes. The capacity lookup used October 24 and 25. Applying it to October 23 uses a later fit. October 24–25 share training-day information, including direct overlap with the fit manifest. Read these as retrospective stress results, separately from the six previously reported dates.

| UTC date | Complete / scheduled | Original ladder | Urgency settlement | Completed FIFO | Residual | ¢ / matched share | Mean abs. ending / ceiling (shares) | Criteria |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20261023 | 70 / 288 | −$4,101.70 | +$1,642.56 | +$1,985.24 | −$342.67 | 2.06 | 30.1 / 31.1 | pass |
| 20261024 | 81 / 288 | −$2,331.33 | +$1,592.92 | +$2,230.77 | −$637.85 | 2.39 | 31.5 / 21.8 | miss: within inventory limit |
| 20261025 | 72 / 288 | −$3,062.37 | +$3,524.46 | +$3,996.90 | −$472.44 | 5.26 | 31.3 / 39.4 | pass |

## Capacity training overlap

| Fit date | Markets directly used in fit | Settlement there | Other complete markets on same date | Settlement there |
|---|---:|---:|---:|---:|
| 20261024 | 23 | +$365.10 | 58 | +$1,227.82 |
| 20261025 | 23 | +$872.02 | 49 | +$2,652.44 |

## Coverage and verification

- 20261023 — complete: 70, missing_endpoint: 187, stale_future_book: 31.
- 20261024 — complete: 81, missing_endpoint: 137, stale_future_book: 70.
- 20261025 — complete: 72, missing_endpoint: 116, stale_future_book: 100.

As of November 26, 2026. Missing or stale markets have unknown hypothetical P&L. Settlement includes both completed FIFO trades and holdings through the recorded payout. Both arms were reproduced on a previously checked market for each signal archive. The resulting paired ledgers matched the source tape identities, and the independent CSV check reconciled each daily result and refused a $1 corruption. One wrapper import failed before the run and was corrected before scoring. No new strategy variant or fees. No untouched November block was opened by this test.

[Frozen protocol](protocol.md) · [Import correction](amendment.md) · [Results and market-level breakdowns](result.json)
