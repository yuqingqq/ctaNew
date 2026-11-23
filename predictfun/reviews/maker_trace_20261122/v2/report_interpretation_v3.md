# MAKER-TRACE — interpretation v3 (answers review 8f2a2e753; supersedes wording in `report.md`, `report_interpretation_v2.md` and `report_closes.md`; no number changes)

2026-11-23, session aea5bfb9. Review [`20261123_maker_trace_v2_review`](../../automatic/20261123_maker_trace_v2_review/report.md), pinned 0e21193f2. It accepts the economics and the completed-close packet, and accepts the v2 lifecycle tie change as a declared scenario amendment.

## Finding 1 (P2, holds) — exit roles are not a policy prescription

**Superseded sentence** (`report_closes.md`): "closes that exit by taking lose, because they pay the spread and the taker fee".

**Replacement:** FIFO completed closes ending with a **maker** fill were profitable in this sample; those ending with a **taker** fill were negative. **The value of replacing either action is unmeasured.**

For the top 10's maker→taker closes:
- gross −2.65 c/share, fees 0.53 c, net −3.18 c;
- the rest of the loss is not split into entry selection, movement while holding and exit cost;
- a take may also prevent a larger later loss, and inventory that never closed is in the residual.

## Finding 2 (P2, holds; verified at `pf_mt_decision.py` `inventory`: `red = s * pos < 0`) — "takes reduce position" was mislabelled

The published statistic is an **opposing-side action** rate. An action that overshoots through flat counts as "reducing".

Actual |position| reduction, from the review's exact recount on the published action population:

| Rank / role | Non-flat actions | Opposing side (published) | Actually reduces \|position\| |
|---|---|---|---|
| 1 / maker | 152,352 | 77,753 (51%) | 74,550 (**49%**) |
| 1 / taker | 156,412 | 92,142 (59%) | 90,256 (**58%**) |
| 2 / maker | 112,615 | 55,750 (50%) | 54,380 (48%) |
| 2 / taker | 16,436 | 9,730 (59%) | 9,593 (**58%**) |
| 3 / maker | 104,788 | 57,847 (55%) | 55,746 (53%) |
| 3 / taker | 8,581 | 5,225 (61%) | 5,095 (**59%**) |

**Corrected statements:**
- A majority of each address's takes reduce absolute inventory (58–59%).
- Rank 1's maker fills do not (49%).
- The P(sell | long) / P(sell | short) figures quoted earlier are opposing-side rates.

`claims_decision.json`'s `takes_reduce_position` / `next_maker_fill_reduces_position` are read as "opposing-side action". They are not absolute-reduction claims.

## Finding 3 (P3, holds) — presentation

- **Identifiability fixture:** read it as "exhaustion of the original size vs a partial fill followed by an unseen cancellation". The executed amount is observed in that fixture.
- **`good_turnover_lost_to_residual`** means positive completed-turnover net AND a negative residual contribution. It does not mean negative total profit: ranks 1 and 3 net positive; rank 2 nets negative.
- **Reconciliation:** close net + residual excludes residual fees and fee-token drift, which `closes.py` carries (lines 523–529, 658–660). The "gives back 45% / > 100% / 70%" ratios are descriptive ratios of the two reported components, not an exact net identity.

## Finding 4 (still open, nonblocking)

The latent receipt-earlier-than-UUID timing case in `pf_mt_lifecycle.py` is carried as an explicit reuse limitation. It must be fixed, with a refusal, before any input with that pattern is admitted.

## Ruling recorded

- The v2 same-settle-ms tie change is accepted as a declared FIFO scenario amendment, not as proof that v1 was buggy.
- T2 5 / 16,433 remains a low-contradiction screen.
- R4 aggregate economics are unchanged: only 4 lifecycle annotation columns of `fills_valued` changed, in 164–519 rows.
