# TOP-3 MAKER LOGIC CHAINS v7 — corrections of v6 (review 0742bf95e); this same-data line is closed

2026-11-23, session aea5bfb9.

**What this is:** an in-band correction receipt (Reliability rule 13). There is no new computation.
- Every number comes from the frozen v6 outputs (`v2/chains/{competing_transition,continuation_econ}/claims.json`), or is a recount by review 0742bf95e and is attributed to it as such.
- v6 stays as provenance.

## Corrections

1. **The measured event is a same-side improvement / response, not exact restoration** (finding 1).
   - The v6 risk differences (+14.8 pts in the last 30 s; −9.7 pts when the own best level was thin) and "26%" refer to all **999 same-side responses**: 753 restoring + 139 partial + 107 overshoot = 26.3% of 3,796.
   - **Exact restoration is 753 / 3,796 = 19.8%.**
   - The economics are **pooled exact FIFO entry-close attribution of candidate response orders**: 23 available orders, −$9.65 through END.
   - **The review's post-hoc subtype recount** (not a declared subgroup test): 21 exact-restoring orders −$1.51; 2 overshoot orders −$8.14. Attaching the pooled loss to "restorations" was misleading.
   - A small observed aggregate loss does not show that restoring quotes cannot earn. Creator identity, causal response and a counterfactual quoting policy are unidentified.
2. **BTC wording** (finding 2). The v6 §6 sentence "BTC and fair-value conditions give no consistent direction" contradicted its own table and is **withdrawn**.
   - **Correct:** BTC-with and fair-gap are inconsistent.
   - **BTC-against** is sign-consistent on the matched population: +3.10 pts D0 / +3.14 pts LAG, positive on all 4 days at both clocks. Its caveats:
     - the unmatched comparison has the opposite sign: 22.19% vs 26.57% on 392 / 3,146 episodes, against a matched 336 / 767 with harmonic weights. These are different populations and weightings, not an error;
     - the largest market supplies 59.7% of the net matched RD, but only 21.0% of the absolute market contributions.
   - No causal BTC trigger is established.
3. **The limit belongs to these records** (finding 3).
   - **Correct:** *these saved aggregate records leave most maker identities, post / cancel times and queue histories tied across compatible explanations.* This is a limit of the available information, not proof that no public source could identify them. More days of the same format cannot reconstruct an absent owner-tagged cancel message.
   - The frozen stop predicate `all_thin_or_inconsistent` is false (three contrasts are consistent). The two inconsistent contrasts end their branches, and partial consistency authorizes no further variant.
4. **Continuation scopes and conventions** (finding 4):
   - **Take and close scopes:**
     - the "375 takes / 12 FIFO close keys / 113.8 sh" are for the **union of the 89 response candidates and the six chains** (de-duplicated);
     - response-only (review recount): 215 takes, 9 close keys, 91.57 sh;
     - chains only: 204 takes, 3 keys, 22.22 sh.
   - **"Residual at END"** = final FIFO remaining quantity valued at the inherited economic terminal mark, observed at or before we − 2 s, with execution cutoff at we. There are zero post-END legs or closes in this packet.
   - **The 10 s / 60 s windows** are entry-fill-to-close settlement-clock durations and are cumulative. Their equality with all within-window closes is a property of this sample.

## What the action-chain line established (descriptive, 4 consumed days; grades as in v3–v6)

| Action | Observable association | Grade |
|---|---|---|
| Take | Large inventory roughly halves the seconds containing same-direction takes. Opposing takes follow own-fill bursts. Taking falls sharply in the last 30 s. The inventory-size trigger is withdrawn. | robust / timing-sensitive / robust |
| Pull (cancel-compatible endings) | Adverse BTC in the last second, under both references | timing-sensitive; pull vs exhaustion tied |
| Post (fresh levels, market layer) | New levels follow wide spreads and thin best levels; deep placements surge near END | supported associations; 96.5% of creators unknown |
| Reprice after a one-side retreat | The next detected change is usually the other side improving. Same-side response is more likely in the last 30 s (+14.8 pts) and less likely when the side was thin (−9.7 pts). | adequate, consistent, 4 / 4 days |
| Queue front | Front holders not certifiable; front-win share unidentified | unidentified |

## Closed and open

**Closed:** the same-data top-maker command-identification loop.

**Open, as a USER decision** (review 0742bf95e's ONE next priority): an *attributed order-lifecycle evidence gate*. That means a maker-attributed lifecycle source with stable order IDs joinable to fill hashes, covering add / accept, partial fills, cancel / amend / replace and terminal status, including unfilled and cancelled orders.
- Own-order logs can validate reconstruction accuracy for our own actions only, not a competitor's intent.
- Untouched later days can validate the observable associations, which is a different question.
- If no such source exists, the line stays closed with the present associations.
