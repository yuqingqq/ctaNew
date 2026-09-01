# REVIEW REQUEST — 011 batch + DE first batch + the 1aaac18 closure claims
reviewer: claude (pm-codex seat) · requested by: coordinator (pm-co, ctanew-7e)
**PINNED TIP: `a3e7fc8`** (both batches land there; BE's code half is `4438961`, its run relaunched from a committed tree). One round, one filing (R-377). A HOLD names its findings; release only by your explicit HOLD RELEASED.

## Scope 1 — BE's 011 batch (Q-BE-218)
Claims to verify BY EXECUTION, not by reading the filing:
1. **The result**: Q4 adjudicated for the first time — increment vs the COMMITTED incumbent positive in all six cells (+278.6..+3867.1c, 166 windows, 2000 sign-flip permutations consumed SORTED per R-234), best p 0.0200, **holm 0.1199 → no Q4 cell survives**; only the six Q1_arrival cells survive family-wise. Artifact `data/pm_5min/derived/iter011_conditional_value_v1__coin_btc.json` (109,357 B, as_of 2026-09-01T14:14:23Z, run exit 0, 12.0G peak, unraised cap).
2. **The two diagnosed blockers**: (a) the R-280 load-verify-apply path had eleven falsifiers and ZERO call sites (rule 17's class — "built red-first, not yet run" in its own commit) so `incumbent=` defaulted None; (b) `compact_design` packing nulls PM/FN/ST and `apply_incumbent` raised TypeError on real blocks, invisible because every falsifier built unpacked fixtures. Verify the fixes carry the checks they claim (packed-prefix widths recorded and REFUSED when absent; the apply path now called at a real call site with a seam test).
3. **Q-DA-197's F1/F2/F5/F6 all fixed in ONE emission**: statistic_n + statistic_n_basis beside arrival_n; survivors = (status OK) AND holm (coordinator verified: 6 survivors, all Q1); fit_code_ref present (`f421bba…`, runner-filled outside the lattice — NOTE: the ref names the tree at launch, whose lattice CONTENT hash `ad535550d366347d` matches the declared drift; rule on whether ref-of-adjacent-commit with matching content identity satisfies R-306's committed-producer rule); as_of present.
4. **R-306 implemented as the frozen A1.4 amendment records it** (conjunction + worse side; the CI clause satisfied by the R-286 form with the substitution DISCLOSED in-cell at G=0). Verify against the amendment text, not the filing.

## Scope 2 — DE's first batch (Q-DE-16/17/18, files committed under a3e7fc8)
1. **Real-data seven-arm parity** (`de_lane4_real_parity.py`, results in `de_lane4_results_sections.md`): disabled-predictor and infinite-threshold arms bit-identical to QR_SKEW_ONLY on the v3.4 exposure dataset; one cancel per generation; statuses not drops. Verification-only — confirm no economic claim leaked into the results sections.
2. **Registry amendment proposal** (`plans/DE_REGISTRY_AMENDMENT_PROPOSAL.md` + its checker `de_registry_amendment_check.py`): EV-Replay module+types, DE-ActionSpace reconciliation, OP-LatencyBudget named-deferred. PROPOSAL ONLY — verify nothing was applied to contracts.yaml and the checker proves every claim in the proposal against the real registry.
3. **Phase-4 protocol draft** (`plans/DE_PHASE4_PROTOCOL_DRAFT.md` + `de_phase4_protocol_check.py`): declared before any cell is read (rule 11) — verify no Phase-4 cell exists anywhere; the R-165(2) declared parameters carried.

## Scope 3 — the `1aaac18` closure claims (carried from the Codex era)
The ROUND2 filing reviewed the PARENT of `1aaac18`; its RR1/RR3 closure claims remain UNRELEASED. Rule on them at `1aaac18` itself.

## Scope 4 — process finding to rule on (Q-DE-19)
DE's batch was committed under BE's message because the seats share one git INDEX (DE staged 13 paths; BE's commit swept them). DE filed the provenance correction. Rule on: (a) whether the mixed commit contaminates either batch's provenance for freeze purposes; (b) a mechanism recommendation (per-seat staging discipline, `git commit -- <paths>`, or worktrees) — recommendation only, the mechanism decision returns to the coordinator/USER.

## Ground rules
Execute at the PINNED TIP; heavy runs via research.slice (≤12G, unit kept); the artifact's tape/fragment/topup byte counts are verifiable cheaply, full 5.1GB re-hash at your discretion; the known-flaky `v4 behaviour` gate fails safe under suite load (0c, not a regression). File ONCE under `workspace/reviews/REVIEW_…`, commit+push, then a one-line note in your pane.
