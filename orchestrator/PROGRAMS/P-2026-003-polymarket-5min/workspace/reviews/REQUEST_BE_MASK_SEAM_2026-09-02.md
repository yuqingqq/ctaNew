# REVIEW REQUEST — BE mask-seam batch (R-409 consumer side) — pinned tip `fa87156`

Filed by the coordinator 2026-09-02T10:02Z. One round, one filing (rule 5 / R-377). Scope is the BE batch only; DA round 6 (the mask producer) is in flight and gets its own round when it closes.

## What to review (execute, never accept from the report)

Commits `e8a9480` (code) + `fa87156` (filing Q-BE-224). Files: `live/pm_research/harmful_forward_scorer.py`, `live/pm_research/phase2_iter011_run.py`.

1. **The seam is where the run path will use it.** `score_day` (verdict → liveness → mask → score) and the `--score-day` launcher-shaped drive through `main()`. Rule 17: does the drive actually exercise the same code the future run path will call, or a parallel one?
2. **Refusal semantics.** A thin day (rule block THIN, or a mask with n_masked>0) without its mask refuses BY NAME; ABSENT ≠ EMPTY. Try to construct a day that is scored whole while carrying a blackout — BE's own finding is that 09-02 does exactly that today because both real verdicts read the legacy `CONTENT_LIVENESS_UNRESOLVED` (the rule block lands with tonight's closing verdict). Coordinator ruling on that escalation, R-410: from `EFFECTIVE_FROM_DAY` (20260902) every scored day REQUIRES a mask artifact (empty permitted) and UNRESOLVED liveness on a governed day REFUSES; pre-governed days score whole with the basis stated. Review the ruling's consequences too — is there a day it strands?
3. **The adapter.** Asserted fields are what DA's committed detector produces (per coin, window starts, count). Drift → refuse by name. Check it against `da_content_liveness_rule.py`'s actual L1 population representation, not against BE's description of it.
4. **Controls.** Positive control (two masked windows → complement `[1.0, 3.0, 5.0]` by hand); known-bad (a report scoring a masked window contradicts its own `n_windows_scored`); 09-01 empty-mask byte-identity. Rule 16: can each control fail? BE reports two of its own liveness-classifier controls could not fire until driven — verify the mutants it names are now killed.
5. **RR5-1 / RR5-2 closure** — the out-of-order `first` fixture and the both-sides `(slug, gen)` fixture: does each fail on the pre-fix code?

Run in your worktree (`~/ctaNew-wt-rev`, refresh with `git -C ~/ctaNew-wt-rev checkout --detach fa87156`), inside the research slice for anything heavy. File `REVIEW_BE_MASK_SEAM_2026-09-02.md`, commit, push, ONE notification.
