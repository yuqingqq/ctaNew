# Quote-default / learned-pull research loop
As of 2026-11-12T04:04:32.688048+00:00. **running: stress_20261102**. v4 unchanged; Codex core 8 with global research cap two (user 2026-11-12).

Completed prior test: own-order/calm study d475eccb, canonical comparison_v4.json; no promoted variant.
Current branch: research/predict-default-pull-20261111. One Codex worker on core 8; two research workers allowed globally (user 2026-11-12); consumed dates only; no live changes.
Plan: controls →8-window parity smoke →9 neutral build days →6 expanding model folds →6 primary replay days →primary report →6 stress replay days →final report.

| Stage | Status | As of | Detail |
|---|---|---|---|
| controls | complete | 2026-11-11T19:16:54.511169+00:00 | Passed; logs/controls.log plus artifact/feature checks retained. |
| smoke | complete | 2026-11-11T19:17:37.972088+00:00 | Passed; logs/smoke.log plus artifact/feature checks retained. |
| build_20261023 | complete | 2026-11-11T19:26:48.614005+00:00 | Passed; logs/build_20261023.log plus artifact/feature checks retained. |
| build_20261024 | complete | 2026-11-11T19:35:44.760164+00:00 | Passed; logs/build_20261024.log plus artifact/feature checks retained. |
| build_20261025 | complete | 2026-11-11T19:44:29.083393+00:00 | Passed; logs/build_20261025.log plus artifact/feature checks retained. |
| build_20261026 | complete | 2026-11-11T19:53:50.454560+00:00 | Passed; logs/build_20261026.log plus artifact/feature checks retained. |
| build_20261027 | complete | 2026-11-11T20:02:44.403680+00:00 | Passed; logs/build_20261027.log plus artifact/feature checks retained. |
| build_20261031 | complete | 2026-11-11T20:11:34.502572+00:00 | Passed; logs/build_20261031.log plus artifact/feature checks retained. |
| build_20261101 | complete | 2026-11-11T20:20:10.656338+00:00 | Passed; logs/build_20261101.log plus artifact/feature checks retained. |
| build_20261102 | complete | 2026-11-11T20:29:13.598768+00:00 | Passed; logs/build_20261102.log plus artifact/feature checks retained. |
| build_20261103 | complete | 2026-11-11T20:38:06.258037+00:00 | Passed; logs/build_20261103.log plus artifact/feature checks retained. |
| fit | complete | 2026-11-11T20:38:36.404523+00:00 | Passed; logs/fit.log plus artifact/feature checks retained. |
| eval_20261026 | complete | 2026-11-11T21:20:20.594411+00:00 | Passed; logs/eval_20261026.log plus artifact/feature checks retained. |
| model_diagnostic | complete | 2026-11-11T21:20:20.421339+00:00 | Frozen neutral-action diagnostic completed; portfolio and stress comparisons remain pending. |
| eval_20261027 | complete | 2026-11-11T22:01:04.820222+00:00 | Passed; logs/eval_20261027.log plus artifact/feature checks retained. |
| eval_20261031 | complete | 2026-11-11T22:42:12.791386+00:00 | Passed; logs/eval_20261031.log plus artifact/feature checks retained. |
| eval_20261101 | complete | 2026-11-11T23:24:01.983452+00:00 | Recovered missing marker after driver interruption: numeric rc=0, all artifact hashes and 105395 feature identities verified. Original artifacts unchanged. |
| driver | interrupted | 2026-11-12T04:04:27.550233+00:00 | Driver interrupted; completed stages retained. |
| eval_20261102 | complete | 2026-11-12T00:05:00.713258+00:00 | Passed; logs/eval_20261102.log plus artifact/feature checks retained. |
| eval_20261103 | complete | 2026-11-12T00:45:10.056923+00:00 | Passed; logs/eval_20261103.log plus artifact/feature checks retained. |
| primary_report | complete | 2026-11-12T00:45:43.363107+00:00 | Passed; logs/primary_report.log plus artifact/feature checks retained. |
| stress_20261026 | complete | 2026-11-12T01:35:45.851437+00:00 | Passed; logs/stress_20261026.log plus artifact/feature checks retained. |
| stress_20261027 | complete | 2026-11-12T02:24:44.564910+00:00 | Passed; logs/stress_20261027.log plus artifact/feature checks retained. |
| stress_20261031 | complete | 2026-11-12T03:15:48.366331+00:00 | Passed; logs/stress_20261031.log plus artifact/feature checks retained. |
| stress_20261101 | complete | 2026-11-12T04:04:32.524067+00:00 | Verified unchanged replay after scheduler handover; two global research slots, Codex core 8. |
| stress_20261102 | running | 2026-11-12T04:04:32.688048+00:00 | v4 unchanged; Codex core 8 with global research cap two (user 2026-11-12). |
