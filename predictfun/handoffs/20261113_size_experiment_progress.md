# Five/50-share experiment: execution progress

First checkpoint as of **2026-11-13T18:22:10.795438+00:00**, with named artifacts checked subsequently. The frozen size pipeline has begun after the complete recovery evidence passed integrity with a negative improvement result. No size/model advantage is claimed from these smoke tests.

## First four declared stages

Each stage below covers the same **four Oct 23 smoke windows**. These are implementation checks, not a multi-day performance population. No confidence interval or economic selection is made from them.

| Stage | Published as of UTC | Native generations / fill records | Named artifacts | Checks |
|---|---|---:|---:|---|
| q5 DEFAULT | 18:19:36.210287 | 436 / 222 | 12 | Accounting, policy events and original trade provenance reconstruct; one missing marked window retained |
| q5 neutral capture | 18:20:32.032828 | 436 / 222 | 24 | Neutral requests, fills, windows and generation economics exactly match DEFAULT; 34 paired actions/captures and saved features reconstruct; all 34 targets eligible |
| q50 DEFAULT | 18:21:10.763891 | 299 / 387 | 12 | Accounting, policy events and original trade provenance reconstruct; zero missing marked windows |
| q50 neutral capture | 18:22:10.795438 | 299 / 387 | 24 | Neutral requests, fills, windows and generation economics exactly match DEFAULT; 43 paired actions/captures and saved features reconstruct; all 43 targets eligible |

The original per-stage validators checked exact artifact coverage and all hashes. The separately frozen fast-execution provenance and all three receipt copies match. Synthetic controls remain bound to the original experiment and computation declarations. Fill-record counts are not distinct order counts and do not establish profitability or retained-priority benefit.

Receipts:

- [q5 DEFAULT](../reviews/model_size_priority_20261113/smoke_run_X_20261023_q5_default.json), SHA `9be5996cac5eb812ef482b768c3c71edf0f19ff4e4dec9e1950525d06d2776f8`.
- [q5 capture](../reviews/model_size_priority_20261113/smoke_build_X_20261023_q5_default.json), SHA `09638b9a808c373e349e77cabc4aab58d69205adbe556aa81ea1ca0714b81449`.
- [q50 DEFAULT](../reviews/model_size_priority_20261113/smoke_run_X_20261023_q50_default.json), SHA `4b921320784fbdfb3022044a57488a789cfbc360772e2016406aacef79cf0013`.
- [q50 capture](../reviews/model_size_priority_20261113/smoke_build_X_20261023_q50_default.json), SHA `c68b7aa40aa6ab8648de7871f0ac1575e6c7372e7822d106744916cc21023751`.
- [Execution dependency](../reviews/model_size_priority_20261113/execution_dependency.json), SHA `7df8db4688373583f0c34c3d28076d3e31f83ea02c41c6aa1f6cc23f0f470c3e`, binds recovery final evidence SHA `3bd21750832f5e65efbcf52b7d3ec94ab5b3db3e8493440ea0b80c25dcc8c6a7`.

At this checkpoint **4/169** declared size stages have receipts. The first full q5 DEFAULT reference run for Oct 23 is active under driver **1576587** (worker 691425). Full neutral capture, fits, local evaluation, sequential value/risk comparisons and final verification remain. The depth driver **2900451** waits for the complete size evidence. Designs, primary selection, thresholds and task order remain frozen. No live changes.


## 2026-11-13 18:32 UTC: first full q5 reference verified

The Oct 23 q5 DEFAULT run completed at **2026-11-13T18:32:06.399938+00:00**.
Its [receipt](../reviews/model_size_priority_20261113/full_run_X_20261023_q5_default.json),
SHA `5427158b23f084be23bf60650be773d172638ef3f546d82ae5afc95e86ebc9d2`,
covers **288 windows**, 26,808 generations and 16,555 native fill records.
Generation statuses are 22,232 `accepted_no_cancel` and 4,576
`postonly_refused`; 19 windows retain missing marked value. Original trade
provenance is reconstructed against 802,157 source trade legs.

The published checks reconstruct policy events and original trade provenance.
The stage uses DEFAULT, with zero forecast scores and zero learned pulls;
it does not test a fitted predictor. Exact artifact coverage, all 12 hashes,
three receipt copies and the frozen fast-execution provenance match under
the existing validators. No interval or economic comparison is made from
this one reference day.

This is **5/169** completed stages. The full q5 capture for Oct 23 started at
18:32:06.689713 UTC (worker **787752**, driver **1576587**). It must verify
neutral-path parity and construct paired action labels before these rows can
enter model training. Depth driver **2900451** remains waiting for full size
evidence. The model-improvement goal remains active.


## 2026-11-13 19:06 UTC: first full q5 capture verified

The Oct 23 q5 neutral capture completed at **2026-11-13T19:06:37.270602+00:00**.
Its [receipt](../reviews/model_size_priority_20261113/full_build_X_20261023_q5_default.json),
SHA `dcfb1b2e7e4874598b3d38bb54f4883963672a51166cb424a9f0de0551b37895`, covers **288 windows** and **26,808 neutral
generations**. Native requests, fills, windows and generation economics exactly
match the independently completed q5 DEFAULT reference: 44,357 requests,
16,555 fill records and 19 windows with missing marked value. Generation
statuses remain 22,232 `accepted_no_cancel` and 4,576 `postonly_refused`.

The assigned observation ledger contains 12,094 observations and **2,995 paired
actions/captures**. Of those targets, **2,980 are eligible** and **15 retain a
missing continuation-mark status**; all 2,995 short targets are eligible. The
training join preserves all 2,995 rows and those same statuses. The stage's
published checks reconstruct every withdrawal label, original fill provenance,
saved feature and preserved metadata/target. Trade provenance is checked against
802,157 source trade legs. These are implementation and data-construction
checks on one consumed date, not forecast or portfolio performance evidence.

At review time **2026-11-13T19:08:22.978003+00:00**, the existing validators confirmed
exact artifact coverage and all **24 artifact hashes**, the three receipt
copies, and the frozen computation provenance. This review did not run a second
numerical replay. No interval, model selection or size advantage is inferred.

This is **6/169** completed stages. The full Oct 23 q50 DEFAULT reference
started at **19:06:37.579342 UTC**, worker **1098260** under driver **1576587**.
Depth driver **2900451** remains waiting for the complete size evidence. Full
size datasets, chronological fits, sequential comparisons and final verification
remain; the model-improvement goal is active.


## 2026-11-13 19:15 UTC: first full q50 reference verified

The Oct 23 q50 DEFAULT reference completed at **2026-11-13T19:15:55.564626+00:00**.
Its [receipt](../reviews/model_size_priority_20261113/full_run_X_20261023_q50_default.json),
SHA `82963848e5bb113c0075fdd739caea32fda936771f2aaab8cb13b197f9655274`, covers **288 windows**, **16,864 generations**
and **22,861 native fill records**. Generation statuses are 14,891
`accepted_no_cancel` and 1,973 `postonly_refused`; two missing marked windows
remain explicit. The stage reconstructs all policy events and original trade
provenance against 802,157 source trade legs.

This DEFAULT run has zero model scores, zero learned pulls and zero guard
decisions. Its 12,725 cancellation requests arise from the retained native
policy, so DEFAULT does not mean an order can never be cancelled. Fill records
are not distinct takers and do not by themselves establish retained-priority
value. No one-day performance comparison, interval or model choice is made.

At review time **2026-11-13T19:17:08.330531+00:00**, existing validators checked exact
artifact coverage, all **12 artifact hashes**, the three receipt copies and
frozen fast-execution provenance. This is an artifact review of the completed
stage, not a second numerical replay or a fitted-model test.

This is **7/169** completed stages. The full Oct 23 q50 action capture started
at **19:15:55.852673 UTC**, worker **1185906** under driver **1576587**. It must
preserve the q50 reference path and reconstruct its paired targets and features.
Depth driver **2900451** remains waiting for all size evidence. No size/model
advantage is established; the full model-improvement goal remains active.


## 2026-11-13 19:42 UTC: first full q50 capture verified

The Oct 23 q50 neutral capture completed at **2026-11-13T19:42:16.216885+00:00**.
Its [receipt](../reviews/model_size_priority_20261113/full_build_X_20261023_q50_default.json),
SHA `5298fae30d8dd6afb2367394d0070b164b594fc3a66c2f5412c7d95a78e6352f`, covers **288 windows** and **16,864 neutral
generations**. Requests, fills, windows and generation economics exactly match
the independent q50 DEFAULT reference: 29,589 requests and 22,861 fill records,
with two missing marked windows retained. Generation statuses remain 14,891
`accepted_no_cancel` and 1,973 `postonly_refused`.

The ledger contains 8,814 observations and **2,151 paired actions/captures**.
Of these, **2,142 are training-eligible** and **nine retain missing continuation
marks**; all 2,151 short targets are eligible. The training join retains all
rows with these statuses. The published stage checks reconstruct every
withdrawal label, saved feature, preserved metadata/target and original fill
provenance against 802,157 source trade legs.

At review time **2026-11-13T19:43:45.197611+00:00**, the existing validators checked
exact artifact coverage, all **24 artifact hashes**, three receipt copies and
frozen fast-execution provenance. This was an artifact review, not a second
numerical replay. The q5 and q50 profiles now both have verified construction
on their first consumed date. Their action populations differ; these counts
do not establish a causal size or queue-value effect, forecast skill or a
portfolio improvement. No interval or economic selection is made here.

This is **8/169** completed stages. The Oct 24 q5 DEFAULT reference started at
**19:42:16.528643 UTC**, worker **1419992** under driver **1576587**. Remaining
size dates, chronological fits, local comparisons, sequential stress replays
and final verification still have to run. Depth driver **2900451** waits for
all size evidence. The model-improvement goal remains active.


## Subsequent full-stage checks

Each row is a completed stage reviewed with the existing validators: exact
artifact coverage, all named hashes, three receipt copies and frozen computation
provenance. The review uses saved evidence and does not repeat the numerical
replay. DEFAULT references have no learned scores or pulls; native cancellation
rules still apply. No per-stage performance selection or confidence interval is
made. Action-capture checks and fitted-policy results remain distinct.

| Stage | Receipt as of UTC | Population | Retained statuses | Evidence |
|---|---|---|---|---|
| Oct24 q5 DEFAULT | 2026-11-13T19:51:48.130995+00:00 | 288 windows; 23,187 generations; 15,215 fill records | 19,792 accepted / 3,395 post-only refused; 22 missing marked windows | Policy events and original trade provenance reconstruct; 12 artifacts; [receipt](../reviews/model_size_priority_20261113/full_run_X_20261024_q5_default.json) |
| Oct24 q5 capture | 2026-11-13T20:23:57.288179+00:00 | 288 windows; 23,187 neutral generations; 2,709 paired actions | 2,698 eligible / 11 missing continuation marks; all 2,709 short targets eligible | Exact neutral parity; labels, features and original fill provenance reconstruct; 24 artifacts; [receipt](../reviews/model_size_priority_20261113/full_build_X_20261024_q5_default.json) |
| Oct24 q50 DEFAULT | 2026-11-13T20:33:01.076606+00:00 | 288 windows; 14,336 generations; 21,205 fill records | 12,902 accepted / 1,434 post-only refused; one missing marked window | Policy events and original trade provenance reconstruct; 12 artifacts; [receipt](../reviews/model_size_priority_20261113/full_run_X_20261024_q50_default.json) |
| Oct24 q50 capture | 2026-11-13T20:57:18.351444+00:00 | 288 windows; 14,336 neutral generations; 1,927 paired actions | 1,925 eligible / two missing continuation marks; all 1,927 short targets eligible | Exact neutral parity; labels, features and original fill provenance reconstruct; 24 artifacts; [receipt](../reviews/model_size_priority_20261113/full_build_X_20261024_q50_default.json) |
| Oct25 q5 DEFAULT | 2026-11-13T21:06:31.745207+00:00 | 288 windows; 22,068 generations; 13,471 fill records | 18,844 accepted / 3,224 post-only refused; 22 missing marked windows | Policy events and original trade provenance reconstruct; 12 artifacts; [receipt](../reviews/model_size_priority_20261113/full_run_X_20261025_q5_default.json) |
| Oct25 q5 capture | 2026-11-13T21:37:50.397635+00:00 | 288 windows; 22,068 neutral generations; 2,638 paired actions | 2,630 eligible / eight missing continuation marks; all 2,638 short targets eligible | Exact neutral parity; labels, features and original fill provenance reconstruct; 24 artifacts; [receipt](../reviews/model_size_priority_20261113/full_build_X_20261025_q5_default.json) |
| Oct25 q50 DEFAULT | 2026-11-13T21:46:39.420688+00:00 | 288 windows; 14,328 generations; 19,151 fill records | 12,811 accepted / 1,517 post-only refused; three missing marked windows | Policy events and original trade provenance reconstruct; 12 artifacts; [receipt](../reviews/model_size_priority_20261113/full_run_X_20261025_q50_default.json) |
| Oct25 q50 capture | 2026-11-13T22:11:07.612584+00:00 | 288 windows; 14,328 neutral generations; 1,926 paired actions | 1,922 eligible / four missing continuation marks; all 1,926 short targets eligible | Exact neutral parity; labels, features and original fill provenance reconstruct; 24 artifacts; [receipt](../reviews/model_size_priority_20261113/full_build_X_20261025_q50_default.json) |
| Oct26 q5 DEFAULT | 2026-11-13T22:21:08.892373+00:00 | 288 windows; 28,107 generations; 16,840 fill records | 23,610 accepted / 4,497 post-only refused; 11 missing marked windows | Policy events and original trade provenance reconstruct; 12 artifacts; [receipt](../reviews/model_size_priority_20261113/full_run_X_20261026_q5_default.json) |
| Oct26 q5 capture | 2026-11-13T22:57:52.254541+00:00 | 288 windows; 28,107 neutral generations; 3,096 paired actions | 3,080 eligible / 16 missing continuation marks; short targets: 3,095 eligible / one missing mark | Exact neutral parity; labels, features and original fill provenance reconstruct; 24 artifacts; [receipt](../reviews/model_size_priority_20261113/full_build_X_20261026_q5_default.json) |
| Oct26 q50 DEFAULT | 2026-11-13T23:07:21.228460+00:00 | 288 windows; 17,930 generations; 22,718 fill records | 16,023 accepted / 1,907 post-only refused; two missing marked windows | Policy events and original trade provenance reconstruct; 12 artifacts; [receipt](../reviews/model_size_priority_20261113/full_run_X_20261026_q50_default.json) |
| Oct26 q50 capture | 2026-11-13T23:34:32.558755+00:00 | 288 windows; 17,930 neutral generations; 2,225 paired actions | 2,212 eligible / 13 missing continuation marks; all 2,225 short targets eligible | Exact neutral parity; labels, features and original fill provenance reconstruct; 24 artifacts; [receipt](../reviews/model_size_priority_20261113/full_build_X_20261026_q50_default.json) |

The Oct24 q5 reference binds 681,400 source trade legs. Receipt SHA
`8306113b624449db0b7e5bcf614b22299c827074f4ba9f37234b25176d843c64`; review as of **2026-11-13T19:53:05.625749+00:00**.

The Oct24 q5 capture exactly matches the independent reference on 37,965 requests,
15,215 fill records, 288 windows and generation economics; 22 missing marked
windows remain explicit. It retains 10,833 observations and all 2,709 sampled
actions, including 11 missing continuation marks. The training join preserves
these statuses; its fingerprint is
`3115ca3bf041508091ade8c8f0769e682fa377bbe09fce2eb82b01b6f7670fe9`.
Targets, saved features, metadata and fill provenance reconstruct against
681,400 source trade legs. Capture receipt SHA `cf2dac0c32fc84c9738128530d76b5057d0945ad6bdc83778fb138a65ead1fdc`;
artifact review as of **2026-11-13T20:24:16.622115+00:00**. This review checks the
saved evidence; it is not a second numerical replay or a performance result.

The Oct24 q50 reference reconstructs policy events and original provenance
against 681,400 source trade legs, including 3,007 repeat orders with complete
identity. DEFAULT has zero model scores, learned pulls or guard decisions; its
10,537 native cancellation requests remain part of the reference. Receipt SHA
`2a14edc520a258e150443a366e57c0313d929fe53f59cd1d2f1dcb826f7a3b9f`; artifact review as of
**2026-11-13T20:33:13.917199+00:00**. No size or economic advantage is inferred
from this one-day reference.

The Oct24 q50 capture exactly matches the independent reference on 24,873
requests, 21,205 fill records, 288 windows and generation economics, retaining
the one missing marked window. It preserves 7,732 observations and all 1,927
sampled actions, including two missing continuation marks; the training join
retains these statuses with fingerprint
`9684e6d89218c7b612b4c13944431507ce4071d0c991bf85e124088fdd5eea12`.
All withdrawal labels, saved features, metadata and original fill provenance
reconstruct against 681,400 source trade legs. Capture receipt SHA
`68b9889508dafce31987882964a949bce57f0e8c2c79a8d14e7a79a842934fe1`; artifact review as of
**2026-11-13T20:58:07.411944+00:00**. This is a review of saved evidence, not
a second numerical replay. Both size profiles now have verified construction
on the first two consumed dates; their action populations remain different,
and no size/queue premium, forecast skill or portfolio gain follows from
these construction checks.

The Oct25 q5 reference reconstructs policy events and original provenance
against 689,852 source trade legs, including 2,178 repeat orders with complete
identity. DEFAULT has no model scores, learned pulls or guard decisions; it
retains 14,490 native cancellation requests. Receipt SHA
`26d66dcf46385fa22c0f9aa2e2bcc402ead15676514ddf39fcf4bed0a7eb52ba`; artifact review as of
**2026-11-13T21:07:28.827713+00:00**. This checks saved construction evidence
and does not establish model or size performance.

The Oct25 q5 capture exactly matches the independent reference on 36,558
requests, 13,471 fill records, 288 windows and generation economics, retaining
22 missing marked windows. It preserves 10,295 observations and all 2,638
sampled actions, including eight missing continuation marks. The training
join retains these statuses with fingerprint
`d18e0f24b6525d897aa62943f885cac258ed4a46588bb71230164a31d4653949`.
Labels, saved features, metadata and original fill provenance reconstruct
against 689,852 source trade legs. Capture receipt SHA
`443cf0f622d40d04f544d64c8b66306aa34d0e444a2886eb8f48c4cdba677724`; artifact review as of
**2026-11-13T21:38:29.705013+00:00**. This checks saved construction evidence;
it does not repeat the numerical replay or establish forecast/portfolio gains.

The Oct25 q50 reference reconstructs policy events and original provenance
against 689,852 source trade legs, including 2,725 repeat orders with complete
identity. DEFAULT has no model scores, learned pulls or guard decisions; it
retains 10,803 native cancellation requests. Receipt SHA
`b938627e56f50427c9d873b288d7355668250dc7aade28c6bf5cb6b5c7dad312`; artifact review as of
**2026-11-13T21:46:49.906409+00:00**. This checks saved construction evidence
and does not establish model or size performance.

The Oct25 q50 capture exactly matches the independent reference on 25,131
requests, 19,151 fill records, 288 windows and generation economics, retaining
three missing marked windows. It preserves 7,670 observations and all 1,926
sampled actions, including four missing continuation marks. The training join
retains these statuses with fingerprint
`8d3ef0db1e178b465bed2d986953d77f7914b041257c1f06d491bc7b01d2cfff`.
Labels, saved features, metadata and original fill provenance reconstruct
against 689,852 source trade legs. Capture receipt SHA
`34c7c84a65e5348f2c8a50803ba59be2dbf245fac176ac283e12eb8e4991228e`; artifact review as of
**2026-11-13T22:13:37.689857+00:00**. This reviews saved evidence and does not repeat the
numerical replay or establish forecast/portfolio gains.

The feature artifact's KEEP/RESET status has 1,923 eligible actions and three
missing marks; withdrawal and training statuses have 1,922 eligible actions
and four missing marks. In `training_actions.parquet`, action
`20261025:3763401:default:18` has a complete residual comparison but three
missing-mark fill legs on PULL. The frozen join correctly retains that row as
`withdrawal=missing_mark;short=eligible`. All other status pairs agree; this
is a difference between target paths, not a dropped action or a changed rule.

The Oct26 q5 reference reconstructs policy events and original provenance
against 818,939 source trade legs, including 2,849 repeat orders with complete
identity. DEFAULT has no model scores, learned pulls or guard decisions; it
retains 19,187 native cancellation requests. Receipt SHA
`0dd58b8fbb05a3f589216662103f765c1d26eed8f7d63d023d040dd1992e9749`; artifact review as of
**2026-11-13T22:21:42.846786+00:00**. This checks saved construction evidence
and does not establish model or size performance.

The Oct26 q5 capture exactly matches the independent reference on 47,294
requests, 16,840 fill records, 288 windows and generation economics, retaining
11 missing marked windows. It preserves 12,749 observations and all 3,096
sampled actions. Withdrawal targets have 3,080 eligible actions and 16 missing
continuation marks; short targets have 3,095 eligible actions and one missing
mark. The training join preserves 15 withdrawal-only missing-mark statuses and
one with both withdrawal and short marks missing. Its fingerprint is
`b35bfbd9761b1ff25f42c7fb07229548b4946142609d8811d33f47e30904fce0`.
Labels, saved features, metadata and original fill provenance reconstruct
against 818,939 source trade legs. Capture receipt SHA
`d5acb244dfc2cde38dce80058fe41a560b177047603732c165fb0073ae38491d`; artifact review as of
**2026-11-13T22:58:40.578523+00:00**. This reviews saved evidence, not a second
numerical replay or a forecast/portfolio performance result.

The Oct26 q50 reference reconstructs policy events and original provenance
against 818,939 source trade legs, including 3,000 repeat orders with complete
identity. DEFAULT has no model scores, learned pulls or guard decisions; it
retains 14,010 native cancellation requests. Receipt SHA
`81c7c9720dfa53f7932226e3766e2777b06969e1d366f68fd83984bbd8affabc`; artifact review as of
**2026-11-13T23:08:11.465151+00:00**. This checks saved construction evidence
and does not establish model or size performance.

The Oct26 q50 capture exactly matches the independent reference on 31,940
requests, 22,718 fill records, 288 windows and generation economics, retaining
two missing marked windows. It preserves 9,330 observations and all 2,225
sampled actions, including 13 missing continuation marks; all short targets are
eligible. The training join retains these statuses with fingerprint
`23317f85925f266ffa47c196b5c68f7a69299366c8c08d4dc6a7aff21131acb9`.
Labels, saved features, metadata and original fill provenance reconstruct
against 818,939 source trade legs. Capture receipt SHA
`7ed79e0564caf1ca4287996ba6a437a17f3ef3115973815aabaa4e9caaeeb22d`; artifact review as of
**2026-11-13T23:34:44.662483+00:00**. This reviews saved evidence, not a second
numerical replay or a forecast/portfolio performance result.

This is **20/169** completed stages. Oct27 q5 DEFAULT reference worker
**3519770** started at **23:34:32.849481 UTC** under driver **1576587**;
depth driver **2900451** waits for complete size evidence. Both size profiles
now have verified reference/capture construction on the first four consumed
dates. Remaining dates, fits and sequential comparisons are pending. The goal
remains active, with size/model advantage unproven.
