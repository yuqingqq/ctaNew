# DE 112's 09-06 rehearsal — the two fixture receipts

**What these are.** Two FIXTURE day runs (`FIXTURE-DAY-1`), launched under
THE_ONE_COMMAND's exact form as real transient `systemd` services on a
**scratch** lock, to rehearse tonight's 09-06 real day. `status` on both is
`FIXTURE_DAY_RUN_NO_REAL_DATA`. They are evidence for Q-DE-112 and for nothing
else: no real day ran, the heavy lock was never taken, and no economic value
exists in either file.

**Why they are HERE and not in the ledger.** No `_FIXTURE__` receipt has ever
been under `data/` — the ledger has no fixture location, and a fixture artifact
sitting in `data/pm_5min/derived/` would be a non-result file beside the real
ones. But REV 85 §4's principle binds: *a receipt of record that a reviewer
cannot open is a claim.* They were written to a session scratch directory under
`/tmp`, which is where each file's own `output_name.directory` still points and
which does not survive the machine. So they land in the repo's existing tracked
fixture location, where any reviewer can open them at any commit.

| file | sha256 | unit | InvocationID |
|---|---|---|---|
| `…_FIXTURE__20260906T195909Z.json` | `97962a627743a2f5c40d2ee9e23decbe659ca70346aeb6c487d2522f86cc2e47` | `de112rehearse` | `42eac56734734a44a8341b2d13a283a5` |
| `…_FIXTURE__20260906T200348Z.json` | `b5c620995ba93649bbd2b006eb2416c5da7f7244d62e5af90f393fb536f4f5ea` | `de112rehearse_2` | `67b68723d9ca40b4b6d2e4b91bd89e8a` |

**THEY CARRY NO `supersedes` LINK, and that is a correction to Q-DE-112.** That
row called the earlier one "superseded by" the later. The runner does not emit a
supersession on a fixture day run and neither file contains the key: they are two
independent launches of the same fixture day, the second after DE 112's two
receipt fixes landed (`provenance.producer_exit_map`, and the merge facts in
`design_chain_orphans_at_emit`). The ordering is recorded in the register and
here — in prose, which is what it is — and NOT as a pair in the bytes. A pair
manufactured after the fact would be exactly the provenance this programme
refuses.

**The difference between them** is the two added blocks: the later receipt
carries `provenance.producer_exit_map` (head `producer_exit_maps_v4.json`,
`829569e2b924d1b7…`) and `n_merge_links` / `merged_tips` / `fork_status` under
`provenance.design_chain_orphans_at_emit`; the earlier one carries neither.
