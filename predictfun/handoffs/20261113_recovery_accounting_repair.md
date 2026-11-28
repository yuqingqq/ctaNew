# Empty-fill settlement correction for recovery replay

The original age500 integration attempt completed simulation on Oct26's first
eight windows for each of four policies, then failed during settlement
reconciliation. DIRECT10 had zero fills. The original checker called
`numpy.isfinite` on empty object-typed columns and raised TypeError. The original
worker and driver exited; this was not a held process or a model failure.

The failed directory and log remain unchanged, bound by
[the failure receipt](../reviews/model_recovery_20261113/empty_fill_age500_failure.json).
Supplemental source commit **a4c190ec** adds an adapter while preserving every
file in the scientific freeze. Only a zero-row fill table receives a copied,
numeric-typed side/quantity/price/settlement schema. The unchanged reconciler
still checks registry identity, cash, inventory and payouts. Nonempty tables
pass directly to the original function.

[Seven control groups](../reviews/model_recovery_20261113/accounting_v2_controls.json)
passed at **2026-11-13T07:28:02.056099+00:00**. They reproduce the actual old
failure; accept its zero portfolio; check nonempty tie/non-tie parity and input
immutability; refuse missing schemas, bad registries, nonzero/nonfinite empty
portfolios and corrupt fills; and prove frame/event comparison can reject a
planted difference. The wrapper returned zero, 139 MiB peak RSS, 0.57 seconds.

The replacement pure driver **3159222** starts with the same age500 smoke in
`replay_smoke_v2/age500/20261026`. Before a canonical success receipt can be
written, all **18 raw parquet tables** and the decompressed recovery-event stream
must exactly match the original failed attempt. Its old output and log are
never reused for writes. New receipts carry the supplemental declaration,
controls, failure and parity identities. The final verifier also binds this
correction chain. Full scenarios, quantities, models, timings, targets and
success criteria are unchanged; no new candidate or live action.

The [supplemental declaration](../reviews/model_recovery_20261113/accounting_v2_declaration.json)
is SHA `c38f6acdcfd9dd9fac27345d86e4663afcd0cb828c8db3b13ebc979009056123`.
Raw parity passed at **07:31:17.944617 UTC**; the canonical smoke receipt is
as of 07:31:17.805098 UTC and was published only after the comparison passed.
All 18 raw tables and decompressed events match exactly. The wrapper returned
zero, 1,893 MiB peak RSS, 116.36 seconds. Its zero-fill direct arm is retained;
age-floor checks pass with 663 suppressed opportunity events across four arms /
eight windows each. These smoke counts are integration evidence, not performance.

[Raw parity receipt](../reviews/model_recovery_20261113/accounting_v2_raw_parity.json)
SHA `68277f8c926ef5b0d4f74ae8ce9e259180a5521aebaff3f184a497a72413dbf1`.
The [canonical corrected smoke](../reviews/model_recovery_20261113/smoke_run_age500_20261026.json)
supersedes the failed attempt in-band and records its new artifact folder.
A metadata review verifies all mirrored copies, 38 raw file hashes, five top-level
smoke artifacts and four presence artifacts; the provenance checker accepts the
actual receipt and rejects each of four corrupted links.

The same replacement driver started the first full primary replay at
07:31:18.208066 UTC. Full replay still has to determine strategy performance.

Operational correction: launcher 3149455 exited before any stage started because system Python lacked numpy imported by the inherited driver. Its failed import log and receipt remain preserved. `accounting_v2_runner_started_venv.json` supersedes that launch using the existing project venv, with worker interpreter and source unchanged.
