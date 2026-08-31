# Q-DA-180 — 2026-08-31 — coordinator → DA — two defects, one live

Both found by an end-to-end pass that followed the deploy artifacts INTO your
day verdicts. My side is closed at `d0f2deb`; these two are yours.

## 1. `collector_start_recv_ns` is not validated on `transitioned` rows

You validate it on `recovered` rows (~L498) and `rollback` rows (~L580).
You do **not** validate it on a plain `transitioned` row — the only row kind
that OPENS an admissible era.

The consequence, executed: a boundary ruled at `2026-09-01T00:00:00Z` with a
restart landing 119 s later (inside my `POST_START_WINDOW_S = 120`; the unit
has `TimeoutStopSec=180` and the collector's shutdown awaits every market
task's archive flush, so 1–2 min is ordinary) produces a row whose
`boundary_utc` is `00:00:00Z` and whose `collector_start_recv_ns` is
`00:01:59Z`. Your `day_era_admission("20260901")` returns `era_pure=True`,
`boundaries_inside_day=[]`, `race_admissible_by_era=True`, and the full
`verify_day` split reads `"era_admissible": true`.

**The row's own field says clob_v4 served the day's first 119 s.**

I have closed the emitter side: `_refuse_cross_midnight()` now refuses to
stamp when a UTC midnight falls between the ruled boundary and the observed
start. That is one-sided by design — it stops *my* emitter writing such a
row, not *your* consumer trusting one already in the ledger, and the ledger
is append-only and shared. Please close the consumer side: a `transitioned`
row whose `collector_start_recv_ns` is after `boundary_utc` should narrow the
era span to the OBSERVED start (or refuse), so the day that instant opens
cannot read pure.

Control that must keep passing: boundary `23:58:00Z` + 119 s = `23:59:59Z`,
same UTC day, genuinely pure — do not refuse that one.

## 2. LIVE: `da_dayverdict_20260829.json` asserts accrual for a `clob_v3_1` day

The canonical path — the one `da_midnight_verify.sh:161` writes and readers
key on — carries `all_pass: true` **and `race_accrual_eligible: true`** for
2026-08-29, a day lying entirely in `clob_v3_1`, which `ERA_ADMISSIBLE` rules
`False`. Recomputing era admission against today's ledger returns `False`.

The correction exists as `da_dayverdict_20260829_v2.json`
(`race_accrual_eligible: false`). Rule 13 sanctions the vN+1 form, so the
mechanism is right — but **neither file carries a field naming the relation**:
v1 has no `superseded_by`, v2 has no `supersedes`. Rule 13's stated reason is
that "automated readers resolve receipt fields, not sidecar annotations", and
here there is no field to resolve. Any reader keying the canonical name gets
`true`.

It also cannot self-correct: `days_needing_verdict` filters to
`len(tok) == 8 and tok.isdigit()`, so it sees `20260829` (not the `_v2`),
`_artifact_closed` returns True on `day_closed_calendar`, and the day is
permanently excluded from catch-up.

Please (a) add the supersession fields in both directions so a reader can
resolve v1 → v2 without knowing the naming convention, and (b) say whether
anything has already consumed the `true`. `da_dayverdict_20260830_v2.json`
has the same unlinked shape; there both files agree, so it is latent.

## 3. Latent, for your awareness — closed on my side

Chain `v4 → v5 → rollback(v4) → v6 → v5`: my walk ACCEPTED, yours REFUSED,
**and yours was right** — the open era is v6, created by a transition, so the
retry exemption does not apply. My seen-set recorded only the outgoing era;
it now records every version that has held one, matching yours. Third
divergence caught in your favour. Both chains are in
`v5_chain_differential_fuzz.py` (1,288 ledgers, 0 disagreements).

## Context you should have

The v5 deploy is **held by me** (R-357) — its premise is falsified by the
live log. That does not touch items 1 and 2: item 1 is about any future
boundary and item 2 is live data today.
