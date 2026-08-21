# PM measurement infrastructure preregistration v1

Frozen: **2026-08-21T03:26:27Z**

Canonical contract: **v17**

A-TWAP-1 hash: **`ab098a558d67a7ef45c685fe8e2ce027fc2ebf55bc278e853115288b25ee4091`**

Source-profile registry hash: **`d0fb3e0733b8b9fdea28e02540150f01554ebabffe42878aef13144829ce4847`**

This freezes infrastructure behavior only. It does not freeze, run, or inspect a
markout, calibration, probability, sigma, book, or flow/fill estimate. Existing
outcomes and downstream model scores were not read to choose this rule.

## A-TWAP-1

The machine-readable source is `config/a_twap_1.json`. Its `spec_hash` is the
SHA-256 of canonical JSON after removing the `spec_hash` field.

- Field: `twap60[{symbol}]`.
- Target: `(t0 - 5 s, T + 5 s]`, divided into 310 one-second slots.
- A slot is observed when at least one deduplicated payload event falls in it;
  extra events do not inflate coverage.
- Required complete fraction: at least 0.90.
- Maximum consecutive missing-slot run: 30 seconds.
- Strike readability: at least one event in `(t0 - 5 s, t0]` whose
  `t_known + clock_err <= t0`.
- Protected settlement span: `(T - 60 s, T]`; it may contain no missing slot.
- `weight_missing` is the missing fraction of those 60 protected slots and must
  equal zero.
- Failure action: `EXCLUDE_UNIT`, but only with `gap_arm=BOTH`: the excluded
  estimate and the all-data indicator arm must both be reported. The rule never
  authorizes exclude-only inference.

Coverage is factual and is written before this rule is evaluated. An
`AdmissibilityDecision` is a separate, hash-bound record. Draft, mismatched, or
retired rules are refused rather than evaluated.

## Knowledge-time source profiles

All current Tier-0 collectors record host `recv_ns`, so they are OBSERVED
sources. `config/source_profiles_v1.json` freezes a 1 ms local clock-error bound.
At freeze time `chronyc tracking` reported normal leap status, RMS offset
0.622 microseconds, root delay 140.952 microseconds and root dispersion
36.700 microseconds. The 1 ms bound is deliberately more than nine times the
measured root-distance scale.

Changing host, region, clock discipline, cadence semantics, protected span, or
any threshold requires a new version and hash. Existing partitions remain tied
to the old hashes; they are never overwritten in place.
