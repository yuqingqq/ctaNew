# Codex early re-review — `e7de218` valuation contract — 2026-08-29

**Exact reviewed commit:** `e7de21886eee20ba9607fa195d1861a32cf225b4`

**Scope:** the committed FD1 source-field rejoin and FD5 exact state-field
repair in `be_fragment_diagnostic.py`. No real fragment score was run and no
diagnostic number was produced.

## Verdict

**FD5 EXACT-SET LEG CLOSED. FD1 HOLD MAINTAINED.**

The R-313 consumer check now compares the actual state-field set with the
pinned set and refuses count-preserving substitutions with explicit missing
and extra names. The source rejoin also correctly refuses absent source
identities, duplicate kept identities, and duplicate source identities.

The valuation gate is reconstructed from `latency or {}` without first
validating the latency target. A malformed kept row can therefore become a
clean `any_fill_ahead=False` row and proceed whenever another row is true. That
is missing valuation input encoded as a no-fill outcome.

## Executed residual

Two `OK` source/kept rows were supplied:

- row 1: complete valid target-latency cell with preventable shares;
- row 2: `latency=None`.

`rejoin_source_fields` accepted the population and returned:

```text
valuation gates = [True, False]
rejoined=2, valuation_gate_true=1, valuation_gate_false=1
```

The all-false guard cannot catch a partial malformed population. The later
boolean field assertion also cannot catch it because `hm.keptrow` always emits
a boolean after applying `latency or {}`.

This contradicts the commit's claimed refusal for an absent valuation gate.
The canonical gate should indeed be reconstructed rather than trusting a
second raw rule, but its inputs require a strict contract first.

## Exact current-artifact fact

I streamed all 482,224 current exposure rows and checked every one of the
472,413 `OK` rows at the frozen 50 ms target:

```text
latency not dict                         0
target latency cell missing/not dict    0
missing value/share/stale subfield       0
non-numeric or non-finite subfield       0
raw any_fill_ahead not boolean           0
raw gate true                       66,980
canonical gate true                 66,980
raw/canonical disagreements              0
```

Thus this is another F1 shape: the exact artifact's valuation inputs are
complete, but the consumer's advertised refusal is unsound. It does not erase
the current data fact or justify running before the remaining release
conditions close.

## Closure

Before `hm.keptrow`, require for every kept row:

- `latency` is a mapping;
- the exact frozen target-latency key exists and maps to an object;
- `preventable_value_cents`, `preventable_shares`, and `stale_shares` exist and
  are numeric, finite, and not booleans;
- share fields are nonnegative and any other existing exposure invariants are
  enforced rather than repaired by default.

Add known-bads for missing latency, missing target key, missing subfield,
wrong type, NaN/Infinity, and negative shares. Retain a valid
zero-fill/false-gate row as a positive structural control so strictness does
not redefine absence of a fill as malformed data.

The other PR3-FD release conditions remain independently load-bearing, and the
single real fragment score stays dark.
