# Codex early re-review — R-313 consumer exact-field check — 2026-08-29

**Exact reviewed commit:** `af95b0971c628feb960cd034efe4d4abfd541050`

**Scope:** the locally committed R-313 addition to
`be_fragment_diagnostic._index_tape`. No real fragment score was run and no
diagnostic result was produced.

## Verdict

**PR3-FD5 HOLD MAINTAINED.** The new independent consumer check rejects a
short or long state mapping, but it does not establish the claimed exact pinned
field identity. It compares only the number of keys.

## Executed residual

The code checks:

```python
if len(state) != len(feats):
    raise DiagnosticRefused(...)
```

I constructed an `OK` row with 45 state keys:

- removed required pinned field `bn_feed_age_s`;
- added undeclared field `not_a_feature`;
- retained the same total count, 45.

Production `_index_tape(..., split="score")` **accepted** the row. The missing
required field encoded as `0.0` and the unknown field was ignored. This is the
same anti-safe behavior R-313 is intended to stop, reached through a
count-preserving substitution.

The five added selftests cover 1/45, 44/45, and 46/45 mappings, but not the
45-for-45 missing-plus-extra case. Consequently their green result does not
exercise field identity.

## Closure

Require exact set equality:

```text
set(state) == set(features_in_order)
```

Refuse with the row identity and explicit sorted `missing` and `extra` names.
Add the 45-for-45 substitution above as a known-bad, while retaining the full
positive control. Only then can this independent R-313 line contribute to
releasing PR3-FD5; the other pre-run findings in
`CODEX_REREVIEW_FRAGMENT_DIAGNOSTIC_PRERUN_2026-08-29.md` remain separately
load-bearing.
