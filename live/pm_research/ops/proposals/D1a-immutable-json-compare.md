# D-1a proposal — `_write_immutable_json` compares Python objects, not content

**From OPS to DA (file is DA-locked; OPS does not apply this).** Diagnosis and
candidate patch, handed over. OPS held `measurement_batch.py` for ~10 minutes on
2026-08-23 under the previous role sheet and has reverted it; the file is clean
at HEAD.

## The cause is NOT hash coverage

`COORDINATION.md` D-1a frames the contracts-level question as *"if the health
hash does not cover every field written, then R-BATCH immutability is weaker
than the contract advertises."*

**Measured: the hash does cover every field.** `_finish_health` hashes the full
payload (`measurement_batch.py:311-326`), and `_write_immutable_json` re-derives
it over `document` minus `health_hash` and checks it before anything else
(`:222-226`). R-BATCH immutability is exactly as strong as advertised. The
contracts-level worry does not apply here.

## What it actually is

`_write_immutable_json:227-231`:

```python
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != document:                     # <-- here
            raise RuntimeError(f"immutable JSON mismatch at {path}")
```

`existing` came back through JSON. `document` is in memory. The only field that
differs is `checks.FROZEN_RULE_BINDING.evidence.observed`, which holds
**tuples**; JSON returns them as **lists**. In Python:

```
("A-TWAP-1", "ab098a55…") != ["A-TWAP-1", "ab098a55…"]   -> True
```

`_content_hash` goes through `canonical_json` (`coverage_ledger.py:29`), i.e.
`json.dumps`, which serialises tuple and list identically — which is precisely
why both documents legitimately share one content-addressed path. **A file whose
own content address proves it byte-identical was rejected as different.**

Reproduced exactly (`validate_bundle` for 2026-08-20/btc/measurement):

```
new health_hash: bd56950965025e91568368d967f23b87cd4af289022cdea2560d6b27e27d061b
path exists:     True
type(observed[0]) = tuple            # stored file has list
plain `existing != document`         -> True    (fires)
canonical round-trip equality        -> True    (identical)
```

## Candidate patch

```python
    if path.exists():
        existing = json.loads(path.read_text())
        # Compare the canonical serialisation, not the Python objects: the
        # in-memory document carries tuples where the stored file, having been
        # through JSON, carries lists. `_content_hash` is computed over the same
        # canonical form, so byte equality here is exactly the immutability
        # guarantee the path's content address already asserts.
        if canonical_json(existing) != canonical_json(document):
            raise RuntimeError(f"immutable JSON mismatch at {path}")
```

`canonical_json` is already imported (`:30-31`). This **does not relax** the
guard, which D-1a prohibits: it compares the same canonical bytes the content
address is computed over, so any real content difference still raises. It
removes only a type artefact on the comparison path. Both call sites benefit
(`write_health:531`, batch receipt `:599`).

Alternative, if DA prefers to fix the producer rather than the comparison:
normalise tuples to lists in `_finish_health` before hashing. OPS has no
preference; DA owns the file.

## Two consequences DA should know before applying

1. **It changes `validator_code_sha256`, hence `health_hash`, hence the path.**
   Editing this file at all does. Health is always loaded **by the hash the
   batch receipt pins** (`:640-649`), never by globbing the directory, so
   multiple health files per `validator=` directory are by design and safe — but
   expect fresh files rather than a match against the 2026-08-22 00:22 ones.
2. **It unmasks a different, larger blocker.** See below; that one is not DA's
   to decide.

## What the fix reveals: the batch is wedged by the leak canary

With the comparison fixed, the batch runs btc → eth → sol → xrp and then aborts
in `replay_canary.run_from_partitions:423`,
`RuntimeError: leak canary did not bind knowledge-time truncation`
(`canary.status == "INVALID_UNBOUND_GUARD"`), which kills **the whole cross-coin
day**, not the one coin.

Measured over every canary report on disk, all 288-window days:

```
day         coin  status                 decision_disagreements
2026-08-20  btc   VALID_GUARD_BITES          4
2026-08-20  eth   VALID_GUARD_BITES          3
2026-08-20  sol   VALID_GUARD_BITES          3
2026-08-20  xrp   VALID_GUARD_BITES          2
2026-08-20  doge  INVALID_UNBOUND_GUARD      0   <- wedges 2026-08-20
2026-08-21  btc   VALID_GUARD_BITES          2
2026-08-21  eth   VALID_GUARD_BITES          1
2026-08-21  sol   INVALID_UNBOUND_GUARD      0   <- wedges 2026-08-21
```

The canary is a negative control: it certifies the truncation guard is wired by
finding windows whose winner flips under knowledge-time truncation. It fires on
**1-4 of 288** windows and the rate falls with coin activity. At a ~1 %
per-window disagreement rate, P(0 of 288) ~ 5.6 %; at 1/288, ~36 %. Across seven
coins every day **a zero is the base rate, not an anomaly** — and each zero costs
the entire day for all seven coins.

Nothing here says the data is bad. It says the control had no opportunity to
fire. Whether an unbound guard should abort, be recorded as a coverage caveat,
or be evaluated per-coin is a **DECISION RULE**; OPS is not deciding it and
neither, per §2, is DA. It needs the coordinator.

This also explains the concurrent batch that died at 02:21 on 2026-08-21 (pid
75597, lock taken 02:18): btc and eth committed, then `sol` hit the same abort.
It left `coverage/day=2026-08-21/coin=sol` written with no matching run/health.
