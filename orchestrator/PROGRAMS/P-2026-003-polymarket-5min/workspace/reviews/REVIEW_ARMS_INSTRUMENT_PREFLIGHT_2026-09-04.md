# Review — the instrument, before the arms: DA32-R1 is not fixed, and the §8.1 value check does not exist

reviewer: pm-codex · filed 2026-09-04T11:08Z · tip `7a9e83f`
executed in `~/ctaNew-wt-rev`. **No race seal opened.** Nothing written under `data/`.

**The arms have not landed** — no DE round with real heads is at the tip — so per the dispatch I took item 3. **BE33-R1 and BE33-R2 have not landed either** (see the end), so there were no fixes to review. What I did instead is the prerequisite the dispatch names in item 1: *check the instrument before you trust its verdict on the arms.* That was the right call, because the instrument is not ready.

---

# ITEM 1 — **DA32-R1 is NOT fixed at the tip, and the consequence is worse than I filed**

The dispatch says DA's verifier *"has just been repaired for a defect you found in it."* At `7a9e83f` it has not been. `_hash_score_matches` is byte-identical to what I filed:

```python
    except Exception:
        return False
```

Last round I showed the helper returns `False` — the symbol meaning *"this score is genuine"* — on a row it could not evaluate. This round I followed it into the caller, and the caller converts that `False` into an affirmative verdict. `stub_or_real` computes `n_reproduced = sum(1 for s in scores if _hash_score_matches(s, salt))`, and `n_reproduced == 0` selects `REAL_EVIDENCED`. Driven on four score populations against the same document:

| scores supplied | `score_test` | `n_reproduced` | **verdict** |
|---|---|---|---|
| 5 genuine scores | NO_SCORE_REPRODUCED_FROM_IDENTIFIERS | 0 | `REAL_EVIDENCED` |
| 5 stub scores | EVERY_SCORE_REPRODUCED_FROM_IDENTIFIERS | 5 | `REAL_CLAIMED_BUT_EVIDENCE_SAYS_STUB` |
| **5 malformed rows** (no `slug`) | NO_SCORE_REPRODUCED_FROM_IDENTIFIERS | **0** | **`REAL_EVIDENCED`** |
| **5 junk rows** (`gen: None`, `score: "n/a"`) | NO_SCORE_REPRODUCED_FROM_IDENTIFIERS | **0** | **`REAL_EVIDENCED`** |

`malformed verdict != genuine verdict` → **False**.

**The verifier issues its strongest positive verdict — "this arm read the market" — on rows it could not evaluate at all**, and reports `n_scores_reproduced_from_identifiers: 0` beside it, which reads as affirmative evidence of realness rather than as an absence of evaluation.

This is the codomain predicate exactly: the unevaluable path lands on `0`, `0` is inside the codomain of a legitimate count, and `0` is the value this consumer reads as *real*. It is the same defect the module was written to catch, in the module that named it.

**It matters now rather than later.** DE is re-running the arms because it found the arm identities were wrong; this verifier is the instrument that would establish, from independent evidence, that each arm loaded the predictor it claims. A run whose scores are malformed for any reason — a schema change, a missing field, a partially-written file — would be certified `REAL_EVIDENCED`.

> **Clause, unchanged from last round and now with its consequence attached.** `_hash_score_matches` returns a third value (`None`/`UNEVALUABLE`) or raises; `stub_or_real` counts unevaluable rows as their own status and refuses to reach `REAL_EVIDENCED` while any exist. *"I could not evaluate this row"* and *"this row is genuine"* must not share a symbol, and `0` must not be reachable from both.

**Do not trust a `REAL_EVIDENCED` from this instrument on the arms until that lands.**

---

# ITEM 2 — preparatory: the declaration half is right; the **value** half does not exist

Good news first, and it is real. All three unproducible fields are in the verifier's **own independently-written** required list (15 fields, *"written from the plan rather than copied from the producer, so the audit compares two enumerations made separately"*), and the producer declares each of them honestly — `source: null` with a substantive reason, not a zero:

| field | declared |
|---|---|
| `maker_pnl_cents` | `source: null` — *"the replay values CANCELLATION (harm avoided minus sacrifice), not a maker book. A complete maker P&L needs spread earned on every fill minus adverse selection on every fill…"* |
| `spread_capture_cents` | `source: null` — *"`de_rho_estimator` computes a spread denominator PER RECEIVED FILL for rho; there is no book-level spread capture, and summing rho's denominator would be a different quantity"* |
| `inventory_loss_cents` | `source: null` — *"the inventory block carries NET and PEAK ABS shares and the increasing/reducing split, but no valuation of the position it leaves behind…"* |

And the audit's structure is right for this: `neither` (no source, no why) is the dishonest case, `absent_named` (no source, a why) is the honest one, and `ambiguous_zero` flags counter fields that have a source but no `evaluated_flag`/`denominator`/`absent_status`. `missing_from_producer: []`, `shared_sources: {}` — no field borrowing another's producer.

### ARM-R2 — MEDIUM — the audit reads the enumeration, never an arm's emitted values

`section_8_1_audit(fields=…)` takes the producer's **declaration** as data. There is no path in the verifier that takes an arm's **output document** and checks what the three fields actually carry. So if an arm emits `maker_pnl_cents: 0` — a counter initialised and never written, a default, a `sum([])` — the enumeration would still say `source: null` and the audit would still report it `absent_named`, while the emitted result carried a zero. **Nothing would notice**, and per the dispatch those three are exactly what separates a strategy-P&L verdict from an overlay increment.

> **Clause, and it is the codomain predicate applied to §8.1.** Add a value-level check: for every field the enumeration marks `source: null`, the emitted arm document must either **omit** it or carry an explicit `NOT_AVAILABLE` sentinel — **never a number**. A numeric value for a field declared unproducible is a refusal, not a note. This wants to exist before the arms' output is read, not after.

---

# ITEM 3 — the two BE33 items are still open, reported rather than re-found

Neither fix has landed at `7a9e83f`:

- **BE33-R1**: `be_fill_ledger.py:198` still reads `"exactly_once_total": "NOT COMPUTABLE — no tranche identity"` — the unscoped wording, in the per-coin block a reader sees — while `:78` still carries the correct `"NOT COMPUTABLE FROM HELD ARTIFACTS"`.
- **BE33-R2**: `def publication_provenance` is still defined in **both** `da_arm_replay_verify.py` and `be_read_cells.py`.

Nothing to review; both remain as filed.

---

## Findings

| # | sev | |
|---|---|---|
| **ARM-R1** | **HIGH** | DA32-R1 is **unfixed at the tip**, and its consequence is worse than filed: malformed and junk score batches both produce **`REAL_EVIDENCED`**, the verifier's strongest positive verdict, driven four ways. `0` is reachable from both "no stub scores" and "could not evaluate anything". **The instrument is not ready to judge the arms** |
| **ARM-R2** | MEDIUM | the §8.1 audit checks the producer's enumeration and never an arm's emitted values, so a `0` emitted for a field declared `source: null` would pass unnoticed — and those three fields are what separate a strategy-P&L verdict from an overlay increment |
| — | — | **Good, and worth recording before the arms:** all three unproducible fields sit in the verifier's independently-written required list and are declared `source: null` **with substantive reasons**, not zeros; `missing_from_producer` empty; no shared sources |
| — | — | **BE33-R1 and BE33-R2 are still open** — dispatched but not landed, so there were no fixes to review |

## Disposition

**Do not read a verdict from `stub_or_real` on the arms until ARM-R1 lands.** That is the whole of this round: the dispatch's premise was that the instrument had been repaired, and it has not been, so a verdict taken from it now would be the failure mode this phase exists to prevent — *a plausible answer from a path that did not really run*, produced by the instrument built to catch exactly that.

The declaration-level preparation for item 2 is genuinely in good shape and I would not hold the arms for ARM-R2 — but the value-level check should land before the arms' output is *read*, because that is the moment a zero would be quoted.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `7a9e83f` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No race seal opened**; `be_forward_day` never run. Every measurement above is a call into the verifier on documents I constructed — no artifact was read or written. Nothing written under `data/`. `~/ctaNew-wt-be`, `-da`, `-de` never read. Worktree clean.
