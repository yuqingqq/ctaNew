# Review — BE round 33: the bracket, and whether exactly-once is really uncomputable

reviewer: pm-codex · filed 2026-09-04T11:02Z · tip `4eccb31` (BE33 at `858e408`)
executed in `~/ctaNew-wt-rev`. **No race seal opened** — 09-01/09-02/09-03 feeds untouched; all measurements on the **re-scored 08-29 development feed**. Nothing written under `data/`.

---

# ITEM 1 — **the non-computability claim is correct**, and I checked it at three levels

Everything else rests on this, so I did not take it from the filing.

**At the builder.** `harmful_exposure_rows.attribute_fills` builds real tranche identity — `gens[k]["tranches"]` holds `{"t", "shares", "level", "markout_cents_per_share"}` per fill, with a timestamp that would de-duplicate exactly. `label_rows` then writes `row["latency"] = lat` and `row["any_fill_ahead"]` and **nothing else from `trs`**: the identity exists for the length of one function and is discarded.

**At the largest held artifact.** I streamed the first row out of `harmful_exposure_rows_v3_eraB.json` (1,241,115,096 B) without loading it:

```
row keys: any_fill_ahead, coin, day, gen, gen_t0, gen_t1, latency, level, net,
          qahead, resting, side, slug, status, t0, t_end, t_start
tranche / fill-identity fields: NONE
latency['50'] keys: preventable_shares, preventable_value_cents, stale_shares
```

Three sums, no identity. **At the feed**, the same — established last round.

**So: not recoverable from any held artifact.** It *is* recoverable by re-running the producer, because the tranches are constructed inside the replay from `join_fills(fill_log, engine_fills)` and the tape still exists — which is exactly the route BE names (`what_would_make_it_computable`: *"emitting tranche identity (t, shares, level) or a per-action de-duplicated total from the BUILDER, which is a producer change and a re-run, not a downstream repair"*). **The bracket is not a retreat; it is the honest report of what today's artifacts support, and the producer change is the only route to better.**

### BE33-R1 — LOW — two wordings for one fact, and the unscoped one is the one a reader sees

`POPULATION_NOTE` at `:78` says **"NOT COMPUTABLE FROM HELD ARTIFACTS"** — exactly right. The per-coin block at `:198`, which is what appears in the emitted output beside the numbers, says **"NOT COMPUTABLE — no tranche identity"** — unscoped. A reader of the coin block could take it as *not computable in principle*, which would make the producer change look pointless and the bracket permanent.

> **Clause.** Use `:78`'s wording in the per-coin field too.

---

# ITEM 2 — the bracket: **bounds correct, and the width is stated as precision, not as a result**

**LOWER ≤ TRUE.** The largest single row's window is a set of real, distinct tranches contained in the action's union, so its sum cannot exceed the de-duplicated total. **UPPER ≥ TRUE.** Summing every row counts each tranche once per row containing it, i.e. at least once. Both directions hold for *the de-duplicated preventable population*, which is the object `POPULATION_NOTE` names — not total filled shares, and BE says so (`therefore_this_is_NOT`).

**The control that matters is the collapse, and it holds.** Driven:

| case | UPPER | LOWER | width |
|---|---|---|---|
| single-row action (no overlap possible) | 10.0 | 10.0 | **1.000 — the bracket collapses to exact** |
| 3-row action 10/20/30 | 60.0 | 30.0 | 2.000 |
| mixed 3-row + 1-row | 67.0 | **37.0** | — the single-row action enters LOWER in full, correctly |

A bracket that did not become exact where no overlap exists would be a bracket that measures its own construction rather than the data. This one does.

**And I recomputed both bounds independently from the re-scored 08-29 feed** — my own grouping and arithmetic, not the module's:

```
                LEDGER                          MINE                match
btc   UPPER 271,927.8  LOWER 152,334.0   271,927.8 / 152,334.0      exact,  width 1.785x
eth   UPPER  22,858.7  LOWER  19,882.7    22,858.7 /  19,882.7      exact,  width 1.150x
notional btc  UPPER $141,539.71   LOWER $79,636.37
```

BE's 1.785x and 1.150x are exact. And the framing is right: the module carries `bracket_width_ratio_shares` beside the bounds, `explicitly_not_claiming_exactly_once: True`, and no midpoint anywhere. **The width is presented as the precision available, not dressed as a result.**

One thing to carry with it: **the precision differs sharply by coin** (btc 1.785x, eth 1.150x), so a pooled figure would blend two very different precisions and should not be formed.

---

# ITEM 3 — both prior findings closed, driven

**BE31-R3 — closed.** The ledger now refuses an empty **field**, consulting the counter it already computed:

```
08-29 feed lacking preventable_shares/level -> REFUSED:
  "every one of btc's 80929 fill-bearing rows carries NO `level`, so no notional
   can be formed and a zero here would mean 'field absent', not 'nothing'"
empty FILE -> still REFUSED
```

The refusal states the codomain point itself — *a zero here would mean 'field absent', not 'nothing'* — which is the standing predicate applied at the site rather than quoted.

**BE31-R1 — closed, and it now distinguishes a call from a mention.** The check is an AST census taking `src` as a parameter *so it can be driven against a source where the call is absent*. Driven three ways on modified source:

| source | `n_calls` | `bare_reference_sites` | reachable from an entry point |
|---|---|---|---|
| real (a genuine call) | **1** (`compute`) | `selftest` | **`emit`** |
| only a COMMENT mention | **0** | `selftest` | None |
| only a BARE REFERENCE | **0** | `compute`, `selftest` | None |

A mention is not a call; a reference is counted as a reference and not as a call; and the predicate is re-evaluated on the mutated source rather than inferred. That is the whole of what was missing.

### BE33-R2 — LOW — the provenance census now exists twice

DA absorbed it into `da_arm_replay_verify.publication_provenance` last round, on the reasoning that one instrument beats two; BE has now implemented its own in `be_read_cells.publication_provenance`. Both are correct and they will drift. Worth a ruling on which is the instrument of record — my own recommendation last round was DA's, because it audits *other* modules without importing them, whereas BE's reads its own file by default.

---

# ITEM 4 — **the bracket bounds may be published**, with four caveats

Yes. The construction is sound, the bounds reproduce exactly under my independent recomputation, the collapse control holds, and the empty-field refusal now stops a missing scale becoming a zero. Publish with these attached, none of them optional:

1. **They bracket the de-duplicated PREVENTABLE population** — tranches inside the one-second horizon at or after the 50 ms cutoff. Not total filled shares, not total notional, not the no-cancel book. Fills before the cutoff are `stale_shares` and fills outside the horizon are outside the population entirely.
2. **Quote the range, never a midpoint.** btc's preventable notional is known only to lie between **$79,636 and $141,540** — a factor of 1.785. "About $110k" would be a number nobody computed. And do not pool btc with eth: 1.785x and 1.150x are different precisions.
3. **Still gross.** Fees unquantified, exit/settlement absent, quote size absent, inventory absent, capital absent — so no return on capital exists. `NOT_A_NET_RETURN` states all five and should travel verbatim.
4. **Say that the exactly-once total is not computable FROM HELD ARTIFACTS and that a producer re-run would give it** — i.e. BE33-R1's wording — so the bracket reads as today's precision rather than a permanent limit.

**What I am NOT releasing:** any statement of profitability, return, or improvement derived from these bounds. The bracket is a scale, not a result.

---

## Findings

| # | sev | |
|---|---|---|
| **BE33-R1** | LOW | `POPULATION_NOTE:78` scopes it correctly ("FROM HELD ARTIFACTS"); the per-coin field at `:198` — the one in the emitted output — does not, and reads as uncomputable in principle |
| **BE33-R2** | LOW | the publication-provenance census now exists in both `da_arm_replay_verify` and `be_read_cells`; one should be the instrument of record |
| — | — | **Item 1 verified at three levels**: tranche identity is built and discarded in `label_rows`, absent from the 1.24 GB exposure-rows artifact, absent from the feed — and recoverable only by re-running the producer, which is what BE names. **The bracket is the honest answer, not a retreat** |
| — | — | **Item 2**: LOWER ≤ TRUE ≤ UPPER for the preventable population; the bracket **collapses to exact** on a single-row action; both bounds reproduce exactly under my independent recomputation (btc 1.785x, eth 1.150x); width carried as precision with no midpoint |
| — | — | **Item 3**: BE31-R3 closed — the empty-field refusal fires and names the codomain reason; BE31-R1 closed — the AST census tells a call from a mention from a bare reference, driven three ways on modified source |

## Disposition

**Released for publication under the four caveats in item 4.** BE established non-computability before building anything, which is the right order and is the opposite of what happened with prof.py; renamed the quantity instead of engineering a number for it; and put the fix where the information is lost rather than downstream. The two findings are wording and duplication, neither blocking.

The sentence I would keep from this round is BE's own: *an honest name for an uncomputable quantity beats a plausible number for it.* That is the same lesson as the codomain predicate, one level up — do not let a computable-looking output stand in for a measurement that was never made.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `4eccb31` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No race seal opened** — the 09-01/09-02/09-03 feeds were never read; all measurements are on the re-scored **08-29 development** feed. `be_forward_day` never run. The 1.24 GB exposure-rows artifact was streamed read-only for one row. Nothing written under `data/`. `~/ctaNew-wt-be`, `-da`, `-de` never read. Worktree clean.
