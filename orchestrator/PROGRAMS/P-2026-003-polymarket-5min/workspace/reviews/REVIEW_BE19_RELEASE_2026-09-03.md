# Re-review — BE round 19: the release decision

reviewer: pm-codex · filed 2026-09-03T07:58Z · pinned tip **`cd69879`** (BE19 at `1a69115`, row `Q-BE-244` at `cd69879`)
executed in `~/ctaNew-wt-rev` at the tip. **No seal opened.** `be_forward_day.py` read, never run. Nothing written under `data/`.

# **AMEND.**

**The one thing that must change:** `require_operating_point` must carry `verification` and `coin` into the object it returns. It does not, `_verification_binds` requires both, and so **the production accessor chain refuses its own fence**. The suite's positive control passes only because it hand-injects the two dropped keys.

Driven, at the tip, on the real committed declaration:

```
require_operating_point returns:
  ['_operating_point_token','budgets','causal_declared','causal_verification_note',
   'causal_verified_against_scored_population','declaration_sha16','declared_at_utc',
   'declared_by','derived_from_split','form','provenance_declared','provenance_verified',
   'selected_by_this_module','source','theta_frozen']
  carries 'verification': False   carries 'coin': False

op_declaration_for("btc") -> require_operating_point -> require_fenced_op(rows=…)
  REFUSED  OperatingPointUndeclared: "the operating point carries no `verification` block…"

the SUITE's shape — be_operating_point.py:355
  _fo = dict(_f, coin=_c, verification=_op["verification"])
  PASSED
```

`op_declaration_for` **does** attach `verification` and `coin`; `require_operating_point` drops them both on the way through. So the control at `be_operating_point.py:356-362`, which reports *"the REAL declaration for btc passes the BINDING fence"*, is run on a shape production cannot produce — SEAT_PROTOCOL rule 16's named class, *a fixture must never supply what the code under test should produce*, and rule 17's *suite-green is not pipeline-wired*.

I want to be exact about the consequence, because it cuts both ways. **Nothing can be scored through this defect** — it fails closed, and no driver assembles an operating point yet (`grep` over `be_forward_day.py` and `be_forward_family.py` finds no call to `op_declaration_for`, `require_operating_point`, `require_fenced_op` or `increment`). But the release asked for is *scoring a forward day on this path*, and the path's only assembly chain refuses. Worse than the refusal is what the refusal invites: whoever wires the producer will meet it, and the shape sitting in front of them as the worked example is the suite's `dict(_f, coin=…, verification=…)` — **attaching the binding at the call site, which is precisely where a caller can choose it.** The fence must carry its own evidence, not receive it from whoever is calling.

The fix is small and local: pass `verification` and `coin` through in `require_operating_point`'s `out`, add `verification` to `_OP_TOKEN_FIELDS` so it cannot be swapped after the fence either, and delete the hand-injection from the two controls so they exercise the production shape.

---

## The attacks, all re-run

### (1) BE17-R1 — closed. My forgery and every variant of it now refuse.

| attack | result |
|---|---|
| my BE17 forgery: truthy token, no split, retrospective theta | **REFUSED** — *"the operating point's token does not recompute (anything-truthy… against 7e275899…)"* |
| forgery + a days list present but meaningless (`["not-a-day"]`) | **REFUSED** — token |
| forgery with a **self-computed token that recomputes** (`_op_token` is public, no secret) | **REFUSED** — *"carries no `verification` block"* |
| …+ a **fabricated** verification block asserting my own numbers | **PASSED** — the stated residual, see (3) |

The second row is worth naming: a meaningless-but-present days list does not get past the fence, because the token binds `derived_from_split` and a hand-built op cannot both recompute and be arbitrary — unless the forger computes the token too, which the third row does, and then the verification requirement catches it.

### (2) BE17-R3 — closed, and the check is **not** tautological.

The honest op now reports `token_recomputed: True`. That alone would prove nothing — both sides are `_op_token` over the same fields. What makes it a check is that it binds the object **after** the fence has seen it. Five post-fence mutations, each driven, each refused **by name**:

```
theta_frozen        REFUSED  token does not recompute (68113bf1… against 714d6…)
derived_from_split  REFUSED  token does not recompute (…       against 89d97…)
form                REFUSED  token does not recompute (…       against 11f8a…)
provenance_declared REFUSED  token does not recompute (…       against 55abb…)
source              REFUSED  token does not recompute (…       against 53d1e…)
```

And the four the token does **not** cover are caught by `_verification_binds` instead, each with its own message:

```
verification.all_coins_reproduce=False        REFUSED  "the recomputation did NOT reproduce…"
verification.declared_days_match_the_rows=False REFUSED "asserted rather than derived"
verification.rows_artifact_sha256=0*64        REFUSED  "the verification was run over rows … but the declaration names …"
verification.recomputed_theta_map swapped     REFUSED  "the numbers were changed after the verification"
```

BE's diagnosis of the old defect is confirmed at the code: the token was `_op_token(decl)` over the raw `provenance` while `want` was rebuilt from `provenance_verified`, which gains `verified_by` keys — structurally incapable of agreeing. Carrying the raw block through as `provenance_declared` is the right repair, and the assertion now has content.

### (3) BE17-R2 — structurally closed, and I verified the binding myself rather than reading its receipt.

**My exact attack now refuses.** The real committed declaration, thetas swapped for the scored rows' own cutoff, `theta_map_sha16` recomputed as the builder does:

> `REFUSED: the verification does not bind these numbers to those bytes: the recomputed map for 'btc' {'5%': 1.0845103653350594, '10%': 0.7230267681941027, '15%': 0.5249913727523682} is not the theta this operating point carries {'5%': 0.99, …} — the numbers were changed after the verification.`

**And I ran the binding independently** — `verify_declaration_by_recomputation()` called read-only, nothing written, **805 s**:

| field | my run |
|---|---|
| `rows_artifact_sha256` | `19a50195c34d0af21aba81b9f6b9501ea58656212d8448ddd8cc219da26d0f08` |
| `rows_sha_matches_declaration` | True |
| `declared_days` / `days_derived_from_the_rows` | `['2026-08-24','2026-08-25']` / **identical** |
| `declared_days_match_the_rows` | True |
| `all_coins_reproduce` | **True** |
| per coin | btc `matches True`, `max_abs_difference` **0.0**; eth `matches True`, **0.0** |
| counters | 1,135,943 seen · 1,135,943 in split · **1,135,930 scored** · 13 without features · 471 slugs · 0 missing archive |

**Every one of the eight substantive fields equals the committed block** (a first pass reported a difference; it was JSON key ordering only — normalised, `differing: NONE`). So the receipted claim is not taken on trust: I reproduced it from the bytes.

**Can a declaration still name days the rows do not contain? — YES, and this is BE19-R2 below.** Not through the number binding, which holds, but through the overlap check.

**Can the recomputation be satisfied by a verification block that was never run? — YES, and BE says so.** I built one: a made-up rows sha, made-up thetas, `all_coins_reproduce: True`, and a token computed over my own fields. It passed and produced a complete increment. **Is the residual correctly bounded?** Yes, with one clarification BE's wording leaves implicit: `require_fenced_op` **opens no file** — every check it makes is internal consistency of the object it is handed. The bound is therefore *an audit outside the path*, not a check inside it. That audit exists, costs one command over known bytes, and I have now run it (805 s, above). Stated that way the residual is correctly bounded and **does not reopen R2**: R2 was that the honest declaration could be defeated by editing one field, and that is closed — defeating it now requires fabricating a receipt of a computation, which is a different act and a recorded one.

### (4) The renamed field — gone from the fence, **not** from the module.

`declared_split_does_not_intersect_scored_population` is what the fenced output carries and what the code computes (`bool(declared) and overlap == []`), and `"causal_verified_against_scored_population" in fence` is **False** — driven. But `require_operating_point` still emits `causal_verified_against_scored_population: False` at `be_forward_metric.py:438`, asserted at `:1229`. Hard-False and accompanied by a note, so the risk is small; the name is nonetheless the one we agreed must not appear in a receipt.

---

## Findings

### BE19-R1 — **BLOCKING** — the production accessor chain refuses its own fence, and the control that says otherwise is a fixture

`require_operating_point` drops `verification` and `coin`; `_verification_binds` requires both; `op_declaration_for` supplies both and they are discarded in between. `be_operating_point.py:355` re-attaches them by hand before the positive control runs.

> **AMEND clause.** Carry `verification` and `coin` through `require_operating_point`'s `out`; add `verification` to `_OP_TOKEN_FIELDS` so it is bound like every other field; remove the hand-injection at `be_operating_point.py:355` and the equivalent in `be_forward_metric.selftest`, so both positive controls exercise the shape production produces. One driven check settles it: `require_fenced_op(require_operating_point(op_declaration_for(c)), "10%", rows=…)` must PASS with no keys added at the call site.

### BE19-R2 — MEDIUM — the overlap is computed against the **asserted** split while the **verified** one sits beside it

`_verification_binds` reads the block's boolean `declared_days_match_the_rows` and never compares the block's `declared_days` to `op["derived_from_split"]["days"]`. So the two can disagree. Driven — honest op, `derived_from_split.days` relabelled to days the rows do not contain, token recomputed:

```
op relabelled to ["2026-01-01","2026-01-02"]     PASSED
  n_declared_split_days 2 | scored_split_overlap []
  declared_split_does_not_intersect_scored_population: True
  verification.declared_days: ['2026-08-24','2026-08-25']    <-- the real derivation days
```

The theta stays honest — it is bound to `recomputed_theta_map` — so this does **not** reopen R2's number binding. What it defeats is the overlap guard's *subject*: the fence compares the scored population against a list a caller asserted, while the list derived from the rows is in the same object. The right list is already there.

> **AMEND clause.** Compute the overlap against `verification["declared_days"]` (derived from the rows), and refuse when it disagrees with `derived_from_split["days"]` — a split asserted differently from the split verified is exactly the discrepancy the verification exists to expose.

### BE19-R3 — LOW — `causal_verified_against_scored_population` survives at `be_forward_metric.py:438`

Gone from the fence, still emitted by `require_operating_point` and asserted at `:1229`.

> **AMEND clause.** Rename it there too (`derivation_not_yet_checked_against_a_scored_population`) or drop it and keep `causal_verification_note`.

---

## Also verified closed (my earlier findings, driven not read)

| was | now |
|---|---|
| **BE17-R4** — the tolerance anchor was a constant in the file it guarded | `earliest_commit_with_todays_values()` derives it from the file's own history: **`1e9b6626e23d` at 2026-09-03T05:39:05Z**, `values_sha16 4a2a28ceb124a3fd`, 4 commits examined, 4 carrying today's values, **`no_constant_consulted: True`**. It corroborates the constant instead of trusting it |
| **BE17-R5** — `forward_eligible` overclaimed | gone as a key. The return now carries `cutoff_is_a_function_of_these_scores`, `not_read_off_these_scores_at_evaluation_time` and `this_instrument_can_falsify_but_cannot_certify` |
| the `_emits_feed` known-bad falsified 1 of 3 conjuncts | each conjunct is mutated separately and turns both its own field and `emits_feed` False |
| the stale comment at `harmful_forward_scorer.py:600-602` | gone |

Suites at the tip, both launchers, rc 0: `be_forward_metric` **99**, `be_operating_point` **19**, `be_forward_recon` **27**, `harmful_forward_scorer` **75**.

---

## Findings table

| # | sev | finding |
|---|---|---|
| **BE19-R1** | **BLOCKING** | `require_operating_point` drops `verification` and `coin`, so the production chain `op_declaration_for → require_operating_point → require_fenced_op` REFUSES; the positive control passes only because `be_operating_point.py:355` hand-injects them |
| **BE19-R2** | MEDIUM | the overlap uses the asserted `derived_from_split.days`, never the verified `verification.declared_days`; a relabelled op passes with `declared_split_does_not_intersect_scored_population: True` |
| **BE19-R3** | LOW | `causal_verified_against_scored_population` still emitted at `be_forward_metric.py:438` |
| — | — | **BE17-R1, R2, R3 closed**; the residual correctly bounded, its falsifier run by me at 805 s reproducing every field; R4, R5 and both LOWs closed |

## The release

**Not released.** What was asked for is *scoring a forward day on this path*, and I cannot release a path whose own accessor chain refuses and whose passing control is a hand-assembled shape. This is one small change away: BE19-R1's four-line fix plus a driven check that the production chain passes with nothing added at the call site. **BE19-R2 and R3 should ride in the same batch but neither blocks on its own.**

**What I will release on, when R1 is fixed** — stated now so the next round is a re-drive and not a re-argument: the three HIGHs are genuinely closed. A bare theta is unrepresentable; a forged token refuses; an undeclared split refuses; the numbers are bound to the bytes by a recomputation I reproduced independently over 1,135,930 rows with `max_abs_difference` 0.0 on both coins; and my own two attacks are red against the real declaration.

**Open but not blocking, and named so they are not mistaken for cleared:**
1. **The residual.** A fabricated verification block passes — `require_fenced_op` opens no file. Bounded by an out-of-band audit that costs 805 s and that I have now run once. It should be re-run whenever the declaration changes, and the receipt should say when it was last run by someone other than its author.
2. **The decision metric has never been reconciled against any published number**, and cannot be from existing artifacts — `increment()` is BY_THRESHOLD, iteration 011 is BY_COUNT. BE states this; it remains true after this round.
3. **Which artifact ought to be scored** (`PM_PLUS_FINE` / LINEAR vs LGBM_PINNED) is a freeze-level ruling. Not reviewed, not touched.
4. No driver yet assembles an operating point, so the producer wiring is still ahead — and BE19-R1 is exactly the seam it will land on.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `cd69879` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No seal opened.** `be_forward_day.py` read, never run. **Nothing written under `data/`** — the recomputation was called as a function, not through `--verify`, so no artifact was rewritten; the attacks passed forged mappings as arguments and modified no file. No unit, timer or anchor; `DA_MIDNIGHT_MODE` never set. `~/ctaNew-wt-be`, `-da`, `-de` never read. `git worktree list` **34** at quiescence, worktree clean.
