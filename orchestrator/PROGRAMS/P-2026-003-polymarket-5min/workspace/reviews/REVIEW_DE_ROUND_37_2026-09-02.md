# Review — DE round 37 at `218509e` ((γ) declared and asserted, not built on the run path; the seal that seals itself)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `218509e`** (row Q-DE-55, same commit). Verified at the blob: runner
**`5d6d88155881da2d`** (2,398 lines, `EXPECTED_CHECKS = 85`), `de_head_scoring`
**`60ef48fea69e83f1`** (532, 31), protocol check **`f20323d02303baf1`** (26), v2 DRAFT
**`ec1538f1545999d1`** (184).
**Request of record:** `REQUEST_DE_ROUND_37_2026-09-02.md`. **Composed 2026-09-02T21:11:17Z.** One filing, per R-377.

**Constraints observed.** Executed in `~/ctaNew-wt-rev` at `--detach 218509e` (`data/pm_5min`
mirrored); `~/ctaNew-wt-de` / `~/ctaNew-wt-be` never read. Two mutants applied to my worktree copy
of `harmful_exposure_rows.py` for item 3 and item 7, each **restored byte-identical**
(`git status --short` **0** after). The declared OUTDIR never passed to `--run`; `derived/`
**173 before and after**; nothing written under `data/` (the two files in §8 were `stat`-ed only);
no plan file edited — the v2 DRAFT included; no unit, timer, scope or anchor; `DA_MIDNIGHT_MODE`
never set; `git worktree list` **34 at quiescence**, before and after (standing rule 10).

## 1. Counts and the three drives — CONFIRMED (item 1)

Eight DE modules, both launchers, PASS = summary = rc 0, zero stderr:
**31 / 85 / 26 / 21 / 24 / 21 / 184 / 92** — R-478 §1 reproduces.

`pin_statuses()`: **12 entries, 11 IDENTICAL + 1 ADDITIVE_DECLARED, no NOT_CALLED** — the closure
is transitive now; `phase2_arms.py` is **IDENTICAL on a moved file sha** (`3249dfc61c31b8d2` →
`ab19f5c639333bdc`) with `functions_changed: []`. `--run --outdir <scratch>` → **rc 2** by name,
no traceback, nothing created. `preflight()` refuses in **1.30 s**, naming the missing step
(`PA.tape_index(split)` over the fit's own tape).

**The prior count in the row is wrong again.** Q-DE-55 says "74 → 85"; I measured **71** at
`92c7da4` (my round-36 filing, both launchers). With Q-DE-54's "68 for 67" that is the second
consecutive round whose *previous* figure is off — filed **DE37-R4 (LOW)**, a row-hygiene matter,
not a code one: the current count is right and reproduces.

## 2. DE37-C1 — **CONFIRMED, all three facts, measured** (item 2)

**(a) The demand is the ACTION count.** `:1093-1094` builds `treated` from
`_treated_actions(per_arm[head])` and hands it to `MRC.draw`, whose `demand_from_treated` counts
**actions** per stratum. (γ)(2) needs the draw over **all above-threshold events**, so that
`|drawn_here| == |above|` in every stratum. Since every action is an above event but not conversely,
the demand is systematically **too small**.

**(b) `permuted_stream` returns a stream it has already judged wrong.** `:941-990`: when
`len(drawn_here) != len(above)` it sets `ok = False` **and carries on**, building
`list(zip(drawn_here, above)) + list(zip(rest, below))` — two **truncated** zips — with leftover
keys keeping their own value. Above values are dropped; below values are duplicated across keys.

**(c) The verdict is discarded and the predicates are selftest-only.** `run_cell` binds
`ctrl_scores, _perm_ok = permuted_stream(...)` at `:1152-1153` and **never reads `_perm_ok`**
(grep: no other occurrence outside the definition). `stream_predicates` is called at `:2051`,
`:2069`, `:2085`, `:2094` — all inside `selftest`. The only per-draw rejection on the run path is
**P4** (`:1157-1167`).

**Measured myself on the selftest's own fixture** (gens 1/2/3 = 0.9/0.8/0.1, θ 0.5, demand 1,
|above| 2):

| draw | `ok` | control stream | predicate that fails |
|---|---|---|---|
| gen 1 | **False** | `[(1, 0.9), (2, 0.1), (3, 0.1)]` | **P2** (0.8 dropped, 0.1 duplicated) |
| gen 2 | **False** | `[(1, 0.1), (2, 0.9), (3, 0.1)]` | **P2** |
| gen 3 | **False** | `[(1, 0.1), (2, 0.8), (3, 0.9)]` | **P3** (gen 2 carries an above value, undrawn) |

R-478 §2(a) reproduces exactly. In all three the stream is **returned and replayed**.

**RULING — three parts.**

1. **The demand.** The draw must be taken over **all above-threshold events per stratum**, not over
   actions: `treated` becomes the above-event keys, so `|drawn_here| == |above|` by construction
   and `ok == False` becomes unreachable for a well-formed reference. Matching on the action count
   stays where the frozen text puts it — **after** the replay, as P4. The two are different
   variables and the code currently uses one for both jobs.
2. **The placement.** P1–P3 are **stream** properties and must be computed **per draw on the run
   path**, before the replay, with the outcome and the failing predicate in the receipt. An
   `ok == False` or a failed P1–P3 is a **rejection under its own reason**, counted separately from
   P4's — a null whose rejections are all filed under "action count" hides a stream defect behind a
   matching statistic. Concretely: `rej_by_reason = {P1, P2, P3, P4}` beside `rej_by_stratum`.
3. **The falsifier.** It must make `ok` go **False on the run path** — a reference and a draw whose
   stratum has more above events than actions (the ordinary case, as the fixture above shows) —
   and assert that the draw is rejected and re-drawn, not replayed. A fixture that satisfies the
   demand by construction is not a falsifier of this; see **DE37-R2**.

**Does §5's text survive as the text the USER rules on? Yes — the text is right and the code is
what fails.** (γ) is the property the estimand needs and the DRAFT states it in the ruling's own
words. What must not happen is the package reaching the USER while the suite asserts P1–P3 on a
fixture the run path cannot produce, because that presents (γ) as achieved. Either the run path
builds it first, or §5 carries one sentence saying it is declared and not yet built.

## 3. DE37-C2 — **CONFIRMED, driven, and the pairing is worse than the finding** (item 3)

`DECLARED_ADDITIVE_SHAS: dict = {}` (`:161`); `_seal_declarations()` (`:380-401`) fills
`sha_at_declaring_tip` from `(here / name).read_text()` — the **current** file — and
`pin_statuses` compares against `_ast_sha` of that same current file. `want == now` by
construction, and no sha of the declaring tip exists anywhere in the source.

**Driven by me** (worktree copy, restored byte-identical): inserting `_tampered_marker = 1` as
`select_v2_era`'s first body statement leaves the verdict **ADDITIVE_DECLARED**,
`verify_called_code()` **PROCEEDS**, and the seal reads `sha_at_declaring_tip = 91d73910c5be44a5`
where the untampered tip seals **`3b34bdc86b1056ca`** — **the seal moved with the edit**.

**And the pairing, which the coordinator's drive did not show:** I tampered an **undeclared** called
function (`join_fills`) the same way — verdict **BLOCKING**, `undeclared: ['join_fills']`,
`verify_called_code()` **REFUSES by name**. So the pin **blocks an undeclared edit and waves
through a declared one**: the three declared functions are a **permanent exemption**, open to any
future edit of any kind, by anyone, forever. That is the severity, and it is filed as **DE37-R1**.
The docstring at `:155-160` states the opposite ("A later edit to either side re-opens the
question") — prose asserting what the predicate cannot do, the class this programme keeps closing.

**RULING on the seal's form.** The declaring-tip sha must be **a literal in the source**, not a
computation over the file it is meant to pin: extend each `DECLARED_ADDITIVE` entry to
`(reason, sha_at_fit, sha_at_declaring_tip)` — three entries, six literals, written by the seat
that declares and reviewed with the declaration. A committed sidecar keyed by the declaring commit
is equivalent and heavier; prefer the literals, because a reader of the declaration sees what was
declared without resolving a second artifact. **The falsifier must be an edited function body**
— exactly the drive above — not the in-memory dict tamper at `:1995-2001`, which exercises a state
the run path cannot produce.

## 4. DE37-C3 — **CONFIRMED**; below values should stay put (item 4)

`permuted_stream` sorts the below-threshold values descending and assigns them to the `rest` keys
in stream order (`:961-970`), so **below values move, deterministically**.

**RULING: they should stay at their own generations.** Three reasons. (i) (γ) is about the
above-threshold assignment; moving the below values introduces a **second** difference between the
arms, which is precisely what (γ) exists to eliminate. (ii) It is not inert: repost eligibility is
"score < `theta_repost` continuously for `REPOST_DWELL_S`", so a permuted below value changes
**when** a held side becomes repost-eligible — §2's number meeting §5's stream, which the DRAFT's
"re-read under §5" does not mention. (iii) Keeping them in place makes P2 hold trivially for the
below class, so the predicate gets stronger, not weaker. **If DE wants them permuted anyway, then
the assignment must be seeded-random and declared in §5** — a descending sort correlates the below
values with stream order and makes the null's repost dynamics a fixed function of the draw rather
than a sample of them — and §2's pair must be re-read against that.

## 5. DE37-C4 and C5 (item 5)

**C4 — CONFIRMED, and the fixture is constructible at the selftest's level; no feed needed.** P4's
rejection (`:1157-1167`) and `null#2` (`:1172-1180`) are asserted from the parse (`:2010-2028`)
and never driven. R-478 §2(d)'s sketch works, and I have already built its mechanism by hand: in my
round-36 filing (`aa1e44a` §2) a three-generation, one-stratum fixture with a HELD side realised
cancels **{2}** for a drawn **{3}** — a realised-count mismatch, produced in-process with no feed
and no data. Two slugs in one stratum give the count asymmetry the sketch names. "Work the run
owns" is not the right home for it: the run cannot be the first place a rejection is seen to fire.

**C5 — CONFIRMED.** P3 (`:1018`) is
`ctrl_above == {w for w in want if w in set(kc)}` — the right-hand side **filters the draw down to
keys present in the stream**, so a draw naming a key outside it passes, and P1 (a comparison of the
two **streams**) never looks at the draw. Two consequences: a malformed draw is invisible, and when
the intersection is empty P3 compares two empty sets and is **vacuously true**. The fix is one
clause: assert `want ⊆ keys(stream)` first, then the equality unfiltered.

## 6. The last substring check (item 6)

`:1673` and `:1733` are gone — the parse scan finds **one** substring predicate in the three
modules, `:1566-1568`, and its own label says "asserted at the source". The row calls it verified;
it is not a predicate over behaviour. **What replaces it:** the null's values already flow through
`arm_result`, so assert the property on the **objects** — that each accepted draw's recorded value
is identical to the `cost_adjusted_value_cents` of a replay of that draw's stream (recompute one
draw and compare), and that `run_cell`'s null block takes no `harm`-keyed argument (an AST check of
the function's signature and free names, which is the idiom this file now uses everywhere else).

## 7. The pin, otherwise (item 7)

Transitive closure over 75 modules ∩ the twelve pinned files, top-level body in the comparison, and
`phase2_arms.py` IDENTICAL on a moved file sha because the **reached set** is AST-identical — all
reproduce. **Yes, the reached set belongs in the receipt**: the verdict is a function of it, and
wiring the expensive half will reach `tape_index` and `_feature_pass`, which are not in today's
set. A receipt that records `reached: [...]` per file makes that re-opening **visible in the diff of
two receipts**; without it, an IDENTICAL verdict silently changes meaning between runs. That is
cheap and it is the same "name the population of the claim" rule the null's block now follows.

`called#1`'s falsifier (`:2029-2037`) passes a synthetic BLOCKING **row** — the dict again — but
unlike C2's case the state **is** producible on the run path, which I showed in §3 by editing
`join_fills`. So the shipped falsifier is weaker than it could be, not wrong: replace the synthetic
row with the source edit and it drives the whole path.

## 8. The v2 DRAFT and the compute (item 8)

**Still a PROPOSAL throughout** — the header and closing say so, no code cites it, and the "already
implements it" sentence (DE36-R1) is **gone**. §5 carries (γ) in the ruling's words. §1a is now
what I asked for in round 36: the feed **MEASURED** (~28.6 min, with its per-window figures), one
`arm_result` **UNMEASURED** with the synthetic number labelled a **LOWER BOUND** on the exact
fixture shape (20 slugs × one generation × one tranche × one side), the feed's `n_generations` and
`rows_per_generation` published per cell so the first real run prices the replay, and a **third
cost** declared UNMEASURED. I verified the two byte counts myself: `phase2_state_tape_v5.json`
**3,170,987,711 B** and `harmful_exposure_rows_v3_eraB.json` **1,241,115,096 B** — exact.

**The split question is raised correctly and left open** (rule 14): the §3 population spans both
fit splits, so the diagnostic scores generations the heads were fitted on; the DRAFT does not
choose and the run declares consumed splits per cell. That is the right shape — and it is a
**decision the USER should be given with the §5 numbers**, not a footnote, because it changes what
any cell can be read as.

**What must change before the package goes whole**, with items 2–4 ruled: (i) §5 gains one sentence
distinguishing *declared* from *built*, or the run path builds (γ) first (C1); (ii) §5 says what
happens to the **below** values (item 4) and §2's pair is re-read against that; (iii) the seal's
form is settled (C2) — it is not a USER question, but the package's pin claim rests on it;
(iv) the two numbers (§2, §3) and the split question travel together, since all three are read
against the same stream.

## 9. What the coordinator missed — the class (item 9)

**Are there other guards whose falsifier tampers a state the run path cannot produce? Yes — one,
and it is C1's own.** The suite's (γ) fixture (`:2049-2051`) draws **two** keys where `|above| = 2`,
so `len(drawn_here) == len(above)` and `ok` is True by construction — the one case the run path
**cannot** produce, because its demand is the action count. So P1–P3 are asserted only on the state
the defect excludes. Filed **DE37-R2**. Beyond that pair the scan is clean: no predicate that cannot
go red, no docstring standing in for code, no self-grep, one substring check (`:1566`, item 6),
`main()` catching six refusal classes with `LightGBMError` converted at `de_head_scoring`'s
boundary, and the receipt's numbers carrying their populations (`null_population`, the pin rows) —
with the reached set the one gap (item 7).

## Findings

| id | severity | where | one line |
|---|---|---|---|
| DE37-R1 | **MEDIUM-HIGH** | `:155-161`, `:380-401` | driven both ways: an **undeclared** edit BLOCKS, a **declared** one PROCEEDS with the seal moving — the three declared functions are a permanent exemption |
| DE37-R2 | **MEDIUM** | `:2049-2051` | the (γ) fixture satisfies the demand by construction, so P1–P3 are asserted only on a state the run path cannot produce |
| DE37-R3 | LOW-MEDIUM | `:1018` | P3 filters the draw to the stream's keys, so a malformed draw passes and an empty intersection is vacuously true |
| DE37-R4 | LOW | Q-DE-55 | the row's prior count (74) is wrong — 71 measured at `92c7da4`; second consecutive round |

**DE37-C1 CONFIRMED** (three facts, measured). **DE37-C2 CONFIRMED** (driven, and paired with the
undeclared case). **DE37-C3 CONFIRMED** and ruled. **DE37-C4 CONFIRMED**, and its fixture needs no
feed. **DE37-C5 CONFIRMED.** None contested.

## Disposition and round 38's order

**RELEASE `218509e` as round 38's base.** The round did real work: the closure is transitive, the
top-level body is in the comparison, `phase2_arms` is IDENTICAL by reached-set rather than by file
sha, `called#1` has a falsifier, the null carries its rejection accounting, the DRAFT's compute
section is honest and the third cost is declared. Nothing can run — `preflight()` refuses in 1.30 s
naming the missing step — so no finding here can reach an artifact.

**Round 38, in this order:**
1. **C1** — the demand over above events, P1–P3 per draw on the run path with their own rejection
   reasons in the receipt, and a falsifier that makes `ok` False **on that path** (**DE37-R2** is
   closed by the same change).
2. **C2 / DE37-R1** — literal shas in `DECLARED_ADDITIVE`, falsifier = an edited function body.
   Until this lands the pin's claim is weaker than the row states.
3. **C3** — below values stay put (or seeded and declared), and §5/§2 re-read together.
4. **C4** — drive P4 and `null#2` at the selftest's level; it is constructible without the feed.
5. **C5 / DE37-R3**, then item 6's last substring check and item 7's reached set in the receipt.
6. The DRAFT's four changes (§8) — after 1–3, so the text the USER reads matches the code.
