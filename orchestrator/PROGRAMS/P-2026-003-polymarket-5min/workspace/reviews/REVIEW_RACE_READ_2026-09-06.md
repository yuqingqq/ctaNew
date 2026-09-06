# REVIEW — the race read: **all seven preconditions hold at the artifacts**, and the result is **not consistent and could not have cleared the floor**. `day_signs = {"20260903": -1, "20260904": 1, "20260905": -1}` and `permutation_floors.neither_clears_0_05 = true`. DA's two flags are both real and both about what the artifact does not SAY; a `.v2` may add them under R-603's form, with four constraints.

**Filed** 2026-09-06T18:12Z (clock read before composing) · reviewer seat (pm-codex) · tip `2d6059b` (my refresh took me past the brief's `852f0fc`)
· **THE THIRD READER**, after DA 101 and before the coordinator reports. I opened no feed, ran no reader, and wrote nothing under `data/`. Two economic fields are quoted below by name with their values, as the read order permits at this step; nothing else numeric from the economics appears here.

---

# 1. The seven, in order, by predicate

| # | check | verdict at the artifact |
|---|---|---|
| **1** | **DA's verdict first** | `status FLAGGED`, exit 1, **`IS_A_VERIFICATION: False`**, 41 fields recomputed, **41 checks with none failing**, `n_mismatches 2` — both in `day_set`, neither about a number. DA's real path gates on the read artifact's existence by construction (DA 69, verified at REV 49 §3); both of DA's records are post-read, so **I did not re-exercise the absent-artifact gate this round** and do not claim it from these two files. |
| **2** | **`pre_state`** | every item TRUE: `declaration be_race_read_declaration_v4.json`, **`declaration_sha256 = a741b4d6b5ac7f59…`** (the post-BE-65 chain head, the digest REV 74 §1.2 recorded and REV 77 §3 required); `zero_markers_before_the_act true`, `n_existing_OPENED_markers 0`; `declared_result_absent_before_the_act true` with `declared_result_name be_race_read_result_v1.json`; **`marker_dir_realpath = /home/yuqing/ctaNew/data/pm_5min/derived`** — the ledger, in the artifact rather than in a rule; `as_of 2026-09-06T17:56:53Z` from the clock; all three days `pin_present / pin_exists_true / feed_on_disk / feed_matches_its_pin` **true**. DA independently checked `as_of_precedes_every_marker_stamp` — the ordering proved from the stamps, not from the code. |
| **3** | **the `decl` provenance** | REV 77 §2.1 **closed, and better than I asked**: `decl_source = "resolve_days() against the declaration chain head, supplied by the caller and RE-VERIFIED here against a fresh resolution (same declaration, same digest)"`, `decl_supplied_by_the_caller true`, `decl_is_the_chain_head true`, `decl_declaration_sha256` = the head's digest, and `why_this_field_replaced_decl_was_injected` citing the review. I asked for either/or; BE did both — supplied **and** re-resolved and compared. |
| **4** | **exactly three markers** | three, 292 B each, stamped **17:56:54Z — four seconds before the unit opened at 17:56:58Z**; one per declared day, **no fourth**; each carries `day`, its `pin` (the feed's own digest, e.g. `19d03c5d…` for 09-03, equal to that day's `file_sha256_after` in the byte-identity block), `utc`, and a `why` stating that consumed is the safe direction. |
| **5** | **G and the floor** | `day_set`: `READABLE = [20260903, 20260904, 20260905]`, `G_declared 3`, `G_computed_from_the_set 3`, `G_agrees_with_the_declaration true`, the declaration's digest carried. `permutation_floors`: optimistic and pessimistic both G 3, `WHICH_ONE_IS_RESOLVED: "the CONSERVATIVE one"`, `resolved_best_possible_adjusted_p 0.25` — **the declaration's own arithmetic (2⁻³ × m = 2), recomputed independently by DA and agreeing.** |
| **6** | **byte identity** | `all_unchanged true`; `digest_covers_every_byte_parsed` true **per day**, with each day's parsed-stream hash beside the file's; `computed_not_asserted: "the coverage claim is a predicate over the parsed stream's own hash, not a literal"`. DA's `one_pass_hashing` says the same from its side: every byte parsed is the byte hashed, in one pass, and nothing parsed is used before the digest matches. `on_mismatch_declared: "the read is VOID -- enforced"`. |
| **7** | **what a refusal left behind** | **No refusal occurred.** The artifact carries no refusal field and a complete per-day result; the three markers and the single declared result exist and `writes` says `{artifact: be_race_read_result_v1.json, and_nothing_else: true}`. The state the GO warned of — markers written, no result — is not the state on disk. |

**All seven hold.** The act is auditable from its own artifact, which was the point of the pre-state block.

---

# 2. The estimand, as the declaration frames it

The statistic is the interim's **primary, MATCHED_VOLUME**, net cents on the incumbent's increment, under `ruling: "R-588: OPTION A, NO RE-SEAL … because that is what 09-01 and 09-02 were read with and changing it after seeing two days would be a choice after seeing. BY_THRESHOLD is reported, never primary (rule 7)."` The artifact states its own limit up front: `R_529_A_UP_FRONT: "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY AND NEVER A HOLM-CLEARING VERDICT (R-529(A))."` And `decides_nothing: "REPORTED (rule 14)."`

**The two fields, quoted by name with their values — and nothing else numeric:**

> **`day_signs` = `{"20260903": -1, "20260904": 1, "20260905": -1}`**
> **`permutation_floors.neither_clears_0_05` = `true`**

**The one line for the coordinator's report:** *the three readable days are **not** consistent in direction — two of the three signs are negative and one is positive — and the permutation floor at G = 3 with m = 2 does not clear 0.05, so **no unanimity was available to clear Holm even in principle**.* Both halves matter and neither is a result about the world: the first says the days disagree; the second says that under the declared design this read could not have produced a significant number whatever the days had done — the ceiling my REV 44-era filing computed at G = 5 is 0.25 here at G = 3, and it was declared before the act, recomputed by DA after it, and stated in the artifact by the reader itself.

**Nothing else is quotable from this read and nothing else should be quoted.** No per-day magnitude, no per-coin value, no BY_THRESHOLD figure appears in this filing, and none belongs in the coordinator's line.

---

# 3. DA's two flags, and the `.v2`

Both flags are **real, and both are about what the artifact does not say** — neither touches a recomputed field (all 41 match).

1. **`day_set.20260901.ABSENT` / `.20260902.ABSENT`** — the pins name five days; the artifact mentions three. Declaration v4 §3.3 requires every pinned day to be **said**, with `READ_BUT_UNRECOVERABLE` as the expected status for those two, precisely so that a silently smaller G cannot be reported as the declared one (R-600's hazard, and the reason the by-name refusal exists at all). **Silence is not compliance**: a reader of this artifact alone cannot tell a day that was declared unrecoverable from a day that was dropped.
2. **No source identity** — no commit, no reader digest. Every other result-bearing artifact in the programme carries one (rule 22's class), and this is the one that cannot be re-run.

**May a `.v2` add them? YES — this is the correctable kind, under R-603's form and REV 72's asymmetry.** Nothing new is *published*: the two days' statuses are **copied from the pins**, which are declarations that existed before the act, and a reconstructed source identity is a provenance fact resolvable at git. No number changes, and `day_signs` and the floors are untouched. **Four constraints, from the programme's own precedents:**

- **v1 untouched; the `.v2` supersedes by the `{path, sha256}` pair** (R-608), and v1 stays as provenance.
- **The reconstruction rides in the KEY** — BE 60's `.v2` and REV 59 §6: `producing_code.status: RECONSTRUCTED_NOT_A_STAMP` as the block's first field, `builder_commit_RECONSTRUCTED` rather than `builder_commit`, naming BE's report and the git objects (`c4c0d0d`, reader `32c0e4b9…`) as the source, and stating plainly that it was **not** captured at run time. An automated reader keying the plain field name must not get a value that looks stamped.
- **The two added days carry their status and nothing else** — `READ_BUT_UNRECOVERABLE`, copied from the pin, with an explicit statement that they are **not** in `READABLE` and that **G remains 3**, so no reader can infer G = 5 from their appearance.
- **It may not recompute or restate any statistic**, add any field derived from the feeds, or touch `day_signs`, `permutation_floors` or `byte_identity`.

With those, the `.v2` makes the artifact compliant with the declaration it was read under, which is the only thing still missing from it.

---

# 4. BE 70 against the GO and REV 77 §3

What I can check at the artifacts: **no re-run verification is claimed anywhere** (`writes: and_nothing_else true`; nothing in the result offers a re-read as evidence), and **the result names no expected direction** — `R_529_A_UP_FRONT` states the opposite, that direction and consistency are all this read can establish. The pre-flight items the GO required are visible as `pre_state` fields and all true (§1.2). **BE's own report text I have not seen**, so I do not certify what BE said in it — only that the artifact BE produced makes no prohibited claim and carries every field the GO required it to carry.

---

# 5. Not established

- **I did not re-run the reader, open a feed, or write anything under `data/`.** The three markers and the result are as BE left them; the two DA records are as DA left them.
- §1.1: DA's absent-artifact gate was **not** re-exercised this round (both records post-date the read); it was verified at REV 49 §3 and is asserted here only as design.
- I did not recompute the statistic or any per-day value, and I did not open the feeds to check the parsed-stream hashes — §1.6 rests on the reader's computed predicate and DA's independent recomputation of the same 41 fields.
- BE 70's report and DE 105/DA 98's rounds are outside what I read.
- The two quoted fields are the only economics in this filing; the floor (0.25 at G = 3, m = 2) is the declaration's own arithmetic, stated in the brief and recomputed by DA.

**Routing:** BE — the `.v2` in §3's shape, if the coordinator rules for it. Coordinator — the one line in §2, and the ruling on the `.v2`.

**Context ≈ 42 %.**
