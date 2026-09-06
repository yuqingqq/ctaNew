# REVIEW — **v5 is a valid pre-registration for the second read**, and it answers the "how can a declaration naming today be *pre*-anything" objection better than the objection was put: the bar is **before any named day CLOSES**, not before the first begins, and I verified the substance — **no 09-06 feed or score exists**. One decision in it is not the declaration's to make and belongs to the user: **that a second read happens at all.** DA 112's form is right and is the third instance of one rule.

**Filed** 2026-09-06T20:36Z (clock read before composing) · reviewer seat (pm-codex) · tip `c5b8a48`
· At the artifact; nothing written under `data/`; **nothing opened, no pin read, no marker created.** *Landing:* `land_register_row.sh` is register-only; this filing lands by file pathspec under the same post-condition.

---

# 1. The day set, G and the floor

| field | at the artifact |
|---|---|
| `population.READABLE` | `20260906, 20260907, 20260908, 20260909` |
| `population.CONSUMED_BY_THE_FIRST_READ` | `20260903, 20260904, 20260905`, with `nothing_from_the_consumed_days_enters_this_read` |
| `population.READ_BUT_UNRECOVERABLE` | `20260901, 20260902` — **every pinned day is said**, which is the finding DA raised against the first read's artifact (REV 78 §3) closed at the declaration rather than after the fact |
| `G` / `why_G_is_4` | 4, **the USER's ruling R-531**, "not chosen here and not derived from what any artifact shows"; the day set is **derived** — "the four consecutive UTC days that follow the first read's last consumed day" |
| `permutation_floor` | `2^-G × m`, G 4, **m 2**, `best_possible_adjusted_p 0.125`, `clears_0_05 false`, `computed_here_not_quoted` |
| `read_horizon` | last day closes + **`close_lag_minutes: 60`, stated as "A DECLARED BOUND, NOT A MEASUREMENT… no close lag has been measured for this pipeline and none is asserted here"** |
| pins | the WHOLE set asserted before the first marker, "because the act consumes day by day and a day spent cannot be given back when a later day proves unpinned" |

**Three things I would have asked for and did not have to.** The day set is *derived* from the first read's last consumed day, so choosing the window was not a choice. The floor's `m` is stated rather than implied. And the horizon's lag is **named as a bound and not as an observation** — the discipline this programme reached the hard way (R-632's "records the correlation, claims no mechanism").

## 1.1 The pre-registration bar — v5 states it better than the brief did

The brief's bar is "pre-registration is only pre- if it precedes the first day it names". v5 answers with `declared_before_any_of_its_days_closed`: `written_at_utc 2026-09-06T20:27:5xZ`, `first_declared_day_closes_utc 2026-09-07T00:00:00Z`, and the reason — *"every day this declaration names is still OPEN at the moment it is written, so the day set, G, the floor, the statistic and the horizon cannot have been picked on anything anyone has seen."*

**That is the correct bar, and it is the stronger one.** A day's sign cannot exist before the day closes and its feed is built; what rule 11 forbids is choosing on what has been *seen*, not on what has *elapsed*. **And I checked the substance rather than the claim: `/home/yuqing/ctaNew_forward_runs` holds no `20260906` directory or artifact** — there is nothing about 09-06 to have seen. The declaration puts both timestamps side by side so a reader can check it without me.

---

# 2. The statistic, and what may be quoted

`statistic` is MATCHED_VOLUME as the interim's primary with the definition carried in the artifact, `BY_THRESHOLD` **reported and never primary** (rule 7). `R_529_A_UP_FRONT` — *"THIS READ ESTABLISHES DIRECTION AND CONSISTENCY AND NEVER A HOLM-CLEARING VERDICT… Nothing numeric is quotable from it beyond `day_signs` and `permutation_floors.neither_clears_0_05`"* — is the same two fields I was permitted at REV 78, fixed in the declaration **before** the read rather than in a brief afterwards. **AGREED.**

# 3. The new family

`result.family = be_race_read2_result`, `artifact = be_race_read2_result_v1.json`, and **the reader takes the name from this field, not from a constant in its own source**. So the first read's chain (`be_race_read_result` v1/v2) cannot be extended by the second read even by mistake, and a resolver walking either family gets one read's history. **AGREED** — and the "named by the declaration, not by the source" clause is the part that makes it hold when someone later copies the reader.

# 4. Consumption

Markers before the first byte; **"ANY content at a marker path means CONSUMED"** — a half-written, empty or non-object marker refuses rather than raising out of the guard, which is REV 76 §5(b) and REV 77 §2.2 closed *in the declaration's own words*, so the next reader inherits the property rather than the fix.

# 5. What a reader may not infer — and the one sentence I would add

The three prohibitions are right, and the second is stronger than I expected: *"The two reads are NOT pooled by this declaration. Pooling would be a third question with its own multiplicity and needs its own declaration written **before either read is opened**."* Since the first read **is** opened, that sentence **forecloses pooling permanently** — which is the correct outcome and worth the coordinator noticing, because it means the two reads can never be combined into a G = 7 statement.

**What I would add, for the reader who arrives holding only the second artifact:** *neither read's floor prices the other's existence.* Two reads are two chances; each artifact's floor counts arms (m = 2), not reads. `R_529_A_UP_FRONT` already forbids any Holm-clearing reading, so this is a clarification rather than a gap — but a reader who sees "0.125, and the signs were consistent" in isolation should be told, in that artifact, that a second test existed.

# 6. The falsifiers

All four named at the declaration: a day outside READABLE refused **naming what was added and what was dropped**; a read before the horizon refused by name; a declared day with no pin — or a pin with no digest — refusing **the whole read before any marker is written**; and the non-head census passing with the reader naming v5. The third is the one that matters most for an act that consumes day by day, and it is the one BE stated the reason for.

---

# 7. The ruling

**v5 is a valid pre-registration for the second read.** The day set is derived, G is the user's, the statistic is the interim's with no re-seal, the floor is arithmetic with m stated, the horizon's lag is declared as a bound, every pinned day is said, the result has its own family, and the declaration opens nothing (`opens_nothing`, `decides_nothing`).

**The one thing in it that is not the declaration's to decide: that a second read happens at all.** R-531 rules the race G = 4 and directional; whether it contemplated a *second* four-day race after the first returned inconsistent signs is not something this artifact can establish, and no field in it claims to. Everything else here is derivation; this is a decision, and it is the user's — **the honest form is that the declaration is valid, and the authorisation to spend four more days rests on R-531 being read as authorising this race.** If it is not, nothing in v5 supplies it. **Put that to the user before the horizon, not after.**

# 8. DA 112 — the right form, and the third instance of one rule

Yes. `da_race_read_verify` asserting the head is v4 and indexing the first read's pins by the head's READABLE days is the defect the moment a v5 exists: **a verifier of a past act was resolving a declaration by the family's CURRENT head.** Resolving by the artifact's own `pre_state.declaration_sha256` is exactly right.

And it is the **third instance of one principle**, which should now be stated once rather than fixed three times: the seal scope resolved from the run's carrying commit and closure digest rather than the current design (REV 75 §2); a receipt's design pin judged by its own pair rather than the chain head (REV 73/§2); and now the read's declaration. **An instrument verifying a past act resolves every declaration by the pair the act recorded — never by the head.** The head is for writers; the pair is for readers of history.

---

# 9. Not established

- I read v5 and checked the absence of any `20260906` artifact under `/home/yuqing/ctaNew_forward_runs`; I did not audit every path where a 09-06 score could appear.
- I did not drive the four falsifiers — they are BE's, reported and not re-run by me; the reader's behaviour under v5 is unexercised until the horizon.
- §5's foreclosure of pooling is my reading of the declaration's own sentence, not a ruling by the coordinator or the user.
- §7 rests on R-531 as summarised in the brief and in v5's `why_G_is_4`; I did not re-read the user's ruling.
- I opened nothing and created no marker; the first read's three markers and the ledger are untouched.

**Routing:** USER (through the coordinator) — §7, before the horizon. BE — §5's one sentence into the result artifact's template. DA 112 — §8's form, and the rule stated once. Coordinator — §5's foreclosure noted, and §8 into the pattern file.

**Context ≈ 66 %.**
