# REVIEW — **GO for the real read, subject to ONE code change: the artifact would say `decl_was_injected: true` about itself.** The CLI passes `decl=_dc`, so the field the whole test's credibility rests on reads TRUE on the real path while the prose beside it says "on the real path nothing is injected". Everything else in BE 68 drives: the marker directory is guarded, five of six malformed markers refuse as consumed, `pre_state` carries every item of REV 76 §6 condition 3, and a stale pin cannot pass.

**Filed** 2026-09-06T17:06Z (clock read before composing) · reviewer seat (pm-codex) · tip `7d51e1b`
· **NOTHING CONSUMED.** Scratch declaration, scratch feeds, days `20991230/31/32`; **the ledger still holds 0 OPENED markers**; no `--open`, no real pin, no real feed touched.

---

# 1. BE 68, driven

| item | result |
|---|---|
| **(1)** `resolve_marker_dir` with a scratch `PM_DATA_ROOT` | **`DataRootRefused`, naming the resolved root** — before any marker is read or written |
| " with the real ledger root | `/home/yuqing/ctaNew/data/pm_5min/derived` |
| " with an explicit scratch `outdir` and no `fixture=True` | **REFUSED** — a scratch drive can no longer write a marker anywhere without declaring itself a fixture *with a reason* |
| **(2)** a valid marker / `{ partial` / an **empty file** / a marker naming a **different day** / a **directory** at the path | **all five REFUSE as consumed**, naming the day and the path (REV 76 §5(b) closed, and three shapes I had not asked for) |
| **(2)** valid JSON that is **not an object** (`[]`) | **`AttributeError: 'list' object has no attribute 'get'`** — see §2.2 |
| **(3)** `pre_state`, computed **before** any marker is written (the order in `read()`: declaration → guarded marker dir → `pre_state` → marker guard → result-name guard → write) | `as_of` from the clock; `declaration` + `declaration_sha256`; per day `{pin_present, pin_exists_true, pinned_sha256, feed_on_disk, feed_sha256_now, feed_matches_its_pin}`; `existing_OPENED_markers`, `n_existing_OPENED_markers`, `zero_markers_before_the_act`; `declared_result_name` + `declared_result_absent_before_the_act`; `marker_dir` **and `marker_dir_realpath`** |
| **(3)** a **stale pin** | mutating a feed flips `all_feeds_on_disk_at_their_pins` to `False`, and `read()` **REFUSES by name** quoting both digests — a `pre_state` from stale pins cannot pass |

**Every item of REV 76 §6 condition 3 is in the block, plus the realpath** — which answers "checked from the ledger or from a worktree copy" inside the artifact itself rather than in a rule. The marker guard is ordered **before** the result-name guard with the reason stated (a guard that names the days beats one that is true of the whole read — REV 48 §1.6's lesson, applied by BE unprompted). **AGREED.**

# 2. Two defects

## 2.1 `decl_was_injected` will read TRUE on the real path — **fix before the act**

`read()` records `"decl_was_injected": decl is not None`, and the prose beside it says *"the CLI's `--open` builds it from `resolve_days()` … so on the real path nothing is injected"*. But the CLI **does** pass it: `read({d: Path(_feeds[d]) for d in _dc["days"]}, decl=_dc)`. So the read artifact — the one permanent record of the programme's second test — would carry a boolean saying the declaration was injected, contradicted by the sentence next to it.

The behaviour is right (`_dc` is `resolve_days()`'s output). **The artifact's self-description is wrong, and there is no `.v2` for a read that cannot be re-run** (REV 75 §4.1 was correctable because the day could be re-described; a consumed read cannot). Two-line fix, either way: the CLI passes no `decl=` and lets `read()` resolve it, **or** the field becomes `decl_source: "resolve_days() against the chain head" | "supplied by the caller"`. **This is my one condition.**

## 2.2 A marker that is valid JSON but not an object still raises — not blocking

The guard now treats an unparseable, empty, wrong-day or directory marker as consumed; `[]` reaches a `.get` and raises. No writer produces it (`write_open_markers` writes an object), so it is not a GO blocker — but it is the last shape of "a crash is not a verdict" on the one-shot path, and the fix is the one already applied one line up: **anything at that path means consumed; parse only inside the try.** → BE 69, after the read.

# 3. The GO's text

I have not seen the coordinator's draft; against my REV 76 §6 the five conditions are the right ones and BE 68 has made three of them *checkable inside the artifact* (the declaration digest, the pre-state, the marker realpath). **What the GO must add**, given §2.1 and what BE built:

- **Name the commit** the reader runs from, and state that `pre_state.declaration_sha256` must equal `a741b4d6…` — the check is in the artifact now, so the GO can require the value rather than the procedure.
- **Require `zero_markers_before_the_act: true` and `declared_result_absent_before_the_act: true` in the emitted artifact**, not only in the operator's pre-flight — the artifact is what a later reader has.
- **Say the sentence that cannot be automated**: after the markers are written, a refusal leaves the three days consumed and no result, and there is no retry. BE's `what_a_marker_IS` states the marker's durability; the GO must state the consequence.

**What it must not say:** it must not promise a verification by re-running the reader (a second call refuses on the markers — correct, and not a verification), and it must not name an expected direction or magnitude in any form.

# 4. VERDICT

**GO** for the real read **subject to §2.1** (the `decl` field corrected and the reader's selftest re-run). Everything else I was asked to check drives as declared, and the pre-state block makes the act auditable from its own artifact for the first time.

**What I will check myself at REV 78, in this order** — after DA's `--real` verifier has run first (runbook §7, the read order):
1. DA's verifier's verdict on the read artifact **before any number is quoted**, and that it gated on the artifact's existence rather than a flag.
2. The artifact's `pre_state`: `declaration_sha256 = a741b4d6…`; three pins `exists: true` with `feed_matches_its_pin` true; `zero_markers_before_the_act`; `declared_result_absent_before_the_act`; `marker_dir_realpath` = the ledger; `as_of` from the clock.
3. `decl_source` (or its replacement) reading what §2.1 requires.
4. **Three** OPENED markers in the ledger, one per declared day, each naming its day and pin — and no fourth.
5. G = 3 from the declaration, the floor resolved conservatively, and the day set equal to the READABLE set.
6. That the byte-identity block shows every parsed byte covered, and `all_unchanged`.
7. What a refusal, if one happened, left behind — the markers, the absent result, and whether the artifact says so.

# 5. Not established

- Nothing consumed; the ledger's marker count is **0** at filing, verified after my drives.
- §1's rows are driven through `resolve_marker_dir`, `assert_not_already_opened`, `pre_state` and `read(..., consume=False)` on scratch; **I did not run a consuming read** (a scratch feed is a one-arm feed and `be_read_cells` refuses it by name — REV 76 §7).
- I have not seen the GO's drafted words; §3 is written against my REV 76 §6, not against the draft.
- The 09-05 run, DE 105 and DA 98 are not in this batch.

**Routing:** BE — §2.1 **before the act**; §2.2 after the read. Coordinator — §3's three additions and the two prohibitions.

**Context ≈ 39 %.**
