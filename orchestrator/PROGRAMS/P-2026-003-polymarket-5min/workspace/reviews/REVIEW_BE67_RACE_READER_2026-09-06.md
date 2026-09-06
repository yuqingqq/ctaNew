# REVIEW — **the reader is READY and I recommend GO with five conditions.** The day set, G, the by-name refusals and the consumption marker all drive as declared on scratch. Three things must change or be stated first: **the marker directory is resolved by the UNGUARDED `derived()`** while BE's own `require_ledger()` exists; **a half-written marker makes the consumption guard RAISE instead of refusing** — on the one-shot path; and **the GO must say in words what a refusal mid-read leaves behind: three days consumed and no result.** Also, a correction I owe: **REV 70 §0's "`admissibility.reasons` does not exist" was WRONG**, and the mechanism generalises to a hole in the seal census.

**Filed** 2026-09-06T16:50Z (clock read before composing) · reviewer seat (pm-codex) · tip `dd332cf`
· **LIGHT, LOCK-FREE, AND NOTHING CONSUMED.** Every drive used a SCRATCH declaration, SCRATCH feeds and days `20991230/31/32` — never a real day name, never the real pins, never `--open`. **Verified after my drives: `0` OPENED markers exist in the ledger's derived dir**; `de104smoke.service` untouched.

**ROUTING — CHECKED unless a line says AGREED.**

---

# 0. The correction I owe first (REV 70 §0)

R-676 is right and I can give the mechanism. `admissibility.reasons` **is** a key in the 09-03 receipt, value `[]`, in both arms. My census walked **leaves**, and an empty list yields no leaf — so a key whose value is `[]` or `{}` is invisible to it. I reported an absence that was a blind spot.

**And it is not only mine.** DA's `economic_absence()` walks leaves the same way. Driven:

| a receipt carrying | census verdict |
|---|---|
| `D_E0: 0.0` | `n_leaked_fields 1`, `sealed False` ✓ |
| **`D_E0: []`** | **`n_leaked_fields 0`, `sealed True`** |
| **`D_E0: {}`** | **`n_leaked_fields 0`, `sealed True`** |

The 09-03 receipt's own `seal_status` says every economic field is "ABSENT from this artifact, **not present-and-ignored**". A sealed key emitted as an empty container is *present and ignored*, and neither census can see it. **The seal census must walk KEYS, not leaves** — one line in DA's `_walk_paths` consumer, and DE's `_strip_economic` already removes keys so nothing produces this today. **DA 98 / DE.** (My error is the reason I found it; recording both halves.)

---

# 1. The day set is the declaration's — driven

Against a scratch declaration (`READABLE` = three fake days, `permutation_floor.G = 3`):

| invocation | result |
|---|---|
| no `--days` | `days=['20991230','20991231','20991232']`, `G_computed=3` |
| `--days` **widened** by one | **REFUSED**, naming the set and the extra day |
| `--days` **narrowed** by one | **REFUSED**, naming the set and the missing day |
| a scratch declaration whose **chain head declares two days** | `days=[…two…]`, `G=2` — the set follows the DECLARATION, not the code |

R-600 (a)/(d) and REV 48 §1.6's "G would come from the invocation" are closed: the CLI can neither widen nor narrow, and the head is resolved, never a filename. **AGREED.**

# 2. G, and the 0.25-while-reporting-5 route

`resolve_days` computes `G = len(READABLE)` and **refuses a declaration that disagrees with its own `permutation_floor.G`** (driven: READABLE of 3 against a declared G of 5 → refused, naming both). And `floors()` resolves the **conservative** value: `floors(3,3) = floors(5,3) = 0.25`, so an optimistic G cannot improve the resolved floor — the pessimistic one governs by construction.

**So the answer to "can a five-path call still reach 0.25-while-reporting-5": not through the CLI.** The remaining route is the `decl=` parameter — a caller can hand `read()` a declaration dict, as the battery does. The CLI's `--open` builds it with `resolve_days(...)` (line 825) and hands that, so the injection surface is the battery's only. Worth one sentence in the artifact: `decl=` is an injected observation, and a real read never supplies one. **AGREED with that note.**

# 3. The by-name refusals — reachable, in order, before anything generic

Driven, each on the third declared day:

```
no pin              -> "20991232 is in the declared READABLE set and has no pin…"
pin marks ABSENT    -> "20991232 is DECLARED READABLE but its pin marks the feed ABSENT (`exists: false`…)"
feed not on disk    -> "20991232 is DECLARED READABLE and pinned, but its feed … is not on disk. Named, not folded into a generic absence."
```

All three fire from the per-day loop **before** the declaration comparison and before any marker is written — so a refusing call **consumes nothing**, which is the property that makes the five-path call safe (R-600's hazard). The generic absence cannot fire first: there is no generic check ahead of the loop. **REV 48 §1.6 closed. AGREED.**

# 4. Literals

No `floors(5, 3)` literal remains; the selftest routes through the CLI path. I grepped the module for a hardcoded `0.25`/`G = 3` agreeing with the declaration by coincidence and found none on the read path. **AGREED**, with the standing caveat that a literal that agrees today is invisible to a test that only checks agreement — the census DA 93 built is the durable instrument.

# 5. Consumption — the marker

**Where it lives:** `<outdir>/be_race_read_OPENED_<day>.json`, one file per day, keys `{day, pin, utc, why}`. On the CLI path `outdir` defaults to `_BDR.derived()` — the ledger's `data/pm_5min/derived/`. **Verified now: zero such files exist; no race day is consumed.**

**Driven:** before any marker → `already_opened: []`; after `write_open_markers` → two files; a second call → **REFUSED**, naming the days and the marker paths. Order verified by reading `read()`: the markers are written at :389 and the first parse is at :396 — **written before a byte is read**, so a read that dies mid-way leaves the days marked. Consumed is the safe direction, as BE says.

**Three findings.**

**(a) The consumption guard resolves its directory with the UNGUARDED resolver.** `be_data_root` offers `require_ledger()`, which returns `ledger_check: PASS` only for the env branch or the canonical branch and refuses a root that is not the ledger. **`be_race_reader` calls `derived()` and never `require_ledger`** (grep: absent). Measured: with `PM_DATA_ROOT` **unset** from my worktree, `derived()` returns `/home/yuqing/ctaNew-wt-rev/data/pm_5min/derived` on branch `2_code_tree_carries_the_tape`. Today that path is the ledger **through the R-553 symlink**, so `require_ledger` would pass it and nothing is wrong. **From a MATERIALISED worktree — the state every seat was in at R-621, and what a bare checkout recreates — it is a partial shell**: the reader would look for markers there, find none, and open days the ledger may already record as opened. For a one-shot irreversible act the marker check must be as strong as the act. **Fix (BE 68, before GO or as a GO condition): the reader calls `require_ledger()` for the marker directory and refuses anything else.**

**(b) A half-written marker makes the guard RAISE, not refuse.** Driven: with `be_race_read_OPENED_<day>.json` containing `{ partial`, `assert_not_already_opened` raises `json.decoder.JSONDecodeError` out of the guard. A marker interrupted mid-write is exactly the case the write-before-read order exists for, and at that moment the operator gets a traceback rather than "this day is consumed". **This is the third "a crash is not a verdict" today** (DA's `UnpicklingError`, REV 73 §2.1; BE's `StopIteration`, REV 74 §3) and the only one on a one-shot path. **Fix: the FILE's presence is the fact; parse inside a `try` and treat an unparseable marker as OPENED.**

**(c) Nothing protects the marker.** It is an untracked file under `data/`; no git operation restores it and a deletion silently re-opens the day. That is inherent to a file-on-disk marker and acceptable **if the GO says so**: the markers are the only record that a day was spent, and they are as durable as the ledger directory.

---

# 6. What the READ GO must say — my conditions

**I recommend GO**, subject to these. They are all checkable before the act.

1. **The declaration, by digest at HEAD.** The GO names `be_race_read_declaration_v4.json` at **`a741b4d6…`** (its post-BE-65 digest; content identical to what REV 48/49 approved — verified at REV 74 §1.1) and the reader must resolve **that file as the chain head**. If the head is anything else, no read.
2. **The root, explicitly.** `PM_DATA_ROOT=/home/yuqing/ctaNew` on the invocation **and** §5(a)'s guard — or, if BE 68 does not land first, the GO carries the check itself: `readlink -f` of the marker directory equals the ledger's derived dir, asserted in the same command.
3. **The pre-state, recorded before the act.** Zero `be_race_read_OPENED_*` in the ledger (true now), the three pins present with `exists: true`, the three feeds on disk at their pinned digests. These are the facts a later reader needs to know were true at the moment of the read, and they cannot be reconstructed afterwards.
4. **The result's declared name, before the act.** The read artifact's filename is declared in the GO so DA's `da_race_read_verify.py --real` — which gates on that artifact's existence and refuses `READ_ARTIFACT_ABSENT_THE_READ_HAS_NOT_BEEN_OPENED` until it appears — can find it by the name it expects, and so a reader can tell "the read has not happened" from "the read happened and wrote elsewhere".
5. **What a refusal mid-read leaves behind — in the GO's own words.** The markers are written before the first byte is parsed. **A refusal after that point leaves all three days consumed and no result**, and the read cannot be retried: the outcome is then a named absence, not a smaller G and not a re-run. The GO must say this, because it is the one consequence that no instrument can undo and the one thing the operator must accept before pressing Enter.

**The read order after the act** (runbook §7, unchanged): DA's verifier on the read artifact **before any number is quoted**; then the runner's own read; then this seat's filing; then the coordinator reports the direction. **Nobody re-runs the reader to "check" it** — a second call refuses on the markers, which is correct, and the refusal is not a verification.

---

# 7. Not established

- **Nothing was consumed and no real path was touched**: every drive used a scratch declaration, scratch feeds and days `20991230/31/32`; the ledger shows 0 OPENED markers after my work.
- **I did not exercise a consuming read end to end**: my scratch feed is a ONE-ARM feed and `be_read_cells.load_two_arm_feed` refused it by name — *"this is a ONE-ARM feed and the declared estimand is an increment OVER the incumbent; computing it from one arm would compare the candidate with itself and return a zero that looks like a measurement"*. **A control I did not know about, firing correctly.** So §5's marker semantics are driven through `write_open_markers`/`assert_not_already_opened` directly, and the write-before-parse ORDER is established by reading `read()` (:389 vs :396), not by a full run.
- §5(a)'s materialised-worktree consequence is **reasoned from the measured branch**, not driven on a materialised tree (I have a symlink and will not break it).
- I did not verify the three real pins, the feeds, or DA's verifier's real path this round — items 3 and 4 of §6 are conditions for the GO, not claims that they hold.
- §2's `decl=` surface: I checked the CLI path only.
- The 09-04 receipt and DA 97's record are REV 75's.

**Routing:** BE 68 — §5(a) (`require_ledger` for the marker directory), §5(b) (an unparseable marker is OPENED, never a traceback), §2's note that `decl=` is an injected observation. DA 98 / DE — §0 (the seal census walks keys, not leaves). Coordinator — §6's five conditions in the GO, and §5(c) stated there.

**Context ≈ 34 %.** Held after this filing: nothing beyond what is routed above.
