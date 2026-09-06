# REVIEW — DE 80 + BE 52/53: the day-inputs seam and the exclusivity fix are **right and driven** — but **BE's assembly does not call it: its exact call raises `TypeError`, the day's FRAGMENT is never threaded, and the front door is bypassed**. And on the race read: R-588's estimand ruling is correct at the writer, **and the FEED exists for only three of the five days**

**Filed** 2026-09-06T07:27Z (clock read before composing) · reviewer seat (pm-codex)
· tip `4227d22` (DE 80 `7e26b9a`/`edd0335`/`60f4702`/`6f134a6`, BE 52 `3d7c769`,
BE 53 `8240cd4`, all ancestors) · **NO DATA RUN, NO SEALED FILE OPENED.** I read
receipts, declarations and code, and hashed two tapes and three receipts. I did not
open `be_forward_day_SEALED_scores_*` or `be_forward_day_SEALED_feed_*`.

**ROUTING — CHECKED.** Every claim below is my own observation.

**Rule 20, measured.** A heavy run is LIVE as I file (pid 2986172, `WRITE`, under
`flock -n … systemd-run --user --scope --slice=research.slice -p MemoryMax=8G -p
CPUQuota=100%` — the mandated wrapper). My heaviest steps ran beside it and are
light by measurement: the runner battery **18.45 s / 66 MB**, the race reader
**22.92 s / 22 MB**, the design battery 7.22 s / 209 MB, `phase2_arms` 0.03 s, and
one 991 MB `sha256sum` inside the admit drive. All under 60 s / 1 GiB; **I did not
take the lock.**

## VERDICT

**(A) DE 80's SEAM IS RIGHT AND EVERY FALSIFIER FIRES.** Driven: the consumed
constant on a ruled day REFUSES; a wrong digest REFUSES; no digest REFUSES; a path
outside the ledger REFUSES; a ruled day supplying nothing REFUSES rather than falling
back; **the right pair ADMITS**, verifying `9de88da9…` at load; and `day=None` still
returns the consumed hour unchanged. **§1.**

**(B) AND BE'S ASSEMBLY IS NOT WIRED TO IT — three ways, one of them fatal.**
`be_daybook_build.py:387` calls `R.build_tape_index(splits, path=…, day=…,
expect_sha256=…)`; DE's landed signature is `build_tape_index(splits, *,
tape_path=None)`. **Driven: `TypeError: build_tape_index() got an unexpected keyword
argument 'path'`.** The assembly cannot run. Beyond that: `fragment_slice` is called
with **no `source=`**, so the day's FRAGMENT is never threaded and the consumed-era
`harmful_exposure_rows_v3_eraB.json` (no September slug, R-573) would be used; and
`day_assembly_inputs` — DE's front door with all four verifications — **is not
called at all**. **§2. This is the answer to your question: no, BE is not calling
that call.**

**(C) MY REV 41 EXCLUSIVITY FINDING IS CLOSED, driven in the shape I filed it.** Two
concurrent `flock -s` holders in two processes: **both REFUSED** (were both
ADMITTED). One shared holder: refused. **An exclusive holder ADMITS** — the guard
was not made to refuse everything. **§3.1.** *Three other REV 41 items remain open
and two have grown:* the `/proc/locks` parse still ignores the DEVICE; the
missing-lock-file crash is **still a `KeyError`, now on a different key**; and the
prose *"all 23 day-path checks"* now sits beside `DAY_PATH_CHECKS = 49`. **§3.2.**

**(D) THE RACE ESTIMAND — R-588 IS RIGHT AT THE WRITER, AND THE INPUTS ARE NOT ALL
THERE.** `FEED_FIELDS` carries every field the estimand names, verified at
`be_forward_day.py:151-153` and corroborated in three producer receipts. **But on
surface `/home/yuqing` (`-xdev`, excluding `.git`/`node_modules`/`__pycache__`),
as-of 2026-09-06T07:21:15Z, search completed and not truncated: feeds exist for
09-03, 09-04 and 09-05 ONLY. 09-01 and 09-02 have no feed anywhere, and their
producing receipts record none — while every day that has one records a full
manifest.** v3 opens the feed; two of its five days have no feed to open. **§5.3.**

**(E) AND v3 PINS NO DIGESTS.** `sha256` appears **0 times** in
`be_race_read_declaration_v3.json`. **§5.4.**

---

# 1. DE 80 — THE DAY-INPUTS SEAM, DRIVEN

```
consumed constants:  tape phase2_state_tape_v5.json | fragment harmful_exposure_rows_v3_eraB.json

REFUSES  ruled day + the CONSUMED tape/fragment paths      "the tape is the CONSUMED-ERA constant …"
REFUSES  ruled day supplying NOTHING                       "may not fall back to the consumed-era constants"
REFUSES  ruled day, right path, WRONG sha256               "the tape at … "
REFUSES  ruled day, right path, NO sha256 declared         "a path without a digest is a claim about bytes"
REFUSES  ruled day, a path OUTSIDE the ledger              refused BEFORE the digest is considered
ADMITS   ruled day, the day's tape + fragment at their digests   regime=RULED_DAY_INPUTS_SUPPLIED
         tape sha verified AT LOAD: 9de88da950598e86…  (991,078,272 bytes)
ADMITS   day=None                                          regime=CONSUMED_HOUR_DEFAULT, both constants, sha256 None
```

**Five declared falsifiers, plus two of my own (no-digest, outside-ledger), plus the
consumed-hour positive control.** The last row is the one that keeps the fix honest:
the parameterisation did not loosen the consumed path, which DE said it checked by
running and which I reproduced.

**And the admit is stated the way DE stated it, which is correct:** it proves the
VERIFICATION path. The digest it verified is the one the **SCORE-split receipt**
names (`9de88da9…`, confirmed at v2 and v3 below); nothing here says BE's tape is
well-formed.

**One gap inside the seam.** `build_tape_index` does **not** thread `expect_sha256`
onward (`"expect_sha256" in inspect.getsource(...)` → **False**), so BE's own
`phase2_arms.tape_index(..., expect_sha256=)` — the parameter that verifies the
digest **at load**, as the stream is read — is never used. The digest is checked in
`day_assembly_inputs`, a separate act before the stream opens. That is the
check-and-use split I filed last round; BE built the fix and the threading does not
reach it. One keyword.

---

# 2. **FINDING — BE'S ASSEMBLY DOES NOT CALL DE'S SEAM, AND CANNOT RUN**

Q-DE-80 names the call. Here is what `be_daybook_build.build` actually contains,
against DE's landed signatures:

| DE's landed signature | BE's call | result |
|---|---|---|
| `build_tape_index(splits, *, tape_path=None)` | `R.build_tape_index(splits, path=_dt, day=day, expect_sha256=…)` | **TypeError** |
| `fragment_slice(dst, *, n_windows, source=None, only_slugs=…)` | `R.fragment_slice(frag, n_windows=len(ref), only_slugs=list(ref))` | **no `source=`** |
| `day_assembly_inputs(day, *, tape, fragment)` | — | **NOT CALLED** |

**Driven, with BE's exact keywords:**

```
R.build_tape_index({"score": None}, path=TAPE, day="20260903", expect_sha256="9de88da9…")
  -> TypeError: build_tape_index() got an unexpected keyword argument 'path'

R.build_tape_index({"score": None}, tape_path=TAPE)        <- DE's documented form
  -> admitted (proceeds to stream the tape)
```

**Three separate consequences, in order of severity:**

1. **The assembly cannot run at all.** `be_daybook_build.py:387` raises before any
   work. It is a one-keyword fix (`path=` → `tape_path=`, and drop `day=` /
   `expect_sha256=` or add them to DE's signature) — but it must be *chosen*, because
   BE and DE currently disagree about what the parameter is called and what it takes.
2. **The day's FRAGMENT is never threaded.** Even with the tape fixed,
   `fragment_slice` would slice the consumed-era `harmful_exposure_rows_v3_eraB.json`,
   which R-573 established spans 08-24..25 and holds **no September slug**. The book
   would be built from a tape for the right day and a fragment for the wrong era.
   **DE's seam refuses exactly this — and only when it is called.**
3. **`day_assembly_inputs` is bypassed**, so none of the four verifications in §1 runs
   on BE's path. BE has its own `day_tape_sha` + `assert_day_tape` guards (good ones,
   §4.1), but they check the receipt and the path, not the bytes at load, and they
   know nothing about the fragment.

**Why this was invisible to both seats.** BE 52 landed at 07:10:32Z having checked
origin at 07:09Z and correctly reported *"DE 80 has NOT landed"*; DE 80 landed at
07:12:10Z. Each built to its own picture of the other's signature and neither could
run the pair. **Rule 17 in its plainest form: two green suites, no integration.**
Neither seat's battery calls the other's function with the other's arguments.

**What closes it:** one seat owns the call site and drives it — the actual call, with
the actual arguments, against the landed signature, as a check in whichever battery
runs. Until that exists, "the blocker is cleared on my surface" is true of each seat
and false of the pair.

---

# 3. THE LOCK

## 3.1 **REV 41's exclusivity finding is CLOSED — driven in the shape I filed it**

All on a scratch probe lock; the real lock was untouched and was held by another
seat's heavy run throughout.

```
TWO CONCURRENT SHARED HOLDERS, TWO PROCESSES  (my REV 41 attack, verbatim)
  A_shared  pid 2992249  held=false  holders=[2992247]  assert_rule20: REFUSED
  B_shared  pid 2992256  held=false  holders=[2992255]  assert_rule20: REFUSED
ONE SHARED HOLDER
  solo      pid 2992259  held=false                     assert_rule20: REFUSED
AN EXCLUSIVE HOLDER  (rule 20's own `flock -n`)
  exclusive pid 2992331  held=TRUE   holders=[2992330]  assert_rule20: ADMITTED
NOBODY HOLDING
  holder_is_exclusive=null  someone_holds_it=false      -> refused (fail-closed)
```

**Both directions.** Previously both shared holders were ADMITTED for a 1 h /
6.84 GiB run; now both are refused and the exclusive holder still admits. The third
conjunct is visible in the observation and says why:
`probes: ["LOCK_EX|LOCK_NB", "LOCK_SH|LOCK_NB"]`, *"a shared hold blocks an exclusive
request and admits a shared one; only an exclusive hold blocks both."*

**DE's two driving lessons are both real and worth keeping in the register:** two
shared fds in ONE process is ONE `/proc/locks` holder, so the two-holder case needs a
second process — which is why my own reproduction used two — and `flock -s … sleep`
releases on terminate but not instantly.

*One stale string:* the `how` field still describes only the two OLD conjuncts and
does not mention the `LOCK_SH` probe, while `fresh_probe.probes` beside it does. The
same shape as the "23" below — a prose summary contradicting the computed field next
to it. One line.

## 3.2 Three REV 41 items still open, two of them larger

| item | state now |
|---|---|
| the `/proc/locks` parse ignores the **DEVICE** | **still inode-only** — driven with a crafted `07:99:<ino>` line, `by_self_or_ancestor` → `true` |
| a missing lock file raises **`KeyError`**, not `RunnerRefused` | **still open, and the key changed**: now `KeyError: 'self_or_ancestor_holds_EXCLUSIVE'` — this round's new field joined the same incomplete early return |
| the prose **"all 23 day-path checks"** in a real day's receipt | **still there, and `DAY_PATH_CHECKS` is now 49** — the gap went 23-vs-38 to 23-vs-49 |

None was in this round's scope, so these are ledger items rather than new findings —
but the second one is the same early return acquiring a second missing key, which is
the shape worth naming.

---

# 4. BE 52 — THE TAPE PARAMETER

## 4.1 The parameterisation is right, and its receipt binding is the good part

```
PA.tape_index(split, features_in_order=None, *, path=None, day=None, expect_sha256=None)
PA.assert_tape_for_day(day, path=None, *, expect_sha256=None)
```

The constant stays the consumed hour's default; a ruled day with the default refuses.
And `day_tape_sha` binds the assembly to **the digest the builder receipt published**
rather than to whatever is at the path — with the right predicate: it accepts a
receipt only when `WHICH_SPLIT_…["split"] == "score"`, so v1 (train-split) is
correctly skipped. Driven: `day_tape_sha("20260903","btc")` → `9de88da950598e86…`,
**which is the digest DE's seam verified at load in §1**. On that narrow question the
two halves do agree.

## 4.2 **FINDING — the receipt version search is a hardcoded list against an auto-versioning builder**

```python
for name in (f"be_gate1_state_tape_receipt_{day}_{coin}.v2.json",
             f"be_gate1_state_tape_receipt_{day}_{coin}.json"):
```

BE 52 added *"`--day` now versions to the next free `.vN.json`"* — and a **`.v3.json`
already exists** (07:20Z, superseding v2). The reader does not know about it. Today
both name the same tape sha so the answer is right; the next correction would bind
the assembly to a superseded receipt **silently**. Glob the versions and take the
highest, or read the supersession chain. One line.

## 4.3 **The battery still cannot be driven from a reviewer's worktree — and the gap has doubled**

```
be_daybook_build.py --selftest, from ~/ctaNew-wt-rev
  FAIL: ran 2 checks, expected 12 (EXPECTED_CHECKS=15 minus 3 skipped)   rc 1
```

Round 51: 9 declared, 4 unaccounted. Round 52 (my last filing): 15 declared, **10
unaccounted**. The cause is unchanged — `raise SystemExit(_finish(...))` after the
third skip — and it now strands BE 52's own new checks, including the
`assert_tape_for_day` drives and the `startswith("9de88da950598e86")` digest check at
`:647`. **BE's "15/15" is true at the ledger tree and false where the work is
reviewed, and every round adds more of BE's new work to the unreachable side.** The
fix is placement, not counting: move the fixture-drivable block before the real-data
gate.

---

# 5. THE RACE READ — VERIFIED AT THE WRITER, AND THE INPUT CENSUS

## 5.1 The writer, read directly (no sealed file opened)

```
be_forward_day.py:1914   out[coin].append((int(r["t0"]), FS.expected_cancel_value(fit, fp + ff)))
be_forward_day.py:1995   "per_coin_scores": {c: [list(x) for x in v] ...}
```

**Confirmed: the sealed SCORES are `(t0, expected_cancel_value)` from ONE fit** — no
incumbent, no action identity beyond `t0`, no realised cents. BE 52's refusal was
right about the scores, and v3's `why_the_scores_cannot` is exact.

## 5.2 `FEED_FIELDS` carries what the estimand needs — verified at the writer AND at three receipts

```
be_forward_day.py:151-153
FEED_FIELDS = ("slug", "side", "gen", "t0", "t_start", "score",
               "score_incumbent", "any_fill_ahead", "value_cents",
               "preventable_shares", "level")

be_forward_metric.reduce_window:  value_cents = lat[str(L)]["preventable_value_cents"]
                                  score_incumbent = the incumbent fit's value on the SAME vector
be_read_cells.load_two_arm_feed:  refuses a one-arm feed BY NAME
```

| the estimand names | the field | present |
|---|---|---|
| the ACTION unit (slug, side, gen) | `slug`, `side`, `gen` | ✓ |
| the INCUMBENT comparator | `score_incumbent`, computed in the same pass on the same vector | ✓ |
| realised cents | `value_cents` = the latency-resolved preventable cents | ✓ |
| L = 50 ms | resolved **at write time**; the receipts record `latency_ms_resolved: 50` | ✓ |

**So R-588 is right and I add one corroboration it did not have:** the three receipts
that record a feed also record its field list, and it is `FEED_FIELDS` verbatim. And
the interim's window-level fallback would not have rescued the scores either —
`be_forward_recon.py:84` defines the window increments as
`sum(increment_by_window) == net_cents − incumbent_net_cents`, so that statistic
needs the incumbent and realised cents too. **The window was only the reporting
bucket; the quantity was always the two-arm net.** No estimand in v2, declared or
inherited, was computable from the scores.

## 5.3 **FINDING — the feed exists for THREE of the five race days**

Surface `/home/yuqing`, `-xdev`, excluding `.git`/`node_modules`/`__pycache__`,
as-of **2026-09-06T07:21:15Z**; the search **completed (exit 0), output not
truncated**:

| race day | sealed SCORES | sealed **FEED** | the producing receipt |
|---|---|---|---|
| **09-01** | yes (5 copies) | **NONE ANYWHERE** | **no feed key at all** |
| **09-02** | yes (4 copies) | **NONE ANYWHERE** | **no feed key at all** |
| 09-03 | `~/ctaNew_forward_runs/20260903_be45/` | **yes**, 284 MB | feed manifest, **1,140,081 rows**, L=50, fields = FEED_FIELDS |
| 09-04 | `~/ctaNew_forward_runs/20260904_be45/` | **yes**, 297 MB | feed manifest, **1,194,791 rows** |
| 09-05 | `~/ctaNew_forward_runs/20260905_be44/` | **yes**, 206 MB | feed manifest, **827,727 rows** |

v3 states: *"round 52's refusal … did not establish whether the right file exists.
**It does.**"* **It does for three days of five.** And the asymmetry is not a
filesystem accident: **every day whose run wrote a feed records a full manifest —
path, sha256, bytes, n_rows, fields — and 09-01 and 09-02 record none.** A producer
that wrote one says so.

That sits against v3's `interim_evidence`, which says the interim statistic *"was
COMPUTED, on 09-01 and 09-02"* via `load_two_arm_feed`. I checked that function: it
takes a **path** and streams a JSONL feed, refusing a one-arm feed by name — so the
method is exactly as BE 53 says. **What I cannot reconcile is the input:** the two
days it names have no feed on this surface and never recorded one.

**I am not asserting BE 53 is wrong** — my own rule is to suspect the probe, and I
have not proved those feeds never existed, only that they are absent now and
unrecorded then. **Either reading has the same operational consequence, and it is
large:**

* if the 09-01/02 feeds are simply gone, the race read opens **three** feeds, not
  five — and the two missing ones are the two days already consumed;
* if the interim read those days from something else, then BE 53 has identified the
  METHOD correctly and not the INPUT, and the same question stands.

**This must be established before the read opens.** It is the same shape as my BE 50
finding: the read would proceed on inputs that cannot support the sentence the
declaration commits to — last time the statistic, this time the population. And
R-529(A)'s floors are stated at G = 5 and G = 3; which one governs is exactly what
this decides.

*Operational note:* the three feeds that exist live in `~/ctaNew_forward_runs/`,
**outside** `data_root` (`/home/yuqing/ctaNew/data`), which v3's own `data_root`
block asserts is the ledger — consistent with `FeedWriter`'s docstring, and it means
the read will open paths outside the ledger tree. v3 should name the run directories.

## 5.4 **FINDING — v3 pins no digests**

```
grep -c sha256   be_race_read_declaration_v3.json   ->   0
```

The byte-identity recompute carried forward from v2 proves the file did not change
**during** the read. It cannot prove the file is the one the declaration meant. For a
read that consumes five days irreversibly, each input should be pinned by **path +
sha256 at declaration time** — the discipline DE's params v3/v4 and BE's own
state-tape receipts already use, and the discipline `day_assembly_inputs` enforces on
the assembly one seam over. Three of the five digests can be taken today; the other
two are §5.3's question.

## 5.5 The reader itself

10/10 under my run (22.92 s / 22 MB), and the two properties I asked for are there:
**my collapsing-series known-bad now passes** — ties and a 1e-9 perturbation give the
**same** sign and quantities agreeing to 1e-6, where the old flip-count gave −1 and
+1 — and the digest is taken **on the bytes parsed**, with a mutation between the
two passes VOIDing the read. BE 54 retargets it to the feed; I drive that next round.

---

# 6. THE RECEIPT CHAIN — MY §1.3 IS CLOSED

| | checked | result |
|---|---|---|
| v1 restored to its landed bytes | `sha256sum` vs `git show 57ccc62:` | `bddb56437ef33733…` — **identical** |
| v2 supersedes v1 | its `supersedes` block | names v1's path, the split assignment, the `.WRONG_SPLIT` tape, and records *"I did it wrong first"*. **No predecessor digest.** |
| **v3 supersedes v1 and v2** | its `supersedes` block, and I hashed both | names **both** by path **and sha256** (`bddb5643…`, `ebb2d8ad…` — **both verified by me**), states what changed and that no number moves |
| the false field | v2 vs v3 | **v2 still carries** `what_this_build_did: "day fragment -> TRAIN"` and `status: PROVISIONAL`; **v3 corrects both** |
| all three tracked, tree clean | `git ls-files` / `status` | ✓ |

**v3 is the shape rule 13 asks for**, and it is better than what I asked for — it
names the predecessors by digest, which v2 did not. My §1.4 remains open: `train_split`
still carries **no sha256 and no row count** in v2 and v3, so `EMPTY_BY_CONSTRUCTION`
is still asserted rather than evidenced.

---

# 7. VERDICT

**`--day` remains APPROVED for the 09-03 smoke, and the parameterised seam is
APPROVED on DE's side** — every falsifier fires, the consumed path is unchanged, and
the exclusivity hole is closed with the good case still admitting.

**The pair is NOT ready**, and the blocker is now an integration one, not a design
one: **BE's assembly cannot call DE's seam (§2), and the fragment is not threaded at
all.** Both are small; neither is reviewed; and no battery on either side would catch
them, because no battery calls the other seat's function with the other seat's
arguments.

**My recommendation on the race-read estimand.** Adopt **MATCHED_VOLUME on the FEED**
— R-588's ruling is right, it is the statistic the two consumed days were read with,
it is the one with a rule cited (rule 7, matched on the decision variable), and every
field it names is in `FEED_FIELDS` by design. **A re-seal is not warranted for the
estimand.** But the ruling settles the STATISTIC and does not supply the POPULATION:

1. **Establish the 09-01/09-02 feeds before opening anything** (§5.3). If they do not
   exist, say so in v4 and state G honestly — a directional read on three days is a
   different sentence from one on five, and R-529(A)'s floors already distinguish
   them (0.0625 at G=5, 0.25 at G=3).
2. **Pin every input by path + sha256 in the declaration** (§5.4), before the read.
3. Then BE 54's retargeted reader, driven on a synthetic feed — mine next round.

**Order for the seats:** §2 (the call site, one owner, driven end to end) · §5.3 (the
feed census, before the read) · §5.4 (the digests) · §1's `expect_sha256` threading ·
§4.2 and §4.3 · §3.2's three open one-liners.

Nothing here ran the assembly, opened a sealed file, or read a race day.

---

## CONTEXT

Approximately 55%. Below the reset threshold; I will report the 80% crossing.
