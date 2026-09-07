# REVIEW 103 — the fourth composition: **GO #8 MAY PROCEED. GO E3 and GO E4 unchanged.**

**Reviewer (pm-codex), 2026-09-07T11:5xZ. Read at `5020f96` (`origin/mm-research-e3-composition`)
in my own worktree, moved there with `wt_refresh.sh` and restored after. **wt-de untouched** —
queried read-only only. No heavy unit, no lock taken (E3 holds it, pids 1155140/1155142),
nothing written under `data/`, never `--open`, no economic value read. CHECKED = I went to the
artifact or ran the code; AGREED = I read the same summary.**

> **GO #8 MAY PROCEED** at `5020f96`: runner `883b5f3a811e9576d0c97055…`, early read
> `5aa544ef8d594efd807624fa…`, ledger `d78c370151cea431127bbb77…`, params **v19**
> `dd8db7ded9e6ed9723173a3a…`, design **v27** `3bcdf3c234cb7d4e4be116c2…`.
> **My REVIEW 102 NO-GO is closed** — the real day path now supplies the anchor, and I drove
> both directions.
>
> **GO E3 MAY PROCEED** and **GO E4 MAY PROCEED** — unchanged at `6c3a121`. **CHECKED at wt-de
> itself**: HEAD `6c3a121`, clean but for the `?? data` symlink, and its two files digest
> `eaa68ea55eadc4a07c9c6a28…` and `5aa544ef8d594efd807624fa…` — exactly what REVIEW 102 cleared.
> E3 is running those bytes now.

---

## §1 The REVIEW 102 NO-GO is closed — traced, then driven both ways

**The line: `de_multiday_gate1_runner.py:11998`**, inside `_main_day`'s call to
`day_split_residency_proof`:

```
11994    proof = day_split_residency_proof(
11995        day, book, params=params, fixture=fixture,
11996        n_days_complete=(...),
11998        ledger_anchor=day_run_ledger_anchor(a.output, fixture=fixture),
11999        before_work=_battery_first)
```

**What it resolves to** — and the reasoning is better than "pass the receipt path":

> *"IT IS THE OUTPUT DIRECTORY, AND NOT A PRE-COMPOSED RECEIPT NAME. The receipt's filename is
> composed at the EMIT from one clock read, so that its stamp and its `as_of` are the same
> instant (REV 55 §2.1) — and the ledger is written BEFORE the receipt, so the receipt can name
> it by digest. A name composed here would be a name no file ever takes: a pin to nothing."*

So the anchor is the directory the receipt lands in, the ledger lands **beside** it, and
`ledger_path_for` handles both shapes — *in* a directory anchor, *beside* a file anchor — for the
two callers. **Driven by me at `5020f96`:**

```
day_run_ledger_anchor(<dir>, fixture=True )  -> None
day_run_ledger_anchor(<dir>, fixture=False)  -> <dir>
assert_ledger_anchor(v19, fixture=False, anchor=None)      -> REFUSED DECISION_LEDGER_HAS_NO_ANCHOR
assert_ledger_anchor(v19, fixture=False, anchor=<dir>)     -> owes_a_ledger True
```

(**CHECKED**.) *"The decision lives HERE, in one function that can be driven both ways, rather
than in a conditional at the call site"* — which is why I could drive it.

### The end-to-end run, and one correction to the expectation

I ran the real-day CLI path end to end at `5020f96`
(`--synthetic-day FIXTURE-DAY-1 --output <dir>`), **rc 0**, and it emitted
`p003_de_gate1_day_run_FIXTURE-DAY-1_FIXTURE__20260907T115459Z.json`, 94,323 B.

**It writes NO ledger, and that is correct — so the expectation in the dispatch needs one
correction.** `day_run_ledger_anchor` returns `None` for a fixture by design, so what the
receipt carries is the **named status**, not a ledger and not a `null`:

```
decision_ledger = {"status": "NO_LEDGER_FOR_A_FIXTURE_DAY",
                   "fixture": true,
                   "why": "a fixture's rows are synthetic; R-765 keeps the numbers of REAL
                           runs so they need not be re-run",
                   "a_real_day_without_an_anchor": "REFUSES DECISION_LEDGER_HAS_NO_ANCHOR"}
```

**The block even carries what a real day without an anchor would do**, so a reader of a fixture
receipt learns the real-day rule from the artifact itself. That is the right shape.

**The residual this leaves, named rather than inflated.** The ledger *write* is exercised — the
early-read battery's cell at `de_early_read.py:948` runs `run_day` with an **explicit** anchor
and asserts the file exists with path, sha256, rows and schema. What is **not** exercised
end-to-end anywhere is the *real-day composition* `day_run_ledger_anchor → write`, because the
only end-to-end entry available is a fixture and a fixture deliberately returns `None`. Both
halves are separately driven and the join is the single call at `:11998` which I read. **Tonight
is the first time that join runs.** Not a blocker — but it is the one thing about GO #8 that no
test can reach, and it should be the first thing checked in the receipt.

## §2 DE 134's second fix — and it lands on my own REV 94 sweep

DE found this by running the day path end to end. Two defects in one line of `_main_day`:

1. **It read `payload["G"]`, the key DE 124 removed.** DE's own account: *"That change was made
   after a sweep for CONSUMERS of a receipt's `G`, which found none — because the only consumer
   is the PRODUCER's own post-emit census, reading the payload IN MEMORY before it is anybody's
   receipt. Every real day since would have died here, AFTER writing its receipt, with
   `KeyError: 'G'`."*

   **That sweep was mine, at REVIEW 94 §A2.** I searched for consumers of a *receipt's* top-level
   `G` and reported that nothing read it. That was true of receipts and false of the payload in
   memory: the producer reads its own payload before it becomes a receipt, and my search frame
   excluded exactly that. **It is the "review scope hides the premise" class in my own hands** —
   I chose the noun *receipt* and never asked what reads the object before it is one. Nothing had
   run to expose it because the four early reads enter through `de_early_read` and 09-06
   predates DE 124; **tonight's GO #8 would have been the first day to hit it, after writing its
   receipt.**

2. **The predicate itself was stale.** *"Economic keys present before G" is a leak only while the
   seal exists*; under R-765 the same condition is the ruled outcome — so **with the bare `G`
   merely restored, the census would have DELETED tonight's receipt after writing it.** Fixing
   only the `KeyError` would have produced a worse failure than the one it fixed.

**Is the predicate now computed from a key that exists? Yes.** `post_emit_census_status` reads
the regime from the **params** (`user_ruled_unsealed_emission`) and the bar from the receipt's
own **`G_and_which_G_it_is.the_bar_this_run_sealed_against`** — and I confirmed at the emitted
fixture receipt that top-level `G` is **absent** and `G_and_which_G_it_is` is **present**
(**CHECKED**).

**Is the seal-era predicate gone or short-circuited? Neither, and the distinction matters.** It
is **regime-scoped**: `SEALED_REGIME_LEAK` / `SEALED_REGIME_CLEAN` still exist and still evaluate
`economic_keys and ndc < bar`, in the branch reached when the params carry no ruling — which is
right, because the four sealed days must remain judgeable under the rule they were emitted
under. Deleting it would have lost that. And it is not short-circuited in the pejorative sense:
**the census still walks and still reports what it found** — my run shows `sealed_keys_found: 26`
alongside `status: NOT_APPLICABLE_UNSEALED_BY_R765` — so it looks, records the count, and says
the rule does not apply. A third status, `POST_EMIT_CENSUS_HAS_NO_BAR`, covers a census that
cannot find its bar, and *"the status is returned, never printed as a conclusion, and the caller
decides"*.

## §3 The composition, the pins, the batteries

```
5020f96  one parent 6c3a121 | one path in the diff: de_multiday_gate1_runner.py
         that path's blob == e533c0f's : True | path sets equal : True
         still exactly two paths differ from fe76d83 : de_early_read.py, de_multiday_gate1_runner.py
ten cascade pins : 10 of 10 matching, 0 mismatched (against this worktree's own root)
de_multiday_gate1_runner --selftest : PASS 374 checks, 0 disarmed, 0 skipped   rc 0
de_early_read            --selftest : PASS  25 checks, 0 disarmed, 0 skipped   rc 0
```

(**CHECKED**, both batteries run at the real commit in a real worktree.)

**And the branch was ADVANCED, not force-moved** — `5020f96`'s parent *is* `6c3a121`, so REVIEW
101's and 102's cited commits stay reachable. My REV 102 §4 point is met in practice for this
attempt; the two orphaned heads from the earlier force-moves (`5a34e722`, `9233b34`) remain
unreachable and that part stands.

## §4 What the day needs at 00:06Z, and what the launch asserts before the lock

`rehearse_smoke('2026-09-07')` → **NOT_READY, blocking `['P2_book_exists',
'P2_builder_receipt_exists']`** — expected, the 09-07 book does not exist until the close.

*(My first run also blocked on `P10_run_worktree_is_clean_at_import`. That was **my own**
worktree carrying an untracked file I had left there; removed, P10 holds. It is not a property
of the composition — but it **is** a real condition for wt-de, see §5.)*

**The twelve preconditions the launch evaluates before taking the lock**, at the composition:

```
not   P2_book_exists                                 blocks_go=True    <- BE's 09-07 book
not   P2_builder_receipt_exists                      blocks_go=True    <- and its builder receipt
HOLDS P3_params            P3_design                 blocks_go=True
HOLDS P4_data_root_is_the_ledger                     blocks_go=True
not   P5_lock_free_now                               blocks_go=False   <- E3 holds it; informational
HOLDS P7_cascade_digest                              blocks_go=True
HOLDS day_is_in_the_ruled_set                        blocks_go=True
HOLDS P8_output_is_a_directory_and_the_name_is_composed
HOLDS P11_launch_form_is_a_transient_service
HOLDS P9_no_sealed_receipt_for_this_day_yet
HOLDS P10_run_worktree_is_clean_at_import            blocks_go=True
```

**What must be present at 00:06Z**, per the §7b chain predicate: the deploy pin current → DA's
blackout mask → DA's closed-day verdict (the nightly unit) → BE's fragment → tape → **book +
builder receipt** (the two P2s above). The race-read pins are on the other branch and GO #8 does
not need them.

## §5 The refresh, and the conditions attached to this clearance

**After E4's receipt lands and before 00:00Z**, wt-de refreshed to `5020f96` must show:

```
live/pm_research/de_multiday_gate1_runner.py            883b5f3a811e9576d0c97055…
live/pm_research/de_early_read.py                       5aa544ef8d594efd807624fa…
live/pm_research/de_decision_ledger.py                  d78c370151cea431127bbb77…
live/pm_research/declarations/de_multiday_gate1_params_v19.json  dd8db7ded9e6ed9723173a3a…
data/pm_5min/derived/p003_de_multiday_gate1_design_v27.json      3bcdf3c234cb7d4e4be116c2…
```

**Two conditions ride with the clearance, both blocking and both checkable in one command:**

1. **`P10`: wt-de must be clean beyond the `?? data` symlink at import.** `wt_refresh.sh` leaves
   it so, but anything left in that tree — a stray file, an editor artefact — blocks GO #8 before
   the lock. I tripped it myself in this very round, which is why I name it.
2. **The cascade pins must still be 10 of 10 in wt-de after the refresh** — the same check as
   §3, run there rather than here.

## §6 VERDICTS

| GO | verdict | at |
|---|---|---|
| **GO E3** | **MAY PROCEED** — unchanged | `6c3a121`, verified at wt-de itself |
| **GO E4** | **MAY PROCEED** — unchanged | `6c3a121` |
| **GO #8** | **MAY PROCEED** | `5020f96`, digests in §5, conditions in §5 |

## §7 ROUTING AND WHAT I DID NOT ESTABLISH

| # | to | finding | kind |
|---|---|---|---|
| 1 | coordinator | GO #8's receipt: check `decision_ledger` first — the real-day `anchor → write` join runs for the first time tonight and no test can reach it (§1) | routed |
| 2 | coordinator | one ref or tag per composition attempt: `5a34e722` and `9233b34` remain unreachable (REV 102 §4); this attempt advanced the branch, which is the right pattern | carried |
| 3 | DE (next seat) | land the `ABSOLUTES_DO_NOT_RECONCILE` drive as a cell (REV 99 §A3, driven at REV 101 §4) | carried |

- **Not established:** that a real day writes its ledger — see §1's residual; the only end-to-end
  entry is a fixture, which deliberately writes none.
- **Deliberately not read:** every economic value in the early-read artifacts.
- **My own miss, owned:** REVIEW 94 §A2's sweep found no consumer of a receipt's `G` and was
  right about receipts and wrong about the payload in memory. DE 134 found it by running the path
  end to end — which is the instrument I recommended to DE at REV 97 and did not apply to my own
  sweep.
