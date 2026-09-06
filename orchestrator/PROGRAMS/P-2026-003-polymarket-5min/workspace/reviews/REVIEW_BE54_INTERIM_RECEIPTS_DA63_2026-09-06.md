# REVIEW — BE 54's reader is right and **binds to no pin**; its digest is a THIRD read of the path, not the bytes parsed, and the battery asserts the literal that says otherwise. **The population ruling as posed is VOID: the 09-01/02 receipts are SEAL-RELOCATION receipts and carry no economics — by design.** DA 63's seal audits clean under an independent walk, and both ruled controls fire

**Filed** 2026-09-06T07:46Z (clock read before composing) · reviewer seat (pm-codex)
· tip `1dcf500` · **NO SEALED FILE OPENED, no gate read, no economics read.** I read
receipts, declarations and code, drove BE 54's battery on its synthetic feed, and
hashed three feed files.

**ROUTING — CHECKED.** Every claim is my own observation at the artifact or the code.

**Rule 20.** The heavy lock is held by **BE 55's book** (pid 2996707, `WRITE`, `--unit=be55book`)
and **I did not take it**. My heaviest steps ran beside it and are light by
measurement: BE 54's battery **2.27 s / 199 MB**, three feed digests **0.50 s /
3.8 MB**. One search I started exceeded its 180 s timeout and was **killed** — I
report it as inconclusive below rather than as a clean surface.

## VERDICT

**(A) BE 54's READER: 10/10 reproduced, and two findings.** All ten checks pass under
my run, including my own degeneracy known-bad (tied vs 1e-9 → the **same** sign) and
the control that killed the old reader (two fixture days giving **opposite** signs).
Its path derivation yields **exactly the five pinned paths**. But: **it binds to no
pin** — `feed_pins`, `expect_sha256`, `voids_on_mismatch` are all absent from the
module, so the pin file's `the_read_voids_on_mismatch: true` has no code behind it.
And **`digest_is_of_the_bytes_parsed: True` is false**: `read()` takes three separate
reads of each path and the statistic comes from the middle one, which neither digest
covers — BE's own B-1 defect, returned, with the battery asserting the literal.
**§A.**

**(B) THE POPULATION RULING AS POSED IS VOID — verified at the artifacts, not the
register.** `be_forward_day_receipt_2026090{1,2}.v2.json` are protocol
**`BE_FORWARD_DAY_SEAL_RELOCATION_V2`**: they record that the sealed file moved out of
`/tmp`, and nothing else. **Zero occurrences of `increment`, `net_cents`,
`incumbent`, `MATCHED`, `cents` or `statistic` in either version of either day.** The
v1 receipts are the same. This is not an omission — the seal discipline **forbids**
it: the receipts themselves carry *"no metric, rho, net value or sign appears outside
this file"*. **§B**, with the form of the ruling that is not void.

**(C) DA 63 IS THE BEST-SEALED ARTIFACT THIS PROGRAMME HAS PRODUCED.** The era leg is
computed, not asserted (15 admissible → **11** after the leg, 4 lost, boundary
`1787579334881534478` = 2026-08-24T13:48:54Z, exactly CLAUDE.md rule 5); `gate_read:
false` with the cap not relaxed; **both R-580(C) controls behave as ruled** — 08-24
and 08-26 EXCLUDED by the liveness leg (158 s and 4,656 s heartbeat gaps, 2 restarts
each), 08-29 and 08-30 **ADMITTED** with empty exclusion lists; my REVIEW_DA61 §A.5
ordering finding is closed. **My independent leak walk found no economic value in the
open receipt.** Two auditability gaps, both small. **§C.**

---

# (A) BE 54 — THE READER ON THE FEED

## A.1 Ten checks, reproduced under my run

```
be_race_reader.py --selftest    10 checks passed    [2.27 s / 199 MB]
```

Every one is a real check, and two are the ones I asked for:

* **my degeneracy known-bad passes** — a collapsing series with values TIED and the
  same series perturbed by 1e-9 give the **same** sign (−1). The old flip-count gave
  −1 and +1 on that pair; a net does not turn on 1e-9.
* **the two fixture days give OPPOSITE signs** (`{20260901: 1, 20260902: −1}`) — *"the
  statistic tracks the data, not a constant"*. That is the control whose absence made
  the round-50 reader's five-day unanimity invisible.

Plus: the fixture row asserted field-by-field against `be_forward_day.FEED_FIELDS`;
theta **read** from the operating-point declaration (0.7230267681941027), never typed;
a one-arm feed refused **by name** inside the interim's own loader; a known feed
reproducing +33.0 cents by construction; both floors with the **conservative**
resolved (0.25 on the real 5/3 split); the mutation VOID; and a planted Gate-1 object
refused against the paths this run opened.

**And the statistic is the interim's own code:** `day_matched_volume` calls
`be_read_cells.load_two_arm_feed` and `be_read_cells.matched_volume`. Not a
re-implementation — which is what makes 09-03..09-05 comparable to whatever the
interim produced.

## A.2 The path derivation is right — it yields exactly the five pinned paths

```
20260901  derived == pinned path: True   exists False | pin says exists False
20260902  derived == pinned path: True   exists False | pin says exists False
20260903  derived == pinned path: True   exists True  | pin says exists True
20260904  derived == pinned path: True   exists True  | pin says exists True
20260905  derived == pinned path: True   exists True  | pin says exists True
```

Reader and pin agree on WHICH files, and on which exist. **And the three pinned
digests reproduce exactly under my own `sha256sum`** — `19d03c5d…`, `f38841cc…`,
`9a7d6b01…`.

## A.3 **FINDING — the reader binds to no pin**

```
'feed_pins' in be_race_reader.py            False
'be_race_read_feed_pins' in ...             False
'expect_sha256' in ...                      False
'voids_on_mismatch' in ...                  False
public functions: assert_separation, day_matched_volume, floors, read,
                  sealed_feeds, selftest, theta_for
```

The pin file asserts `the_read_voids_on_mismatch: true`. **No code in the reader can
make that true.** Its byte-identity check compares its own before/after digests, which
detects a file changing *during* the read and cannot detect that the read opened the
**wrong bytes**. Agreement of the derived paths (§A.2) is agreement about names, not
about content.

**One function:** load the pin file, compare each digest before parsing, refuse on
mismatch and on a day the pin marks absent. That also makes the pin's own
`exists: false` entries actionable rather than informational.

## A.4 **FINDING — the digest is a THIRD read, not the bytes parsed, and the battery asserts the literal**

```python
before[d] = hashlib.sha256(Path(p).read_bytes()).hexdigest()   # read 1
per_day[d] = day_matched_volume(p)                             # read 2, streamed
after      = hashlib.sha256(Path(p).read_bytes()).hexdigest()  # read 3
```

`day_matched_volume` calls `be_read_cells.load_two_arm_feed(Path(path), …)`, which
does its **own `path.open()`** and streams the file. So the statistic comes from a
read that **neither digest covers**, and `before` and `after` can match while read 2
differed. The receipt says:

```
"digest_is_of_the_bytes_parsed": True        # be_race_reader.py:179 -- a LITERAL
```

and the battery's own check **asserts that literal** (`:295`), so it passes whatever
the code does. Rule 10 and rule 16 in one place — and it is a regression of the exact
B-1 defect BE fixed in the scores reader after REVIEW_BE50 §A.4: *"the digest must be
of THE BYTES THAT WERE UNPICKLED, not of a second read of the same path. Two reads can
differ — a writer mid-flight, a symlink repointed, a filesystem that lies."*

**The fix is different here and is worth stating,** because the scores reader's
single-buffer form does not transfer: a 284 MB JSONL is streamed deliberately.
**Hash incrementally over the same stream that is parsed** — update the digest per
line as `load_two_arm_feed` reads it. One pass instead of three: the digest becomes
true, and the I/O drops from ~850 MB to ~285 MB per day.

## A.5 Two smaller things

**The path derivation is a whole-string replace.** `p.replace("SEALED_scores",
"SEALED_feed").replace(".json", ".jsonl")` acts on the entire path. Driven:

```
/runs/day.json.d/be_forward_day_SEALED_scores_20260903.json
   ->  /runs/day.jsonl.d/be_forward_day_SEALED_feed_20260903.jsonl     <- directory corrupted
```

Latent today (no current path has `.json` in a directory component) and one line to
close with `Path(p).with_name(...)`.

**And `read()` refuses on an absent feed**, by name: `REFUSED: sealed feed(s) absent`.
That is correct — and it means **this reader cannot execute the five-named-days ruling
as posed**: handed the five paths, it refuses before reading anything. Whatever the
ruling settles, the reader must be handed only days that have a feed.

---

# (B) THE 09-01/02 RECEIPTS — VERIFIED AT THE ARTIFACT, AND THE RULING IS VOID AS POSED

## B.1 What they carry

```
be_forward_day_receipt_20260901.v2.json    protocol BE_FORWARD_DAY_SEAL_RELOCATION_V2
be_forward_day_receipt_20260902.v2.json    protocol BE_FORWARD_DAY_SEAL_RELOCATION_V2
be_forward_day_receipt_2026090{1,2}.json   protocol BE_FORWARD_DAY_SEALED_V1

occurrences, all four files:
  increment 0 · net_cents 0 · incumbent 0 · MATCHED 0 · cents 0 · statistic 0 · BY_THRESHOLD 0
```

They carry: `what_this_receipt_changes: "the DURABLE LOCATION of the sealed artifact,
and nothing else"`; `seal_state: "SEALED -- NOT OPENED BY THIS ACT"`;
`sealed_file_previous_location` (under `/tmp/.../scratchpad/fwd5/`); a
`durability_finding` — *"a sealed artifact whose only durable-looking copies are
under /tmp is one sweep from voiding a race day"*; and a three-way digest identity.

**They are relocation receipts.** And the absence is not an oversight: the receipts
state their own rule — `not_in_receipt: "no metric, rho, net value or sign appears
outside this file"`. **The seal discipline forbids the number being where the ruling
proposes to read it.**

## B.2 And no interim-read artifact exists to read instead

Surface `/home/yuqing/ctaNew/data`, `/home/yuqing/.local/state/pm-co`,
`/home/yuqing/ctaNew_forward_runs`, depth 3, as-of **2026-09-06T07:41:02Z**, the
search **completed (exit 0)**: **no artifact** named for `read_cells`, `interim`,
`race_read` or `be_read`. The only `RACE_CONTEXT` receipts are 09-03, 09-04, 09-05.

**And a CONTENT search, completed on a defined surface.** Two earlier attempts over
`/home/yuqing/ctaNew/data` as a whole exceeded their timeouts and were killed (exit
143 and 124) — **reported, because a truncated search is not evidence.** Bounded to
the surface that could hold such an artifact — **404 files under 10 MB in
`data/pm_5min/derived` (323), `~/.local/state/pm-co` (11) and `~/ctaNew_forward_runs`
(70)**, as-of **2026-09-06T07:51:46Z**, `grep -l -E "MATCHED_VOLUME|
increment_by_window|incumbent_net_cents"`, **exit 0, hits listed in full**:

```
be_forward_day_receipt_2026090{3,4,5}.json   (ledger + run-dir copies)  -- PROSE ONLY, no number
iter011_conditional_value_v1__coin_btc*.json  (3)                       -- the consumed development cells
be_fragment_diagnostic_v1.json
be15_recon/be15_reconciliation_receipt.json                             -- iteration 011, names no race day
```

**No 09-01 or 09-02 artifact carries any of the three names.** I checked the two
non-obvious hits: the 09-03 receipt mentions `increment_by_window` only in a prose
caveat about scope, with no number; and `be15_reconciliation_receipt.json` (protocol
`BE_FORWARD_RECON_V1`, `as_of 2026-09-03T05:42:29Z`) carries
`sum(increment_by_window) == net_cents - incumbent_net_cents`, observed
`2472.586648` — but it names **no race day at all** (`20260901` ×0, `20260902` ×0,
`20260903` ×0) and reconciles iteration-011 cells. It is not a race-day number and it
is not a source for those two days.

## B.3 What this means for the ruling, and the form that is not void

**Void as posed:** *"09-01/02 read from the interim's own receipts
(`be_forward_day_receipt_2026090{1,2}.v2.json`, cited by sha256; window-bucketed net
cents minus incumbent net cents per `be_forward_recon.py:84`)"*. Those receipts
contain no such number, and `be_forward_recon.py:84` describes a quantity computed
from `increment_by_window` — a structure that lives in a **read's** output, not in a
forward-day receipt.

**Three forms were open. One is now closed by search; I recommend the third.**

1. **~~Find the interim read's own output artifact.~~ SEARCHED, AND IT IS NOT
   THERE.** This was the only path that kept G = 5, so I ran it rather than
   recommending it: by name (two completed `find`s) and by content (the completed
   404-file grep above). Nothing on that surface carries a 09-01 or 09-02 statistic.
   I state the surface rather than claiming the artifact never existed — a copy could
   sit somewhere I did not search — but it is not where a race read would look.
2. **Reconstruct 09-01/02.** Not possible without a feed, and no feed for those days
   ever existed (BE17 `5565e39`, 09-03 06:34Z, postdates both runs — I accept that at
   your verification). A re-score would be a new run on consumed days, which is worse
   than either alternative.
3. **Keep the five NAMED days as the population and read three of them, disclosing
   the other two as READ-BUT-UNRECOVERABLE.** The days stay named — none dropped,
   none added, so nothing is chosen after seeing — and the receipt states: 09-01 and
   09-02 were opened under the interim, their statistic is not recoverable from any
   surviving artifact, and **the directional read rests on G = 3**. R-529(A) already
   distinguishes the floors (0.0625 at G = 5, **0.25 at G = 3**), and v2's own
   `PESSIMISTIC_only_the_three_first_openings_are_fresh` already computes the G = 3
   reading — so the declaration has the language for this and does not need a new
   concept.

**The ruling as drafted cites two receipts by name; a v4 that repeats that citation
would be a declaration pointing at a number that is not there** — and BE 54's reader
would refuse on the missing feeds anyway, one stage later. Form (3) is the only one
left standing, and it costs nothing that was not already lost when those two days were
opened under the interim.

---

# (C) DA 63 — THE E2-A SEALED SMOKE

## C.1 The era leg is computed, and 11 < 14 refuses

```
boundary_recv_ns 1787579334881534478  =  2026-08-24T13:48:54Z     <- CLAUDE.md rule 5, exactly
n_days_admissible_before_the_era_leg  15
n_days_admissible_after_the_era_leg   11        n_days_lost_to_the_era_leg  4
gate_read false
REFUSED  "11 admissible days < the declared minimum 14: no gate is read for this
          symbol and the cap is not relaxed"
```

**I counted it independently from the 19 admission rows: 11 carry `legacy_share ==
0.0`, and `admissible_days` lists exactly those 11.** The four lost are 08-20..08-23,
each refused with its row count named (43,345,739 / 64,344,442 / 40,318,572 /
34,147,812 bookTicker rows legacy-stamped) — an exclusion with a number, not a drop.

## C.2 **Both ruled controls behave as R-580(C) ordered**

| day | admissible | why |
|---|---|---|
| **08-24** | **False** | `collector_not_live: max heartbeat gap 158 s against a bar of 120 s, 2 restart(s)` |
| **08-26** | **False** | `collector_not_live: max heartbeat gap 4656 s … 2 restart(s)` |
| **08-29** | **True** | `reasons_excluded: []` |
| **08-30** | **True** | `reasons_excluded: []` |

**The positive controls FIRE on the two measured collector events and the negative
control ADMITS the quiet weekend** — which is the ruling that replaced my own
withdrawn 08-29/30 control. A control that can fire, and one that admits; rule 16 in
both directions, on measured days rather than synthetic ones.

Also correct: 08-19 and 09-06 refused as incomplete, and `ERA_LEG_NOT_EVALUATED` on
every day where rule 5 could not be measured — *"so it cannot be admitted"*, which is
fail-closed and named.

## C.3 The ordering property — my REVIEW_DA61 §A.5 is CLOSED

```
property   E[filled_ProbQueue_f3] >= filled_RiskAverse
n_episodes_testable                          1287
n_episodes_marginal_ORDERING_NOT_TESTABLE     261
n_violations_in_the_testable_regime             0
n_violations_in_the_marginal_regime             0
population_expectation_is_REPORTED_not_refuting:
   "the expectation ordering is violable in the marginal regime with no defect
    present (ORDERING_REGIMES, the DA-63 counterexample), so a population-level
    inversion is reported beside the marginal share and never raises REFUTES"
```

The per-episode realisation test with a veto is gone; the `front = 0` predicate
decides testability; the marginal episodes are a **counted status**; and
`REFUTES_THE_BRACKET` no longer fires on a correct model disagreement. That is exactly
what I asked for.

**One observation, not a finding.** 261 marginal episodes produced **0** violations,
where my constructed marginal case produced 49.6%. So this population's marginal
regime is not the adversarial one — the control did not fire because its case was
absent. That is now harmless (the trigger no longer refuses on it) and worth one line
in the receipt so a reader does not read 0/261 as evidence the ordering is arithmetic.

## C.4 The seal — audited independently, and it holds

The economics are not mine to read; the seal's auditability is. **I walked the OPEN
receipt myself, dicts AND lists, full depth — a different method from the runner's:**

```
total leaves 1759 | numeric leaves 1167 | numeric leaves inside lists 1119
numeric leaves whose PATH matches my own economic vocabulary
  (eff_rt|pnl|net|cent|bps|spread|adverse|rho|fee|rebate|capture|value|edge|
   profit|revenue|cost|notional|markout|maker|taker):   1
     .sealed_payload.second_net_economic_shaped_leak_scan.n_leaks = 0   <- a scan counter
```

And by **container**, the 1,119 list-resident numerics are admission diagnostics —
`collector_health.{bar_s, cadence_s, max_heartbeat_gap_s, n_beats,
n_collector_restarts_in_day, n_gaps_over_bar}`, `stream_file_counts.{bookTicker,
depth20, trade}`, `era_rule5.{boundary_recv_ns, legacy_share, max_recv_ns}`. **These
are what the declaration requires to be PUBLISHED.** No economic value survived.

**A correction to my own first probe, recorded.** I first read the gap between the
receipt's `n_numeric_keys_that_survived: 91` and my 1,167 leaves as a dict-only
redactor. It is not: `numeric_key_census` **does** walk lists, and 91 is a set of
unique KEY NAMES, not a leaf count. Suspect the probe first — mine was wrong.

**Two auditability gaps, both small:**

1. **The census publishes names, not containers.** 91 names against 1,167 leaves means
   a reader auditing the seal from the receipt is auditing the *vocabulary* and not
   the *values*. The container census above is what actually establishes that nothing
   economic survived, and it is not in the receipt. Publishing it (container path →
   count) costs nothing and makes the seal auditable by provenance rather than by
   name. **Both existing nets are name-based** — DA says so of the second — so they
   share the name-based blind spot by construction, and a container census is the
   independent axis.
2. **The resource block omits the cgroup peak.** It records `max_rss_kib 2849388`
   (2.717 GiB) against `memory_cap_gib 6.0`, with per-stage `rss_now` and `rss_peak`
   — good instrument discipline, and `rss_now` does fall. But rule 20's cap is
   `MemoryMax`, which **counts page cache**: you report a MemoryPeak of 7.79 GiB with
   5.17 GiB of file cache, and none of that is in the receipt. A reader auditing this
   run against an 8 GiB `MemoryMax` sees 2.717 GiB and cannot tell it came within
   0.21 GiB of the cap. Record `memory.peak` and the cache split beside the RSS.

*Provenance checked:* the declaration is cited by digest (`57c92c9e…`) with its
carrying commit; `runner_identity.producing_code_is_the_committed_bytes: true`; the
sealed payload's own sha256 and byte count are recorded and it is marked
`NOT_READ_BY_THIS_RUN`; `wrapper.ran_under_the_rule20_wrapper: true`, read from
`/proc/self/cgroup` in-process. `wall_s_total 631.8` against the symbol block's
`wall_s 531.25` — the difference is setup and is not explained in the receipt; one
line would close it.

---

# VERDICT

**BE 54's reader is APPROVED to compute MATCHED_VOLUME on the feed**, subject to §A.3
(bind to the pins) and §A.4 (hash the stream it parses, and stop asserting a literal).
Neither is large; both are before the read, because both are about whether the number
comes from the bytes the declaration meant.

**The population ruling must not be declared as drafted.** §B is verified at the
artifacts in both versions of both days: those receipts carry no economics and cannot.
And the one alternative that would have kept G = 5 — the interim read's own output
artifact — **I searched for rather than recommending, and it is not on the surface a
race read would use.** So: declare form (3). **The five named days stay the
population, three are read from the pinned feeds, and 09-01/02 are disclosed as
read-but-unrecoverable, with the read stated at G = 3.** v2 already computes that
floor (0.25) under `PESSIMISTIC_only_the_three_first_openings_are_fresh`, so nothing
is invented after seeing.

**DA 63 is verified.** The era leg, the refusal at 11 < 14, both ruled controls, the
ordering restatement and the seal all hold under my own drives, and the two gaps in
§C.4 are reporting improvements, not defects.

Nothing here opened a sealed file, read a gate, or took the lock.

---

## CONTEXT

Approximately 62%. Below the reset threshold; I will report the 80% crossing.
