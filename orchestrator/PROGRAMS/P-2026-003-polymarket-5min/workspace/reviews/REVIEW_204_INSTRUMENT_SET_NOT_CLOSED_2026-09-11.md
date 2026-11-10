# REVIEW 204 — (1) the as-of assertion drives 5/5 including two arms DE's own falsifier lacks. (2) **I CANNOT SAY THE SET IS CLOSED.** Zero of eight falsifiers run the production path, two have none at all, and the opt-in exists in one of three copies.

**REV 164, 2026-09-11T11:30Z** (clock read separately). Read-only except scratch-root drives.

---

## (1) DE 293 VERIFIES — AND MY DRIVE ADDS TWO ARMS

`GROWING` now carries **`collector_health.jsonl`** (copied like the rest, so the §2 symlink
refusal covers it). `raw/` is handled by **as-of listing + exit re-verification** rather than
copying — 4.4 GB / 2,016 files for one day, and the reason is stated in the launcher.
`de_asof_listing.py`: `listing` at launch (`exit 9` if not taken), `verify` at the valuation's
exit (`exit 8`), refusal `RUN_INPUT_MOVED_DURING_RUN:<file>`.

**DE's falsifier: 4/4. Mine, on my own scratch root, 5/5 — including two arms DE does not
have:**

```
[PASS] unchanged slice ADMITS
[PASS] growth OUTSIDE the day's slice ADMITS          <- the collector may keep writing later windows
[PASS] a changed file INSIDE the slice REFUSES by name
          -> REFUSED RUN_INPUT_MOVED_DURING_RUN:btc-updown-5m-1788826200.jsonl.gz -- 1 file(s) ...
[PASS] a REMOVED slice file REFUSES        <- MINE
[PASS] an ADDED slice file REFUSES         <- MINE
```

**The design is right**: growth outside a closed day's slice is allowed and counted; any
change inside it refuses by name. **Item (1) is sound.**

---

## (2) **THE SET IS NOT CLOSED. I WILL NOT SAY IT IS.**

**The measurement, across all eight:**

```
instrument                        falsifier?   invokes the PRODUCTION path?
de_preflight_matrix.py               yes            NO  (calls matrix()/book_acceptance() directly)
be_build_preflight.py                yes            NO  (calls check_tree()/check_day() directly)
da_population_freeze_verify.py       yes            NO  (calls verify() directly)
be_book_window_census.py v2          yes            NO
de_asof_listing.py                   yes            NO  (calls listing()/verify(), never main(argv))
da_unit_script_drift.py              yes            NO  (calls scan() directly)
launchers/chain_day.sh               ** NONE **     --
launchers/preflight_gate.sh          ** NONE **     --
be_heavy_run.sh (the opt-in)         no cell for it --
```

**`subprocess`/`systemd-run`/entry-point invocation appears in ZERO of the eight falsifiers.**
Every one proves the unit. **Rule 17 fired three times today; this is the same shape, set-wide.**

### THE THREE THAT BLOCK THE FREEZE, IN ORDER

**(a) `chain_day.sh` and `preflight_gate.sh` HAVE NO FALSIFIER — AND THEY *ARE* THE PRODUCTION
PATH.** The snapshot-root refusal I drove at REVIEW 203 I had to **extract by hand** into a
scratch script; there is no `--falsify`, so nothing runs it, and nothing would notice if it
stopped firing. **This is rule 15's class, not rule 17's, and it is worse.**

**(b) `be_heavy_run.sh`'s `BE_SNAPSHOT_ROOT` opt-in — THE ONE PROPERTY THAT MATTERS IS
UNPROVEN, AND IT IS THE DIRECT SUCCESSOR OF THIS MORNING'S DEFECT IN THE SAME FILE.**

```
wt-deval : 526  --setenv=PM_DATA_ROOT="${BE_SNAPSHOT_ROOT:-$REPO}"     <- the opt-in
~/ctaNew : 524  --setenv=PM_DATA_ROOT="$REPO"                          <- NO OPT-IN
wt-be    : 524  --setenv=PM_DATA_ROOT="$REPO"                          <- NO OPT-IN
```

**The fix exists in ONE of three copies of the same launcher.** BE's runs launch from `wt-be`
and the shared tree's copy is unfixed — so a book built through either still resolves the
**live** root while a valuation launched from `wt-deval` resolves the **snapshot**. *If a
build and a valuation in the same chain disagree about the data root, that is the REVIEW 203
defect with the sign flipped.*

**And the property — "a unit launched through this wrapper sees `PM_DATA_ROOT=<snapshot>`" —
can only be proven by LAUNCHING A UNIT AND READING ITS `Environment` BACK.** I can read line
526 and confirm the expansion is correct bash; **I cannot prove the unit receives it, and
neither can any cell in the set.** That is exactly how the original went undetected for
hours.

**(c) `de_preflight_matrix.py` — unit-proven with a MEASURED wiring defect still open.**
REVIEW 197 drove it: three gates refuse from the module's own directory and pass from the
tree root, because the **callee** resolves a relative `DECL` from cwd. Its production path is
`python de_preflight_matrix.py <cert> <params>` from `preflight_gate.sh`. **The falsifier
returns 5/5 from the right cwd and 3/5 from the wrong one, and no cell says the answer
depends on where you stand.**

### WHAT CLOSES IT — ONE CELL PER INSTRUMENT, AND IT IS THE SAME CELL

> **Invoke the entry point the way production invokes it and assert the same verdict:**
> `subprocess.run([sys.executable, "<module>.py", *argv], cwd=<a DIFFERENT directory>, env=<the unit's env>)`
> — for the shell instruments, run the script itself against a scratch root.
> **One cell each. It would have caught (c) outright, and it is the only thing that can
> establish (b) short of launching a unit — which (b) should also do, once.**

**Two of the eight additionally need a falsifier to exist at all (a).** Until (a), (b) and (c)
are closed I would not freeze the instrument class: **a freeze that locks in two instruments
with no falsifier and one with a measured, open wiring defect makes those permanent by
declaration.**

### WHAT I *WILL* SAY IS CLOSED

**`de_asof_listing.py` is sound on its own terms** (§1, 5/5 with my two extra arms), and
**`da_unit_script_drift.py` is the only instrument in the set with a production consumer
already wired** — `da_midnight_verify.sh` references it, and two freeze declarations name it.
Those two are the model: **the first has both directions driven, the second is actually
called by something that runs on a schedule.** The other six are green units of unproven
wiring.
