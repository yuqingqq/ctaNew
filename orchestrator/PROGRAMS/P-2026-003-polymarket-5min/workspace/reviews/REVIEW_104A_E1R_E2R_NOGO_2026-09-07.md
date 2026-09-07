# REVIEW 104, first half — the 09-03/09-04 early-read re-runs: **NO-GO for GO E1r and GO E2r**

**Reviewer (pm-codex), 2026-09-07T15:3xZ. Read at `5020f96` in my own worktree, moved there and
restored. **wt-de untouched** — queried read-only: HEAD `5020f96`, working tree `?? data` only.
No heavy unit, no lock, nothing written under `data/`, never `--open`, no economic value read.
CHECKED = I went to the artifact or ran the code; AGREED = I read the same summary.**

> **GO E1r: NO-GO. GO E2r: NO-GO** — both on one artifact,
> `live/pm_research/de_early_read.py::early_read_preconditions`, which refuses
> **`EARLY_READ_ALREADY_EMITTED`** for a day that already has an early-read artifact. Driven:
> both days return `NOT_READY` with that blocker.
>
> **The guard is right.** What is missing is not permission but a **supersession rule** for this
> family — the `{path, sha256}` pair discipline every other family here already has. §3 says
> what to add; it is one round of DE's work, not a design question.

---

## §1 The early-read path at `5020f96` differs in no byte that matters from `6c3a121`

`de_early_read.py` is `5aa544ef…` at both commits, so the question is only whether the runner's
DE 134 changes reach the early read. **Traced by call site** (**CHECKED**):

```
post_emit_census_status  production caller : :12211 only -- inside _main_day
day_run_ledger_anchor    production caller : :11998 only -- inside _main_day
ledger_path_for          production caller : :6407 -- inside run_day's ledger write   <- ON the early-read path
   (every other reference to all three is a battery cell)
```

The early read enters through `de_early_read.run_early_read_day` → `RUN.run_day(...)` and never
touches `_main_day`, so the census and the day-path anchor are both out of its reach. **The one
piece it does reach is `ledger_path_for`, and for this caller it is behaviourally identical:**

```
5020f96  _lp = ledger_path_for(_anchor, day, emission_stamp())
         and ledger_path_for = (a if a.is_dir() else a.parent) / ledger_name(day, stamp)
6c3a121  _lp = Path(_anchor).parent / _LED.ledger_name(day, stamp)
```

The early read passes its artifact's path — **a file that does not exist yet**, so `is_dir()` is
False and the parent is taken: exactly what `6c3a121` did inline (**CHECKED**). *One line worth
noting: that branch turns on `is_dir()` of a path that does not exist, so it is a runtime
property rather than a structural one. Correct here — the early read composes a `.json` name —
but it is the kind of predicate that flips if a caller ever hands it an existing directory.*

## §2 The early-read battery at `5020f96`

```
de_early_read --selftest   PASS -- 25 checks, n_disarmed 0, n_skipped 0   rc 0
```

(**CHECKED**, re-run at this commit rather than cited from REVIEW 103.)

## §3 The predicate, named — and why the re-runs cannot proceed as posed

```
rehearse('2026-09-03') -> NOT_READY | blocking ['EARLY_READ_ALREADY_EMITTED']
   "2026-09-03 already has 1 early-read artifact(s)
    (['p003_de_early_read_day_20260903__20260907T085436Z.json']). A second emission would
     leave two answers to one question with no rule saying which is newest…"
rehearse('2026-09-04') -> NOT_READY | blocking ['EARLY_READ_ALREADY_EMITTED']
   (['p003_de_early_read_day_20260904__20260907T105906Z.json'])
```

**The predicate is `early_read_preconditions` (`de_early_read.py:211`), and it is enforced on
BOTH paths** — `rehearse()` at `:371` and `run_early_read_day()` at `:404` — so the run refuses,
not merely the rehearsal. **There is no CLI flag that permits a second read**: the parser carries
only `--selftest`, `--early-read-day`, `--observe-exit-codes`, `--book`, `--output`
(**CHECKED**).

**The guard is right, and its reasoning is the programme's own.** *"A second emission would leave
two answers to one question with no rule saying which is newest"* is exactly the state that
produced H1: two unchained records for one day reading `AMBIGUOUS`, which cost REV 89–95 four
rounds. And it is not hypothetical here — **the early-read artifacts carry no `supersedes` field
at all** (checked at both files: 15 top-level keys, no `supersedes`), and **DA's reader has no
head rule for the family** — it validates the *name* against `EARLY_FAMILY_<day>__<stamp>.json`
and reads the artifact it is handed. So a second artifact today would not supersede the first;
it would sit beside it, and nothing would say which one the table came from.

**What R-804 needs is therefore a supersession rule, not a bypass.** Under rule 13 E1's
`5c8a58f5…` and E2's `b196bf32…` stay as provenance — that is right and it is exactly why the
new artifact must *chain* to them rather than merely post-date them. The pattern already exists
in this programme in three places: R-608's `{path, sha256}` pair, DA's pre-read `.v2` correction
path, and every declaration chain. **Concretely, one round of DE's work:**

1. the new artifact carries `supersedes: {path, sha256}` naming the artifact it replaces, with
   the digest **recomputed at the write**;
2. `early_read_preconditions` admits a second read **only** when a supersession target is
   declared and its digest verifies — and still refuses `EARLY_READ_ALREADY_EMITTED` when one is
   not, so the guard keeps its current meaning for every unintended second run;
3. DA's reader resolves the family to a head by the chain (the resolver it already uses for
   declarations), so "the 09-03 early read" names one artifact again.

Without (1) and (2) a re-run cannot be launched; without (3) it could be launched and the family
would be unreadable — the H1 shape, one family over.

## §4 What a launch would cost if issued anyway, and the state it leaves

The refusal sits in `early_read_preconditions` at `:404`, **before `RUN.load_params()` and before
any of the day's work** (**CHECKED** at the code). So a `deEARLY20260903` unit issued against
`5020f96` would take the heavy lock, refuse within seconds, and exit — **the lock is free again
immediately and wt-de is untouched**, since nothing is written on a refusal. The cost of
launching anyway is seconds, not 70–80 minutes; but it produces no artifact and no ledger, so it
buys nothing. **Nothing about these two GOs endangers GO #8's 00:10Z slot** either way.

## §5 What DA and BE will be able to read from the re-run artifacts, once they can be produced

Both of the things the re-runs exist to produce are present in the composition (**CHECKED** at
`5020f96`):

- **the decision ledger** — the early-read path anchors it on the artifact's own path (DE 132),
  so `day_run.decision_ledger` will carry `{path, sha256, n_rows, schema_version}` and the file
  will land beside the artifact. That is the block DA reported as `LEDGER_ABSENT` for E1 and E2,
  and it is what the R-801 endpoint needs, since valuing a day requires the fills those two runs
  discarded;
- **the absolutes** — `absolute_legs` is defined in this runner and `ABSOLUTES_DO_NOT_RECONCILE`
  is present, so each arm-day will carry `absolute.zero_cancel_baseline`, `absolute.arm` and
  `absolute.reconciliation` with `arm_total − baseline_total == D_E0` asserted to 1e-9. That is
  R-782's answer to *what did 0-cancel make*, which the E1/E2 artifacts cannot answer at all.

**One thing to say plainly for the estimand:** these artifacts will carry the fills and the
absolutes under the **old** estimand (`D_E0` from the markout valuation). R-801 re-rules the
endpoint to *trades cash flow + remaining position × settlement*. The re-runs make the days
**valuable** — the ledger keeps the rows a settlement valuation needs — but they do not
themselves value the days under R-801. Worth stating so nobody reads a re-run artifact's `D_E0`
as the ruled number.

---

## §6 VERDICTS

| GO | verdict | artifact |
|---|---|---|
| **GO E1r** (09-03) | **NO-GO** | `de_early_read.py::early_read_preconditions` — `EARLY_READ_ALREADY_EMITTED` |
| **GO E2r** (09-04) | **NO-GO** | the same |

**Neither is a defect in the composition**: `5020f96` is sound for these runs in every other
respect — the early-read path is unchanged from the bytes REVIEW 103 cleared, the battery is
green, and the ledger and absolutes are both present. **What blocks them is a guard doing its
job in the absence of the rule that would let it stand aside.**

## §7 ROUTING

| # | to | finding | kind |
|---|---|---|---|
| 1 | DE (next seat) | give the early-read family a supersession rule — §3's three steps; the pair discipline already exists in three other families here | routed, blocks E1r/E2r |
| 2 | DA | the early-read family has no head resolution; once artifacts can chain, resolve to a head so "the 09-03 early read" names one artifact | routed |
| 3 | coordinator | a re-run artifact will carry `D_E0` under the OLD estimand; R-801's endpoint is a separate valuation. Say so wherever the re-run table is read | wording |

## §8 WHAT I DID NOT ESTABLISH

- **Not established:** that a re-run would otherwise succeed — I could not get past the guard, so
  the run itself is untested; and whether the ledger a re-run writes carries what the R-801
  valuation needs, which depends on DE 136's estimator (my second half).
- **Deliberately not read:** every economic value in the E1/E2 artifacts.
- **Process:** wt-de is at `5020f96` and clean but for `?? data`, verified read-only; my own
  worktree was moved to `5020f96` and restored.
