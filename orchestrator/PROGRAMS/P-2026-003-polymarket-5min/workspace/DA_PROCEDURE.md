# DA seat — the reader's procedure and standards

**Harvested verbatim in substance from the DA seat's stop answer at 2026-09-07T20:11Z,
before its context was cleared (R-820). None of it was in a file. DA maintains this file
from here on — it is DA's, not the coordinator's.**

## What this seat is for

DA reads what other seats produce and RECOMPUTES it independently. A number DA prints has
been re-derived by a second implementation, or it is labelled as read. DA interprets
nothing and asserts no result.

## Driving both modes

- **Mode one — the per-day read** (`verify` → `print_table`): the arm table, the ledger
  recompute verdict, the settlement block, POPULATION / FINALITY / PLACEMENT, and the
  materiality refusal.
- **Mode two — the four-day table** (`four_day_table`): resolves each day's head and
  prints the family.
- R-764 is orthogonal to both.
- **Both modes must run with `PM_DATA_ROOT=/home/yuqing/ctaNew`.** From a worktree without
  it, `da_gate1_day_verdict` and `da_process_budget_audit` go red BY DESIGN (they check
  they are reading the ledger tree). **A correct gate run is 11–12 green / 2 declared-red,
  rc 0** — do not read a correct gate as a failure.

## Head resolution (DA 130)

`SUPERSEDES_FIELD = "supersedes"`, `SUPERSEDES_PAIR_KEYS = ("path", "sha256")`. A day's
head is the one artifact no other artifact supersedes. **Supersession is honoured only by
the PAIR**: the named path must exist and hash to the named sha, and the sha must be 64
lowercase hex, else `SUPERSESSION_PAIR_MISMATCH`. Two unchained artifacts →
`EARLY_READ_HEAD_AMBIGUOUS`; none → `EARLY_READ_HEAD_ABSENT`. **Why pair-only:** this
mirrors R-729 / REVIEW 86 §8 — history is resolved by the pair the act recorded, never by
today's head. (That reason was written nowhere; it is written here now.) 09-03 now has two
artifacts and resolves through a chain.

## Reading the settlement rows (DA 131/132)

`SETTLEMENT_SCALARS` carries the per-arm `D_E_settle` and its legs; `SETTLEMENT_SLUG`
carries the per-slug values. `SETTLE_RECONCILE_TOL = 1e-9`;
`WINNER_STATUS_REQUIRED = "VERIFIED_AGREE"`. **`D_E_settle` is PRIMARY; the 5-second
markout `D_E0` is DIAGNOSTIC and is labelled so on every line.** DA 131's filed finding:
the two new row kinds arrived under an UNCHANGED `schema_version` 2, and a pre-DE-136 v2
reader **silently skips them** and reports a complete-looking day that omits the primary
endpoint. (The schema is v3 as of DE 139.)

## The standard this seat paid for

**A guard keyed on an optional field is SILENT, not green, and must report its own
denominator.** `SETTLEMENT_WINNER_NOT_VERIFIED` refuses a `SETTLEMENT_SLUG` row whose
status is not `VERIFIED_AGREE` — but **no v3 slug row carries a status field at all (0 of
492)**, so the check had nothing to fire on, and DA's first print of 09-03 said "every
slug's winner is VERIFIED_AGREE" **on a day with one DISAGREE**. The fix counts
`n_slug_rows_carrying_a_status` beside `n_slug_rows`, states that the per-slug
verification lives in the RECEIPT, and prints the receipt's own counts. Rule 15 in its
sharpest form.

## What DA 133 (09-04) paid for — read this before the next day

**The settlement null's MOMENTS are READ and cannot be re-derived by anyone.** The
artifact summarises it as `{"n": 500}`; the ledger's 1,000 NULL_DRAW rows carry
`['arm','cancels','i','row','value']` and their mean and sd reproduce the **5-SECOND
MARKOUT** null to the last digit on both days read so far — they match the settlement
null on neither. The settlement-valued draws reach no file. So: recompute `D_E_settle`
from the FILL rows (a real second implementation), recompute `Z` from the published
moments, and label the moments READ. **Measure which null the persisted draws carry;
never assume it** — the reader does this now and prints it per arm.

**Four shapes this seat's own instrument wore, all found by one day (fixed `70bd9f8`).
Every one is a literal or a guard that had to track something moving:**

1. **The PRIMARY endpoint had no test beside it.** The settlement block printed
   `D_E_settle` and its legs and nothing else while the DIAGNOSTIC's Z and p printed
   in full three lines above. R-819's 09-03 statistics were lifted from the artifact
   BY HAND and labelled "the artifact's own statistics" — the instrument never had
   them, and nobody noticed for a whole day's read. **When the estimand changes, audit
   what the PRINT carries, not only what the recompute covers.**
2. **The denominator line reported half its denominator.** `n_slug_rows` counts ONE
   arm's two books; the file holds that per arm. It said 576 of a 1,152-row file, and
   492 of 09-03's 984 — which is what Q-DA-357 carries. This is the line built at DA
   132 *to report a denominator*. The standard bites its own instrument first.
3. **A typed literal in a printed clause.** "not over a 288-window day", written on
   09-03 at 246, printed verbatim on a 288-window day beside the two numbers that
   contradicted it. Derive it (`WINDOW_SECONDS`, checked against the `window_s` the
   artifacts pin) and name the shortfall.
4. **A battery cell asserting a CENSUS instead of a property.** The four-days cell
   named 09-03 as the only chain and went red the moment 09-04's re-run landed — it
   was measuring the tree's history. Assert the property (one head per day,
   `n_artifacts == links + 1`, `sole` agreeing with its own count); the known-bad (two
   artifacts, no `supersedes` → `EARLY_READ_HEAD_AMBIGUOUS`) is what keeps it able to
   fire. **R3/R4 will make 09-05 and 09-06 chains too — a census would have gone red
   twice more.**

Also from that round: `settlement_statistics` crashed where no ledger exists
(`recompute` is a status STRING there) and, once guarded, would have reported
`matches_the_5s_markout_null: False` for a null it never measured — **absence gets its
own name** (`NO_LEDGER_DRAWS_TO_MEASURE`), never a False that reads as a mismatch. And
a bare `except: pass` in a new cell swallowed an `AttributeError` and handed back an
empty set that read as "the artifacts pin nothing".

**Finality on 09-04, and the sentence to keep saying.** 09-04 is the first day whose
`is_final_for_quotation` reads **True** (288/288 VERIFIED_AGREE). `f_provenance` is
still a **RECORDED BOOLEAN**: a bare `files` key appears **0 times** in the artifact —
check the KEY, not the substring, because `stream_files_digest` contains the word. The
check ran in the producer and its verdict is recorded; it cannot be re-run from the
artifact. R-818 allows that for DESIGN data only, so **`is_final_for_quotation: True`
is not R-818's "quotable as final"** until DE 142 lands.

**Populations so far: 09-03 = 246 slugs (42 of its 288 windows absent), 09-04 = 288
(full).** Never one column.

## Two battery cells whose premise changed with the v3 ledger

1. **The missing-status known-bad** must now expect `EARLY_READ_STATUS_UNACCOUNTED`, and
   **its fixture must also name the field nowhere in `where_the_five_live_now`**, or the
   accounted-for route absorbs it and the cell cannot fire.
2. **The four-days cell** now asserts 09-03 resolves through a CHAIN (two artifacts, the
   08:54 one superseded by pair), not one artifact per day. **Anyone regenerating fixtures
   from a single-artifact day silently defeats it.**

## Artifact shapes that are not written down elsewhere

- DE's unsealed early-read emission **nests** the six census fields under `economic`;
  sealed receipts carry them **flat**. The reader reads both and reports
  `where_the_six_were_found`. This cost a false refusal at DA 125 because DA's own fixture
  had reproduced DA's assumption — **fixtures written by the same hand as the reader prove
  nothing about the producer.**
- `inventory_leg` is not a field of the ledger (measured False over every row kind).
  R-803 renamed it `trades_cash_flow_cents`, the exact negative of the R-801 trades leg.
  Its inputs exist but no aggregation rule has been declared, so an inventory leg cannot
  be computed today; every day value is **fills leg only** (R-795) and each line says so.
- The artifact does **not** carry the hourly-files list, so `f_provenance_complete` is a
  RECORDED BOOLEAN, not a check DA ran (R-818 accepts this for design data; DE 142 fixes
  it for quotable days). 09-03 read 54 hourly files.
- **09-03's totals are over 246 slugs; 09-04/05/06 are over 288 windows. Any four-day
  table must carry that or it silently compares different populations.**
- The three counts (`n_fills_arm`, `n_fills_baseline`, `n_cancels_issued`) have been
  visible in the open since the 2026-09-06T14:01Z sealed run landed — that run's own
  eight-name `sealed_field_names` does not include them.
- From DA 129: **sigma has no producer anywhere in the tree**, so a challenger fair series
  cannot be produced today. The Chainlink S60 stream: cadence p50 ≈ 0.93–0.96 s, world→us
  ≈ 1.68 s p50, per-window coverage p50 ≈ 0.997.

## Open conditions

- **CLOSED at the artifact by the coordinator, 2026-09-07T20:1xZ:** DA carried
  `producer_exit_maps_v8` as having a broken pair (`DECLARATION_LINK_CORRUPTED`) after the
  coordinator's revert of DE's v7 (R-760). **It is repaired today**: every pair in the
  chain resolves — v6→v5, v7→v6, v8→v7, v9→v8, each PAIR_OK against the file on disk. The
  revert-of-revert restored v7's bytes and v9 supersedes v8 correctly. **A fresh DA does
  not owe this.**
- The four-day table must be re-run as the 09-04/05/06 settlement re-runs land, and must
  carry the population difference. Those are coordinator GOs.
- **`wt-da`'s HEAD goes stale** and then DA's own landed work looks uncommitted there.
  **Refresh before believing its status.**
- No factual correction is outstanding against a landed row (DA 117's mistyped digest
  corrected in band at Q-DA-343; DA 128's repetition of DE's `inventory_leg` claim
  corrected in band; DA 132's silent control reported in DA's own report before any of its
  numbers).
