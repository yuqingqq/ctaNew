# Review — the coordinator's own acts of 2026-09-02
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co), no scope list by design

**Subject:** R-416 §3(a), R-418, R-419 (including the transient timer and the format
adoption), and the sequencing decision recorded in R-420.
**Tip read: `649569e`.** **Composed 2026-09-02T11:11:55Z.** One filing, per R-377.

Permitted acts only: worktree reads, `systemctl --user show/list-timers`, the preflight
against **09-01** alone. No production mode, no bare launcher run, no write under
`data/pm_5min/derived/` (verified: my own `ls --time-style=full-iso` digest of that
directory is identical before and after everything below), no timer or service started.

**On the frame.** The dispatch names no scope list and cites the lesson that reviewers
inherit the frame they are given. I have taken that seriously: below, the finding I
consider most consequential is one the entries do not raise, and one of the items is a
**correction to my own recommendation** that the coordinator was right not to adopt.

---

## Verdict

### The acts are sound in their reasoning and mostly in their execution. Four findings, one of which is decision-bearing tonight; and one withdrawal of my own.

Nothing here decided something the register reserves to the USER, introduced a research
number, or chose after seeing — I checked each of those and say how below.

---

## CO-R1 — MEDIUM, and live tonight — the ratified `present_source` runs ahead of the data on an open day

R-418/R-419 ratify the population as *"every window `de_admissible_windows.supply(D,
present)` emits, `present` read from the day's own market ledger
(`data/pm_5min/markets.jsonl`, the windows that existed), minus the committed mask"*,
with `scope_days: FORWARD_RACE_DAYS`, `scope_from: 20260901`, **`scope_to: null`**.

I compared the ratified source against the tape the mask is computed over:

| day | per-coin: raw tape | per-coin: market ledger | ledger-only | tape-only |
|---|---|---|---|---|
| **20260901** (closed) | 288 | 288 | **0** | 0 |
| **20260902** (open, read 11:10Z) | 133 | **135** | **2** | 0 |

On the closed day the two agree exactly (2,016 = 2,016). On the **open** day the ledger
leads the tape by the in-flight windows — for btc those are exactly **11:05 and 11:10 at
11:10Z**, i.e. the trailing pair, 14 windows across seven coins.

**Why this matters.** DE's supplier subtracts the mask from `present` and refuses when the
mask masks a window `present` does not contain — that direction is guarded. The reverse is
not: windows in `present` with **no tape behind them** are *supplied as admissible*. That
is the same shape as the defect DA found and fixed one layer down — *"`range(288) − masked`
credited an open day with every window that had not happened yet … the empty-set trap
inside the complement"* — re-introduced above it by changing the source of `present` from
the tape to the ledger. The ratification is open-ended and carries no settled-day
precondition, and BE's production run path is its consumer.

The programme already owns the predicate that fixes it: the mask carries
`day_closed_calendar`, and BE's adapter refuses an open-day mask (I verified that last
round). But that gate sits on the **mask**, not on `present`.

**Closure:** make the ratified population conditional on the day being settled — either
name the tape's own window set as `present_source`, or add `day_closed_calendar: true` as
a precondition of the ratification. Either is a restatement, not a number.

## CO-R2 — MEDIUM — the format is adopted in the same entry that defers its enforcement

R-419 §4 adopts the block format and rules that *"prose binding is admissible for exactly
one entry — R-418 — grandfathered by ref; any other entry without a block is refused by
name."* The enforcement is dispatched to DE round 9.

Executed at this tip: `check --ref R-418 --day 20260901` → **verified, `binding_source
PROSE`**; `--ref R-419` → **verified, `binding_source BLOCK`, `unverifiable: []`**. So
today **both refs verify**, and prose binding still admits any entry carrying the
vocabulary — which the coordinator's own CO-4 fixture demonstrates (an entry whose body is
a recap sentence returns VERIFIED) and which R-419 itself predicts: *"every coordinator
sweep entry from here on (this one included) will carry that vocabulary."*

So for the interval between this entry and DE round 9, **the register states a rule that
nothing checks** — rule 15's shape applied to the register itself, which is the standard
the register set for everyone else. The disclosure is complete and the dispatch exists;
what is missing is that the adoption took effect on filing rather than on landing.

**Closure:** either date the adoption to round 9's landing, or say in the entry that the
rule is declared-but-unenforced until then, so a reader who stamps a receipt this
afternoon knows which regime they are in.

## CO-R3 — MEDIUM — supersession-as-refusal, as dispatched, will make existing R-418 receipts unverifiable

R-419 §5 supersedes R-418 in band and states that BE round 3's *"sealed scratch receipts
stand as provenance under that ref."* R-419 §6 dispatches round 9 to make supersession a
**refusal**: *"a ref that a later block `supersedes` REFUSES naming the superseder (so a
new run stamping `R-418` after this entry refuses)."*

The intent distinguishes a receipt stamped **before** the supersession from one stamped
after. The mechanism named — *a forward scan of later entries' blocks* — has **no notion
of when a receipt was stamped**. Implemented literally, every existing R-418 receipt stops
verifying the moment round 9 lands, and "stands as provenance" becomes a claim its own
checker refuses. That is the coordinator's own criterion — a claim left unverifiable at
its artifact — applied to the receipts this entry promises to protect.

**Closure, before round 9 is written:** give the check the receipt's stamp time (or its
commit) and evaluate supersession against it; or grandfather by receipt date rather than
by ref. Either keeps *"stands as provenance"* true after round 9 lands.

## CO-R4 — LOW — the 00:14Z timer cannot report the one failure it exists to catch

The transient timer runs `da_governed_verdict_preflight --day 20260902 --json` with
stdout captured to `~/.local/state/pm-co/preflight_20260902.json`.

Executed against 09-01 (permitted): rc **1**, `classification PRE_GOVERNED_ARTIFACT`,
9 predicates, `decides_nothing`, `read_only: true`, and my before/after digest of
`derived/` **unchanged** — the tool is well-behaved and writes nothing.

Executed against a day with **no verdict** (20260820): the tool refuses correctly by name
— *"an absent verdict is not a failing day … would be the empty-set trap"* — but it does
so as an **uncaught `PreflightRefused` traceback on stdout**, with rc **1**: the same rc as
the healthy pre-governed read. So if the 00:06Z run is late or fails, the timer's JSON
capture receives a traceback rather than JSON, and the exit code alone cannot tell "no
verdict exists" from "the verdict is pre-governed" — in the single case the 00:14Z read
exists to detect.

**Closure:** under `--json`, emit a JSON refusal object and use a distinct exit code for
*nothing was verified* — the programme already reserves rc 4 for exactly that in the
verifier.

**On the timer's durability, stated precisely rather than as a finding:** it is armed
(`NextElapseUSecRealtime = Thu 2026-09-03 00:14:00 UTC`, active/waiting) and it is a real
box-level leg — `Linger=yes`, so it survives logout, unlike the two session schedulers.
It lives in `/run/user/1001/systemd/transient/` with `Persistent=no`, so it does not
survive a reboot and a missed firing is not caught up. Three legs with different failure
modes is the right shape; "box-level" is worth reading as "manager-level, runtime-only".

---

## What I checked and found sound

**No number was introduced.** `scope_from: 20260901` is presented as a restated fact, and
it is one: across every verdict on disk, **09-01 is the only day with
`race_accrual_eligible: true`** (08-28 F, 08-29 F, 08-30 F, 09-02 F). It names the first
accruing day rather than choosing a threshold. `00:14`, `MemoryMax=2G` and the format's
field list are schedule, resource and shape — none is a research parameter.

**Nothing reserved to the USER was decided.** R-418/R-419 reserve R-411(i), R-411(ii),
every accrual call and Phase-2 admission explicitly, and I found no field in either entry
or in the emitted artifacts that decides one. The preflight emits `decides_nothing` and an
`open_decisions` list; the mask's `disposition_rule`/`disposition_text` cite R-409 and say
*"This artifact REPORTS the mask; it decides nothing (rule 14)."*

**R-419's own claims verify at their artifacts.** `check --ref R-419 --day 20260901`
reproduces `binding_source BLOCK`, `day_in_scope True`, `unverifiable []`,
`n_supplied_total 1875 == Σ (n_present − n_masked_applied)`. The checker's own suite is
24 checks under both launchers.

**R-416 §3(a) — accepting DA's two-leg deviation — stands.** I weighed it independently
last round and re-tested the pattern here: it cannot match a differently-named unit, and
beyond the scheduled unit it admits only something deliberately given that name. Accepting
a deviation from the coordinator's own dispatch, on the seat's measured argument, is the
process working.

## A withdrawal: one leg of my own RR12-1 closure was wrong, and the coordinator was right to defer

R-420 defers RR12-1's fix to DA round 10, *after* tonight, because **the unit executes the
shared tree**. I verified the premise: `ExecStart=/home/yuqing/ctaNew/live/pm_research/da_midnight_verify.sh`.

That makes the second half of my own closure line wrong. I wrote: *"default `V` to
`$(dirname "$0")/da_forward_day_verify.py`."* But the launcher's hardcoded default is
precisely what keeps a worktree rehearsal exercising **the same verifier the unit will
run**. Deriving `V` from the script's own location would make a rehearsal test worktree
code and the unit test shared-tree code — the rehearsal would stop being a rehearsal of
anything. **I withdraw that leg.**

What stands is the provenance leg: `da_blackout_mask.REPO` hardcoded to the shared tree
means an artifact produced from a worktree records another tree's HEAD and cleanliness,
and `module_sha256_prefix` (from `__file__`) can disagree with the module at
`carrying_commit` without anything noticing. **The right pair of fixes is: derive `REPO`
from `__file__`, keep the executed verifier pinned, and have the run record which tree it
exercised** — so provenance follows the bytes while execution stays on the code that runs.
The coordinator's sequencing — after tonight, in DA's round 10 — is correct either way.

---

## Executed evidence

At `649569e`, 2026-09-02T11:08–11:11Z:

| check | result |
|---|---|
| `derived/` digest before vs after everything | **identical** — nothing written |
| `de_ratification_check --selftest` | 24 checks, both launchers |
| `check --ref R-418 --day 20260901` | verified, **PROSE**, `unverifiable: ['day_in_scope']` |
| `check --ref R-419 --day 20260901` | verified, **BLOCK**, `unverifiable: []`, 1875 == Σ |
| ledger vs tape, 09-01 (closed) | 288/288 per coin, **0** asymmetric — exact |
| ledger vs tape, 09-02 (open) | ledger **135** vs tape **133** per coin; the ledger-only pair is btc 11:05/11:10 — CO-R1 |
| first accruing day on disk | **09-01 only** — `scope_from` is a restated fact |
| preflight on 09-01 | rc 1, `PRE_GOVERNED_ARTIFACT`, 9 predicates, `decides_nothing`, `read_only` |
| preflight on a day with no verdict | refuses by name — but as a **traceback on stdout, rc 1** — CO-R4 |
| `co-preflight-20260902.timer` | active/waiting, next elapse **2026-09-03 00:14:00 UTC**, transient in `/run/user/…`, `Persistent=no`; `Linger=yes` |
| `da-midnight-verify.service` ExecStart | **the shared tree** — the premise of R-420's sequencing |
| timers in the family | `da-midnight-verify.timer` next **2026-09-03 00:06:00 UTC** |

---

## Disposition

- **No hold.** None of these findings should stop tonight's run; three of the four are
  about what happens after it.
- **CO-R1** is the one to act on before a forward-race day is scored from the ratified
  population: bind it to a settled day.
- **CO-R2** and **CO-R3** are about the interval between a rule being declared and being
  enforced, and about keeping *"stands as provenance"* true once it is.
- **CO-R4** is small and worth fixing because the timer's whole purpose is the case it
  currently cannot report.
- **Withdrawn:** the `V`-from-`$0` half of my RR12-1 closure. The coordinator's deferral
  and its reasoning are right, and the fix should split provenance from execution rather
  than move both.
