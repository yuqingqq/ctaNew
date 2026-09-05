# SPECIFICATION — L4, the programme-level ceiling: declared before the number, and it must NOT be built against settlement

**Filed** 2026-09-04T14:22Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `0705ebe`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· **nothing computed toward the ceiling** — this round established only what the
files contain · no sealed forward day opened, no write under `data/`, no other
seat's worktree opened.

> **STATUS — `CHECKED` as of 2026-09-05T15:01:14Z.** This document was committed
> at `8f08387` labelled **DRAFT-PENDING-CITATION-CHECK** — a label that lived in
> the commit message only, which is itself the defect rule 13 warns about, since
> a reader of the artifact could not see its status. The check was run at tip
> `1404302` and is recorded in **§6 below** and in
> `REVIEW_GATE1F_NEGATIVE_EXISTENCE_2026-09-05.md` §5. **The specification's
> load-bearing claim reproduces at a 6.5% larger population and the argument is
> unchanged. Three illustrative numbers did not survive and are corrected in
> §6 — the body below is left EXACTLY as filed, as provenance.**

**ROUTING.** Every statement about the data is **CHECKED** at the files this
round. The specification itself is a proposal for the USER/coordinator, not a
finding.

---

## 0. Answer to your item 2 first, because it changes the specification

**Your fear is correct as you framed it, and the fix is to reframe the
quantity.**

**CHECKED at `resolutions.jsonl` (32,150 records, 32,139 distinct slugs):**

| field | what it holds |
|---|---|
| `closed` | **True on 32,139 of 32,139 distinct slugs** |
| **`outcomePrices`** | **`None` on EVERY closed record — all 32,139** |
| `umaResolutionStatus` | `None` on all 32,150 |
| open snapshots carrying `outcomePrices` | **8 slugs of 32,139** |

> **The repository does not record the realised settlement of any market.**
> `resolutions.jsonl` records that markets *closed*, not how they *settled*.

**So an outcome-based ceiling would have to RECONSTRUCT settlement**, and the
inputs for that are exactly the contested ones — `data/pm_5min/prices/` carries
`crypto_prices`, `crypto_prices_twap_thirty` and `crypto_prices_twap_sixty`,
which is the R-253 / Q-DA-142/146 dispute in three directories. **An
outcome-based L4 is therefore BLOCKED on a question this programme has not
answered, exactly as you suspected.**

### The fix: specify L4 against the TERMINAL MARK, not settlement

**Value a perfect entry at the window's terminal mark rather than at the
settlement.** Then:

* **it is indifferent to R-253 by construction** — no settlement statistic is
  consulted, so the contest cannot bind it;
* **it is TIGHTER, not looser** — the terminal mark is a shrunk version of the
  0/1 outcome, so this ceiling is strictly below the settlement one and is
  therefore the more conservative bound, which is the right direction for a
  retirement test;
* **it is what a trader actually harvests** — you exit at the market, you do not
  have to hold to settlement;
* **it is the programme's own existing convention** — identical in shape to
  `markout_cents_per_share` and to DE59's terminal-mark inventory leg.

**The settlement variant should be named in the receipt as EXISTING, LARGER and
BLOCKED**, so nobody later mistakes the smaller number for an error.

---

## 1. The quantity, exactly

**`V_L4` = the maximum cents a perfect-foresight TAKER could have extracted from
resting liquidity, over the declared population, net of declared fees.**

For one market-window `w` with terminal mark `M_w`, at the single declared
decision instant `t*`:

```
for each ask level (p_i, s_i) with p_i <  M_w :  gain += (M_w - p_i) * s_i
for each bid level (p_j, s_j) with p_j >  M_w :  gain += (p_j - M_w) * s_j
V_L4 = Σ_w gain_w  −  fees(traded notional)
```

**CHECKED — the tape supports this.** `raw/<day>/<slug>.jsonl.gz` is
`recv_ns \t [ {market, asset_id, timestamp, hash, bids:[{price,size}], asks:[…]} ]`
— **full depth ladders per side, per snapshot** (sample: 99,460 snapshots in one
btc window; a bid level of 95,290.49 size at price 0.01). 17 day-directories,
1,974 files per day.

### The five restrictions, each declared before the number

1. **ONE DECISION INSTANT PER MARKET-WINDOW — and this is the biggest
   correctness hazard.** The same resting order appears in thousands of
   snapshots; summing across snapshots re-spends the same liquidity and the
   number becomes arbitrary. **Declare `t*` = the window's open + a fixed
   offset, one instant per window, and refuse any multi-snapshot accumulation.**
   This is CLAUDE.md reliability rule 2 (*"1.99 rows/fill, max 23"*) in a new
   file, and it is the way this computation most plausibly returns a meaningless
   number.
2. **PRICE BAND.** A ladder that offers 95,290 shares at 0.01 will dominate any
   unbanded sweep — the deep tail exists precisely because nobody wants it.
   **Declare a band (e.g. levels within ±10c of the touch), report `V_L4` BY
   PRICE BUCKET as well as in total, and treat the unbanded figure as
   diagnostic only.**
3. **PARTICIPATION CAP.** Taking 100 % of resting size is not executable.
   **Declare a fraction (e.g. ≤ 25 % of the size at each level) and report the
   number's sensitivity to it**, since a ceiling that halves when the cap halves
   is a capacity statement, not an information statement.
4. **FEES, STAMPED.** Report **gross and net**, and **refuse to emit a single
   headline number without the fee schedule recorded in the artifact**. If the
   schedule is not in the repository, that is a counted status and the net
   figure is `NOT_AVAILABLE`, never zero (R-506(E)'s ruling, same class).
5. **EXCLUSIONS ARE STATUSES.** Windows with no book at `t*`, no terminal mark,
   or a gap spanning `t*` are counted with reasons and reported beside the
   total (rule 4).

---

## 2. Your item 3 — population, and it touches nothing sealed

**Declared population: 2026-08-19 → 2026-08-28T06:09:00Z (the freeze epoch), all
seven coins.** Entirely pre-freeze, entirely already consumed as development
evidence.

**Nothing sealed is touched, and the reason is structural rather than
procedural:**

* the race days are 09-01, 09-02 (consumed reads), **09-03 (accrued and NOT
  opened)**, and 09-04 onward — **all excluded by the end bound**;
* **08-29 is also excluded.** It is admissible and deliberately withdrawn
  (R-500), preserved for one development read; a ceiling computed over it would
  spend a day the USER set aside;
* **sub-second admissibility does not bind.** L4 is a per-window quantity at one
  instant, so the `hf_ws_v2` boundary (2026-08-24T13:48:54Z) — which governs
  **sub-second** features — does not restrict it. Pre-boundary tape is usable
  for ≥ 1 s resolution, which is all this needs. **So L4's window is wider than
  the r-survey's, and that is a real gain: ~9 days × 7 coins instead of 3.5.**

**Stated as a check rather than an assurance:** the build should assert its day
set against the declared bounds and **refuse** if any file outside them is
opened. That is a one-line guard and it converts this section from a promise
into a predicate.

---

## 3. The bar — declared now, because afterwards nobody can argue with a ceiling

**Report THREE numbers, never one:**

| # | quantity | what it is |
|---|---|---|
| **A** | `V_L4_oracle` — capacity-aware, banded, fee-net | the upper bound |
| **B** | `V_L4_at_measured_skill` — the same computation where the trader has the programme's **own measured discrimination** (the +2.2 to +3.1 pp lift over base rate from the capture ladder, RESULTS.md §0) rather than perfect foresight | the realistic figure |
| **C** | total fees paid on the traded notional | the floor B must clear |

**THE PRE-DECLARED READING:**

| result | what it licenses |
|---|---|
| **A ≤ 0** (a perfect taker cannot cover fees) | **RETIRE THE DIRECTION.** Decisive and unarguable — no predictor, however good, can pay |
| **B ≤ 0** (the programme's own demonstrated skill cannot cover fees) | **RETIRE**, strongly. A marginal ranker improvement will not rescue a negative B |
| **A large and positive** | **NOTHING.** A perfect-foresight bound on a binary is large by construction. **This outcome licenses no conclusion whatever and must not be reported as encouraging** |
| **A large, B small but positive** | the direction lives or dies on discrimination, and **the required lift is computable by the same inversion as the capture ladder** — that is the next question, not a verdict |
| **A large, B positive and material** | the direction survives, and the binding constraint moves to capacity and fees rather than to prediction |

**The asymmetry is the point and it is the same one I specified for the
r-survey: this instrument can retire a direction and cannot endorse one.**
Anyone reporting a large A as good news has read it backwards.

---

## 4. The failure modes I expect, named before the run

1. **Multi-snapshot double-counting** (§1.1) — the way this most plausibly
   returns a number in the millions. **Guard: one instant per window, asserted.**
2. **The penny tail** (§1.2) — 95,290 shares at 0.01. Unbanded, this dominates
   everything and the ceiling becomes a statement about dead liquidity.
3. **Terminal mark undefined in a gap** — must be `NOT_AVAILABLE` with a count,
   never 0.5 (R-506(E) ruled this exact shape).
4. **A silently missing fee schedule** turning "net" into "gross under another
   name" — hence §1.4's refusal.
5. **The population drifting past the freeze epoch** — hence §2's day-set guard.
6. **And the one that would waste the round: reporting A alone.** A without B
   and C is a number that impresses and decides nothing.

---

## 5. What I did not do

**I did not compute any part of it**, as instructed. Everything above about the
data is from schema inspection: one `markets.jsonl` record, the whole of
`resolutions.jsonl` (32,150 records — the only full pass, and it is 6.9 MB), one
`prices/` listing, and the first snapshot of one pre-freeze raw window
(20260826). **No settlement was reconstructed and no value was summed.**

---

## 6. CITATION CHECK — run 2026-09-05T15:01:14Z at tip `1404302` (appended in band; §0–§5 unaltered)

**Verdict: the specification stands. Its central claim reproduces on a population
6.5% larger than the one it was written against. Three illustrative numbers fail
and are corrected here rather than in the body.**

### 6.1 The load-bearing claim — RE-VERIFIED, not merely re-read

Every structural predicate of §0 recomputed over the whole of
`resolutions.jsonl`:

| predicate | as filed (09-04) | as-of 2026-09-05T15:01:14Z | verdict |
|---|---|---|---|
| records / distinct slugs | 32,150 / 32,139 | **34,224 / 34,213** | population grew |
| `closed == True` on distinct slugs | 32,139 of 32,139 | **34,213 of 34,213** | **holds** |
| `outcomePrices` `None` on every closed record | all 32,139 | **all 34,213** | **holds** |
| `umaResolutionStatus` `None` | all 32,150 | **all 34,224** | **holds** |
| open snapshots carrying `outcomePrices` | 8 slugs | **8 slugs** | **identical** |

**"The repository does not record the realised settlement of any market" is
confirmed at the larger n.** The consequence — specify L4 against the terminal
mark, not settlement — is therefore unchanged, and remains indifferent to
R-253 / Q-DA-142/146 by construction.

### 6.2 Citations that resolve by identity

* **R-253** (`COORDINATION.md:17248`) — the Chainlink settlement-source
  correction. Resolves.
* **R-506(E)** — exists; its subject is *"COORDINATOR RULING — TERMINAL-IN-A-GAP
  IS `NOT_AVAILABLE`, A STATUS WITH A COUNT"*, which is the exact shape both §1.4
  and §4.3 cite it for. Resolves.
* **R-500** (`COORDINATION.md:19452`) — withdraws 08-29 from the race and keeps
  it readable; its own block says *"development read"* twice, so §2's phrasing is
  sound. Resolves.
* **Q-DA-142 / Q-DA-146** — both present in the register. Resolve.
* **RESULTS.md §0 capture ladder** (lines 383–389) — base **48.02%**, CONDVALUE
  **+2.2 pp**, HAZARD **+3.1 pp**, break-even **+3.3 pp**, 10% capture
  **+10.3 pp**. Cited exactly. Resolves.
* **`prices/` directories** — `crypto_prices`, `crypto_prices_twap_thirty`,
  `crypto_prices_twap_sixty`, all three present. Resolves.
* **Raw tape schema** — `recv_ns \t [ {market, asset_id, timestamp, hash,
  bids:[{price,size}], asks:[…]} ]`, full depth ladders per side per snapshot.
  Confirmed by decoding a first snapshot. **`20260826` holds 1,974 files —
  exact.** Resolves.

### 6.3 THREE NUMBERS THAT FAIL — corrected

1. **`resolutions.jsonl` 32,150 records / 32,139 distinct slugs** → **34,224 /
   34,213** as-of 2026-09-05T15:01:14Z. The file is a live symlink into the main
   tree and grows continuously. **The document carried a filing timestamp but the
   population line carried no as-of of its own — CLAUDE.md rule 8, broken by the
   reviewer.**
2. **"17 day-directories"** → **18** (`20260819`…`20260905`). Same cause.
3. **"sample: 99,460 snapshots in one btc window; a bid level of 95,290.49 size
   at price 0.01"** → **UNVERIFIABLE AS WRITTEN: the window is not named.** The
   nearest available check, the largest btc window in `20260826`
   (`btc-updown-5m-1787751600.jsonl.gz`), gives **228,036 lines** and a maximum
   size at price 0.01 of **98,687.3** (94,999.26 in its first snapshot).
   **The penny tail is real and of the stated magnitude — §1.2's hazard and the
   price-band restriction stand — but the exact figures cannot be reproduced.**
   A quoted population needs its surface, not only its number.

### 6.4 What this does not change

All three failures are **illustrative**, appearing in §1's restriction rationale
and §4's named failure modes. None enters the definition of `V_L4`, the five
restrictions, the A/B/C reporting requirement, or the declared population. **The
specification is unchanged and the label moves from DRAFT-PENDING-CITATION-CHECK
to CHECKED.**
