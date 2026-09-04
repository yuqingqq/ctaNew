# REVIEW — DE is right and my bound flips; and no value ceiling has ever been computed in either programme, on a field that has existed for ten days

**Filed** 2026-09-04T13:46Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `b22fb30`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· read-only against the code, the git history, and the arms artifact · no sealed
forward day opened, no write under `data/`, no other seat's worktree opened.

**ROUTING.** Every search result and every computation is **CHECKED**. DE's two
tail-ranking figures (1.13 and 0.10) are **AGREED** — I have no per-fill access
and did not re-derive them; everything I do with them is arithmetic on top.

---

## 1. Your item 3 — I accept DE's reading in full, and the bound does not weaken, it FLIPS

**DE is right, and it is right about the method and not only the number.**
Ranking by signed net and removing the top 43 removes *only the good outliers*,
which is selection; the remainder is then negative partly because you chose the
43 fills that reduce it most. **A robustness statement must de-tail
symmetrically, by |P&L|, which is what the extreme ranking does.**

Both bounds, from the same algebra — for a removed set `T` with spread `x` and
net `n_T`, `r_ex = (|ADV| + n_T − x)/(S − x)`:

| ranking | removed net | remainder net | numerator at x=0 | vs denominator 10,566.95 | direction | **bound** |
|---|---:|---:|---:|---|---|---|
| **winner-ranked** top 43 (1.13) | 9,716.60 c | **−1,117.84 c** | 11,684.79 | **greater** | increasing ⇒ min at x=0 | **r_ex ≥ 110.58 %** |
| **extreme-ranked** top 43 (0.10) | 859.88 c | **+7,738.88 c** | 2,828.07 | **less** | decreasing ⇒ **max** at x=0 | **r_ex ≤ 26.76 %** |

**The inequality reverses.** Under the ranking a specification should be written
against, `r_ex` lies somewhere in **[0 %, 26.76 %]** — and **26.76 % sits BELOW
CONDVALUE's 27.60 % break-even** and above HAZARD's 19.88 %.

### What I withdraw, plainly

**"On 99 % of this book the threshold is already exceeded by a factor of four"
is WITHDRAWN.** It was true of the winner-ranked remainder and it is not a
statement about the book minus its outliers. On the extreme ranking the de-tailed
book **does not clear CONDVALUE's break-even at all** — the upper bound is below
it — and clears HAZARD's only if `r_ex` sits in the upper part of its range,
which is unmeasured.

**Both bounds remain individually correct and neither is now load-bearing**, for
a reason that is mine rather than DE's: **R516-R1 established that `r` does not
determine viability.** So the counterweight I filed two rounds ago was wrong
twice over — **wrong ranking and wrong statistic** — and both corrections point
at the same replacement. **The de-tailed quantity a specification should be
written against is `V_oracle` computed on the de-tailed book**, which is one more
line in the same pass BE is already running.

**One thing worth keeping from the exchange:** that the two rankings disagree
this violently — 1.13 against 0.10 on the same 43-fill budget — is itself a
measurement of the book. It means the extreme set contains large losers roughly
offsetting the large winners, which is exactly why a signed-rank de-tailing and a
magnitude de-tailing give opposite answers. **That disagreement should travel
with any tail statement this programme makes.**

---

## 2. Your item 1 — I searched both programmes. **No value ceiling has ever been computed. One ceiling exists, and it is for a different lever.**

**What I searched** (CHECKED): the full ceiling vocabulary — `oracle`,
`perfect.foresight`, `upper.bound`, `ceiling`, `headroom`, `best.possible`,
`max.achievable`, `attainable` — across `live/pm_research/` (P-2026-003, the
whole surface) and `live/mm_research/` (P-2026-002, all 11 files), plus both
programmes' registers and `RESULTS.md`; and separately the *shape* rather than
the vocabulary (`pnl < 0`, sums over losing fills).

| what exists | what it is |
|---|---|
| **`skew_bound.py`** | **A REAL CEILING — for the SKEW lever.** It bounds the published 15× skew claim's ~40 % fill increase by removing the generous re-post idealisation. A genuine "how much of this could be real" instrument |
| `be_fill_ledger.py` `*_UPPER_BOUND` fields | **attribution** bounds (row-window sum against deduped shares/notional/markout) — accounting envelopes, not policy value |
| `de_constraints.max_size` "feasibility oracle" | a constraint checker |
| `da_population_audit` "the FOURTH ORACLE" | R-509's oracle-of-claims vocabulary — an epistemic oracle, not a value one |
| `harmful_candidate_manifest` "ceiling" | a **memory** ceiling |
| the four `v < 0` sites | all **counting** negative windows/units for reporting; **none sums them** |
| **`live/mm_research/` — the whole of P-2026-002** | **ZERO hits on the entire vocabulary, across all 11 files** |
| both programmes' registers and `RESULTS.md` | **zero hits** on ceiling language |

**So the answer is: it does not exist, with one instructive exception.**

**And the exception sharpens the finding rather than softening it.** This
programme **knows how to build a ceiling — it built one for the skew lever** and
built it well, by attacking an idealisation rather than by measuring a
comparison. **It then spent months on the cancellation lever and never built one
there.** That is not a capability gap; it is a gap in what got asked.

---

## 3. Your item 2 — YES, and the date is 2026-08-25 08:21. Ten days.

**The per-fill P&L has been in the tranche record since the dataset every
subsequent result is built on was created.** `harmful_exposure_rows.py:309-313`:

```python
g["tranches"].append({
    "t": f["t"], "shares": f["shares"], "level": f["level"],
    "markout_cents_per_share": (None if later is None
                                else sgn * (later - f["level"]) * 100.0),
})
```

`markout_cents_per_share` **is** the per-fill P&L in cents per share
(`later = wf.mid_at(f["t"] + MARKOUT_S)`, `MARKOUT_S = 5.0`, `:77`/`:308`). So

> **V_oracle = Σ over tranches with `markout_cents_per_share < 0` of
> (−`markout_cents_per_share` × `shares`)** — one filter, one multiply, one sum.

**It entered at `854115f`, 2026-08-25 08:21** ("exposure dataset v2 … tranche
valuation, per-latency preventable value — rebuilt per the user's eight-issue
review"), verified by `git log -S`.

**Everything this programme has produced post-dates it:**

| milestone | date |
|---|---|
| **`markout_cents_per_share` exists** | **2026-08-25 08:21** |
| freeze epoch | 2026-08-28 06:09 |
| iteration 011 artifact | 2026-09-02 05:21 |
| the whole forward race | 2026-09-01 → |
| every §8.1 arm | 2026-09-04 |
| **`maker_pnl_from_fills` — the function that finally decomposes it** | **2026-09-04 12:56 (`0e8f40c`)** |

**And DE's own commit subject records the discovery in six words:**
*"the P&L found already present under another name."*

**So, plainly, as you asked: yes.** A number that bounds the entire overlay case
— *how much is there to win at all* — has been one filter and one sum away for
**ten days**, on the same records every arm, every null and every operating point
has been read from. **It was not computed because nobody asked for a ceiling,
not because anything was missing.**

---

## 4. What the method finding is, stated without inflation

**It is the same shape as today's other one, and you named it correctly:** we
measured whether a policy beat a baseline without ever measuring whether **any**
policy could. The two are not substitutes — a comparison tells you about the
policy, a ceiling tells you whether the question was worth asking — and only the
first has ever been run on this lever.

**Three qualifications, so this is not over-read:**

1. **It is a finding about what was ASKED, not about competence.** `skew_bound.py`
   is a good ceiling, built by this programme, for another lever.
2. **A ceiling would not have made the ranking work worthless.** The ranking
   result stands on its own — 3.01× and 3.81× adverse concentration at
   below-average spread — and a ceiling would have told you *what it was worth*,
   which is a different thing from *whether it is real*.
3. **The cost of not having it is bounded and knowable now.** `V_oracle` is in
   BE's re-dispatched survey, so within one round the programme will know both
   the ceiling on the measured hour and its distribution across ~600 coin-hours.
   **The right response is to compute it, not to re-litigate ten days.**

**And the transferable rule, which is cheap and which I would adopt rather than
merely note:** *before optimising a lever, compute what the lever is worth under
perfect foresight.* It is almost always a filter and a sum over records that
already exist — it was here — and it is the one number that can retire a research
direction without a single model being fitted.
