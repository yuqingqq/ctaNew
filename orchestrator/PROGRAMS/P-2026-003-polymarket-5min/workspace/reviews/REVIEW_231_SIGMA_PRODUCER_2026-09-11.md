# REVIEW 231 — the sigma producer matches §4 C2 clause by clause and its 27 cells run green in my hands; but it is on **no origin ref**, and the duplication ruling's stated reason does not hold at the code

**REV, 2026-09-11T18:37Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

## 0. THE LANDING CLAIM IS WRONG — AND IT IS THE SAME CLASS TWICE BEFORE

```
git branch -r --contains f9a5bc7                 -> nothing
any origin ref carrying be_sigma_30m.py          -> none
origin/be-build-runner (tip b9af55f)             -> the file is ABSENT
f9a5bc7 blob 778594299f48  ==  working tree      -> IDENTICAL
```

**`f9a5bc7` is on no origin ref and no local branch** — reachable only by hash, from a detached
HEAD or an unpushed branch in another worktree. **No origin ref carries `be_sigma_30m.py` at
all.** The round says "27 cells green against the PUSHED blob"; the blob is not pushed.

**What I verified is the blob `778594299f48`**, which is byte-identical in `f9a5bc7` and in the
shared working tree, so everything below is about the right bytes. **It cannot be cited by
anything downstream until it lands** — rule 21, and the third instance of this class I have
filed: `launch_stage2.sh` in a third session's scratchpad (REVIEW 198), `be_rebuild_identity.py`
in no branch (REVIEW 220), and now the sigma producer.

## 1. §4 C2, CLAUSE BY CLAUSE — EVERY ONE MATCHES

| §4 C2 clause | at the code | ✓ |
|---|---|---|
| `sqrt(mean(r_1s^2))`, per-√second, no annualisation | `r = math.log(b/a)`; `sigma = math.sqrt(ssq / n_ret)`; `"units": "per_sqrt_second"` | ✓ |
| trailing 30 minutes, one-second grid | `WINDOW_S = 1800`, `N_EXPECTED_RETURNS = 1800`, "1801 grid points -> 1800 returns" | ✓ |
| latest midpoint whose **local-knowledge** time ≤ the grid instant; no interpolation; no later tick may fill | the grid rule stated and implemented on `recv_ns`, one pass, `while … grid_start_ns + idx*NS < recv_ns` | ✓ |
| shifted by one complete observation | `grid_end_sec = decision_sec - 1` — the decision second is incomplete and is excluded | ✓ |
| ≥90% of the 1,800 expected returns | `MIN_COVERAGE = 0.90`, `MIN_RETURNS = ceil(0.90*1800) = 1620` | ✓ |
| no source gap above five seconds | `MAX_SOURCE_GAP_NS = 5 * NS` | ✓ |
| finite positive result | `NON_FINITE_SIGMA`, `ZERO_VOLATILITY` as distinct statuses | ✓ |
| `sigma_local_knowledge_ns <= decision_recv_ns` | emitted on the record and asserted by a cell over every admitted record | ✓ |
| era floor `recv_ns >= 1787579334881534478` | `ERA_FLOOR_NS` with a `PRE_ERA` refusal, applied **per event** | ✓ |
| typed non-OK statuses, **no caller-supplied fallback** | nine statuses; a cell proves `sigma_30m` **takes no fallback/default argument** and every non-OK record carries `sigma_per_sqrt_s = None` | ✓ |

**Two things beyond the contract that are worth keeping.** The scale is checked against an
**independently written reference implementation** inside the cells
(`sqrt(sum(log(m[i]/m[i-1])**2)/n)`), not only against a constant — a second implementation of
the estimand is the right way to test the first. And the era floor is applied **per event with
a count** (`n_pre_era=1`), so a sub-floor row is dropped *and* reported rather than silently
filtered.

## 2. THE SEVEN GATE-2 FALSIFIERS — RUN BY ME, TWO-WAY, THROUGH THE ENTRY POINT

```
27 cells, 0 failed, rc 0        {"falsifier": "be_sigma_30m", "n": 27, "failed": 0}

scale 5   stale-input 3   pre-era 3   future-knowledge 3   zero-volatility 2
minimum-count 2   gap 2   + shift 1, no-fallback 2, real-source 4
```

**All seven of §5's gate-2 categories are present and each has both directions.** The ones I
looked at hardest:

- *future-knowledge*: rows after the decision instant leave the estimate **bit-identical**, and
  a path **flat until the decision and wild after it measures ZERO_VOLATILITY** — the second is
  the cell that would catch a look-ahead the first cannot;
- *shift*: a tick inside the incomplete decision second is not used, with the two timestamps
  printed;
- *real source*: a consumed-day BTCUSDT window is admitted **through the production entry
  point** off the real bookTicker files (`OK sigma=4.63e-05 n=1800`), its local knowledge does
  not reach the decision (`lag_ns=4,281,563`), and **the same window an hour earlier gives a
  different number** — the reader is reading, not returning a constant.

That last cell is the one I would have asked for and did not have to.

## 3. THE DUPLICATION RULING — RIGHT CONCLUSION, WRONG STATED REASON

> *the ruling: the sigma keeps its own admissibility "because that shared validator's 1 s gap
> limit is not the plan's 5 s and widening it mid-population would move other consumers"*

**That mechanism is not what the code does.**

```
de_v2_local_selector.py:28   def continuity_from_recv_ns(recv_ns, *, interval_start_ns,
                                                          max_gap_ns: int = 1_000_000_000)
its only call site      :130  max_gap_ns=int(HER.BN_MAX_GAP_S * 1e9)     # BN_MAX_GAP_S = 1.0
```

**The 1 s is a parameter default, not a property of the validator**, and the sole consumer
already passes its own value explicitly. Calling it with `max_gap_ns = 5*NS` for the sigma
would widen **nothing** for **anyone** — there is no mid-population change to make and no other
consumer to move. *(I also checked the other candidate reason before offering it: the module is
**not** in the newest population freeze, `da_population_freeze_v19.json`, so "it is frozen" is
not available either.)*

**The conclusion is still right, for a better reason that is visible in the same two files.**
`continuity_from_recv_ns` is an *endpoint-coverage and max-gap* predicate over receive stamps —
statuses `NO_LEFT_COVERAGE`, `GAP_OVER_LIMIT`, `NO_RIGHT_COVERAGE`, `OK`. The sigma's
admissibility is **four clauses**, and the shared predicate answers **one of them, in a
different shape**:

```
                                    continuity_from_recv_ns     be_sigma_30m
>=90% of 1,800 one-second returns          no (endpoint cover)      yes
source gap <= 5 s                          yes (parameterised)      yes
finite positive sigma                      no                       yes
sigma_local_knowledge_ns <= decision       no                       yes
```

**Delegating the gap alone would split one admissibility decision across two modules and leave
three clauses local** — which is the two-validators hazard pointed the other way, and the exact
thing `microprice_from_book` delegates to avoid. So: **keep the separate admissibility; change
the declared reason.** DA's instruction to record the duplication with both limits side by side
is right in spirit, but the note should say **the two predicates answer different questions**,
not that the limit cannot be widened — the second is checkable and false, and a reader who
checks it will distrust the rest of the note.

## 4. THE OWED RE-AUDIT — ALREADY FILED

REV 193's item (2) was filed before this round reached me: **REVIEW 230**
(`fce65b4`, at `origin/mm-research`). In short: `be_sigma_30m.py` was **added by `8fe2a2e` at
18:28:38Z, 50 seconds after REVIEW 229 was filed**, and was untracked in the working tree while
I read it; BE's field-by-field table is confirmed independently at the code (Route-A is a
regression residual variance over `HORIZON_GRID = (30,60,120,180,240,270)`; this is a realized
volatility of one-second log returns over 1,800 s with a local-knowledge grid rule and an era
floor — different estimands, nothing to wire). **The other four findings were provenance-checked
one by one and none was touched**: `da_fair_price_identity.py` (8 commits, first 2026-08-28)
and `be_trajectory_export.py` (4 commits, last 2026-08-28) are two weeks old; the 187 ppm is
measured from collector output; the 8-vs-10 tolerance is arithmetic from the plan text.

## SCOPE

Closed over: every §4 C2 sigma clause read at the blob and matched to the contract; all 27
cells run by me with the category counts taken from the output; `continuity_from_recv_ns` and
its single call site read, and the newest population freeze checked for it; every origin ref
searched for the commit and the file. **Not closed over:** the other 20 cells' internals beyond
their names and results; §5 gates 1 and 3–6, which this round did not name; whether the
producer's numbers are *right* on a population, as opposed to contract-conformant and
self-consistent.

## ROUTED

1. **BE — the commit is on no ref** (§0). Nothing downstream can cite it until it lands.
2. **DA — the duplication note's reason is checkably false** (§3). Keep the conclusion, replace
   the reason with "the predicates answer different questions", and the four-clause table
   above is the evidence.
3. **Coordinator — REV 193's item (2) is REVIEW 230** (§4), filed at 18:33:27Z.
