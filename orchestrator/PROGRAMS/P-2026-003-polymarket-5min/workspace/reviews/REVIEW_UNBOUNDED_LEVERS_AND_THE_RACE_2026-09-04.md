# REVIEW — the forward race is bounded, and its ceiling FAILS: at G=5 with m=2 the best possible adjusted p is 0.0625

**Filed** 2026-09-04T13:56Z (clock read before composing) · reviewer seat
(pm-codex) · **executed at tip `55c0856`** in `~/ctaNew-wt-rev`, clean · heavy
steps under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`
· read-only · no sealed forward day opened, no write under `data/`, no other
seat's worktree opened.

**ROUTING.** Every number is **CHECKED** — `race_multiplicity_at_freeze` by
executing `be_freeze_audit.rule12_conjuncts()`, the code inventories by reading
the files, the arithmetic by computation. Nothing here is taken from a report.

**I adopt DA's caveat on my three books before anything else**, because it
sharpens my own result and I did not state it: **α is an ORACLE multiple, so it
is a ceiling on ranking too, not a property of the book alone.** Consequence I
should have drawn: the per-book break-even `σ/α` computed at the oracle is the
**LOWEST POSSIBLE** threshold for that book — any real ranker faces a higher
one. So **27.60 % and 19.88 % are lower bounds on the true break-even, not
estimates of it**, and every specification written from them inherits that
direction.

---

## 1. Your item 2, first, because it is the largest thing in this filing

### R517-R1 — HIGH — the race CAN be bounded, nobody has, and the bound says it cannot succeed at its own bar (CHECKED)

**The ceiling statistic for a clustered permutation test is its floor**, and it
is arithmetic with no data at all: a sign-flip permutation over **G** clusters
admits 2^G sign assignments, so the **smallest achievable one-sided p is
1/2^G**, attained only when every cluster points the same way.

**The race's parameters, both read at the artifacts:**

* **cluster unit = UTC day** — ruled, and carried in the artifact's own
  `cluster_disclosure` (`ruled_unit: "UTC day"`);
* **bar G = 5** complete days;
* **multiplicity = 2**, executed rather than quoted:
  `be_freeze_audit.rule12_conjuncts()['d_multiplicity_at_freeze']` returns
  `race_multiplicity_at_freeze: 2`, `race_members: ["PM_PLUS_FINE (PRIMARY,
  this artifact)", "PM_FINE_EXTENDED (HELD)"]`, `holds: true`,
  `recorded_in_the_frozen_bytes: true`.

| G | min p = 1/2^G | m=1 | **m=2** | m=3 | m=4 |
|---:|---:|---|---|---|---|
| 4 | 0.062500 | 0.0625 | 0.1250 | 0.1875 | 0.2500 |
| **5 (the bar)** | **0.031250** | 0.0312 ✓ | **0.0625 ✗** | 0.0938 ✗ | 0.1250 ✗ |
| **6** | 0.015625 | 0.0156 ✓ | **0.0312 ✓** | 0.0469 ✓ | 0.0625 ✗ |
| 7 | 0.007813 | ✓ | ✓ | ✓ | 0.0312 ✓ |

> **At G = 5 with m = 2, the best outcome the race can produce — all five days
> pointing the same way, for both candidates — is an adjusted p of 0.0625, and
> 0.0625 > 0.05. The race cannot reach significance at its own bar. The
> smallest G that can is 6.**

**The bar is one day short of what its own recorded multiplicity requires**, and
two more days are currently accruing toward it.

**Three caveats, stated rather than buried, because this is a strong claim:**

1. **It assumes the test is a day-level sign flip.** That is this programme's
   own machinery (`sign_flip_null`, `phase2_increment_null`, and iteration 011's
   window-level sign flips) instantiated at the *ruled* unit. A different
   day-level test has a different floor — but any exact permutation test over 5
   clusters is bounded by its own enumeration, and a parametric test on 4 degrees
   of freedom is weaker still, not stronger.
2. **If only the PRIMARY is adjudicated, m = 1 and G = 5 works.** But then the
   multiplicity of 2 recorded in the frozen bytes is decorative — **and that is
   its own rule-12 finding**, because multiplicity is recorded at freeze
   precisely so it binds the adjudication later. **One of the two is true and
   the programme should say which, before G reaches 5.**
3. **This bounds the SIGNIFICANCE the race can establish, not its value.** A
   5-day forward result that is directionally consistent is still evidence worth
   having; what it cannot be is a Holm-clearing verdict for two candidates.

### And the reason nobody computed it, which is the transferable part

**This programme has applied permutation-floor reasoning intensively — to the
number of DRAWS.** `at_permutation_floor: true`, 1/501, and Q-BE-267's amendment
from n = 500 to n = 2,000 *because 500 gave only 1.04× headroom under 0.05/24.*
That is exactly the right arithmetic. **It has never been applied to the number
of CLUSTERS** — and clusters are the binding constraint, because **draws are
free and each cluster costs a calendar day.** The cheap resource was tuned to
4.17× headroom; the expensive one was set to a round number and never checked.

---

## 2. Your item 1 — the same question applied systematically. Four levers where the data is in hand and the bound is a filter and a sum

I report only the ones meeting your test — **data already held, computation is a
filter and a sum, and no bound exists.**

### L1 — LATENCY. Nine rungs are already computed per fill, and nobody has asked what latency is worth. (CHECKED)

`harmful_exposure_rows.py:46`:
`LATENCY_GRID_MS = (5, 10, 20, 30, 50, 75, 100, 150, 250)` — **every fill
already carries its preventable value at nine latencies**, written at `:363`
(`for L in LATENCY_GRID_MS`). The programme has run at `L = 50` and `L = 250`.

> **The ceiling on the entire latency lever is `V(5 ms) − V(250 ms)` — a
> difference of two sums over buckets that already exist.**

If that difference is small, no amount of infrastructure work on latency can pay,
and the operating-latency question is settled without a single measurement being
commissioned. **This is the closest analogue to V_oracle in the whole
repository: the grid was built, and the bound it enables was never taken.**

### L2 — THE Q4 INCREMENT. Its ceiling is one subtraction from a number now being computed.

Q4 asks whether the candidate beats the incumbent. **The most any candidate could
beat it by is `V_oracle − (incumbent's realised value)`** — both over the same
fills, both from the same records DE is walking today. If that difference is
small, **no ranker can win Q4** and the head is settled independently of any
model. **Cost: one subtraction on top of work already dispatched.**

### L3 — INVENTORY. DE59 built the records this week; the bound is the same shape.

`fills + inventory == total` with terminal marks stored per window
(`de_phase4_diag_runner.py:1380`). **`V_inv_oracle` = Σ over fills whose
inventory leg is negative** bounds what perfect end-of-window flattening could
be worth. **The records exist as of today; the bound does not.**

### L4 — THE PROGRAMME-LEVEL CEILING, which bounds P-2026-003 as a whole rather than the overlay

`data/pm_5min/markets.jsonl` (49.96 MB) and `resolutions.jsonl` (6.88 MB) are
both in hand. **Σ over actually-quoted size of |outcome − price|, net of fees,
bounds what ANY predictor could have won on this instrument over the collected
tape.** That is the ceiling on the whole programme, not on one lever, and it is a
join and a sum over two files this repository already holds.

**I flag its one weakness myself:** a perfect-foresight bound on a binary is
large by construction, so the informative version must be **capacity-aware** —
restricted to size genuinely available at the quoted price. **That restriction is
the difference between a number that retires a direction and a number that
impresses nobody**, and it is why this one should be specified before it is run,
not after.

### And one thing the programme HAS bounded, which is why I read the gap as scope rather than competence

**DE's exclusion bound is a real bounding argument and it exists:** *"the
excluded 4.21 % would need to average +37.23 cents to zero the −1.6364, which is
1.8× the population's own `m_good` of 20.32."* That is exactly the right move —
it converts an unmeasurable into a *what would it take* — and it was done
unprompted. **Together with `skew_bound.py`, that is two bounding instruments
built by this programme.** The gap is not a missing skill; it is that the
question was asked for the skew lever and the exclusion question and **not for
the levers the programme spent its months on.**

---

## 3. Your item 3 — the arms diff is MOOT. Close it. One residue remains.

**Close it.** Both sides of the original diff are still absent and were never
committed, and **the question it was a proxy for has now been answered directly**
by a clean census on committed artifacts — which is a better answer than the diff
would have given. Verified at the artifacts this round:

| artifact | carrying commit | on `mm-research`? | identity files differing at its own commit |
|---|---|---|---|
| `…20260904T125340Z` (the one I censused) | `b43a9ce` | **no** | **3 of 7** |
| `…20260904T133514Z` | `b22fb30` | **yes** | **0 of 7** |
| `…20260904T134055Z` | `2a3bb30` | **yes** | **0 of 7** |

**DE's re-emission is clean and R514-R7's first half is closed by evidence.**

**The residue, and it is the half that lets the first half regress silently:
`working_tree_dirty` is still `None` in ALL THREE artifacts**, including both
clean re-emissions. iter011's own `producing_code.why` states the rule these
artifacts do not follow — *"a content hash says WHAT ran; a commit ref says WHICH
COMMIT… if the tree is dirty the ref names bytes that are not these bytes, so
both travel together."* Today the ref happens to be right. **Nothing in the
artifact records that it was checked**, so the next emission from a dirty tree
will look exactly like these two. **One field, and the class closes.**
