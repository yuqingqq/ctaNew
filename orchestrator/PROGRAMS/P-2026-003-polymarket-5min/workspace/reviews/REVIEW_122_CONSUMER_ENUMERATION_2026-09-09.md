# REVIEW 122 — is the consumer enumeration complete? No. It was a sample; here is the constructive one

**REV, 2026-09-09T08:39Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled.

**THE ANSWER: MY REVIEW 120 ENUMERATION WAS A SAMPLE, NOT AN EXHAUSTIVE SET. I searched one
spelling — `in gs` — and stopped at three. Enumerated constructively, from the operation
that constitutes the dependency, there are FIVE exposed consumers, and the two I missed are
WORSE than the three I found, because both compute EXCLUSION COUNTS.**

---

## 1. THE CONSTRUCTIVE METHOD, AS IN REVIEW 111

A consumer depends on the old shape when it performs an operation whose correctness assumes
**one key per generation**. That is four operation classes, not one:

1. a **lookup or membership test keyed on a generation's `t0`**;
2. **`len(assembly)` read as a generation count**;
3. **iterating the assembly** as if each entry were a generation;
4. **reading the key's third element** as a `t0`.

REVIEW 120 searched class 1 in a single spelling. **Classes 2–4, and the other spellings of
class 1, were never swept.**

## 2. THE RESULT — TEN MODULES HOLD AN ASSEMBLY AND KEY INTO IT; FIVE ARE EXPOSED

| module | t0-keyed sites | shape-aware | |
|---|---|---|---|
| `be_cancel_axis_null.py` | 3 | 6 | aware |
| `be_score_coverage.py` | 4 | 33 | aware — the module exists for this (BE 112) |
| `be_generation_count_derivation.py` | 1 | 4 | aware |
| `de_phase4_diag_runner.py` | 3 | 5 | aware |
| `be_daybook_build.py` | 1 | 10 | aware |
| **`da_elementwise.py`** | 1 | **0** | **EXPOSED** (REVIEW 120) |
| **`da_elem_grid.py`** | 1 | **0** | **EXPOSED** (REVIEW 120) |
| **`da_elementwise_hz.py`** | 1 | **0** | **EXPOSED** (REVIEW 120) |
| **`da_de53_exclusion.py`** | 2 | **0** | **EXPOSED — NEW** |
| **`de_section81_arms.py`** | 1 | **0** | **EXPOSED — NEW** |

## 3. THE TWO I MISSED ARE THE WORSE TWO

**`da_de53_exclusion.py:33`**

```python
key = (slug, side, float(g["t0"]))
(retained if key in gen_scores else excluded).append(rec)
```

and it then publishes `n_excluded`, `n_retained`, `n_reference`, **`excluded_fraction`** and
`reproduces_DE_counts`. **A generation whose `t0` key is absent is counted as EXCLUDED.**
This is a module whose entire purpose is auditing exclusions, and against a corrected
assembly it would report an **inflated exclusion fraction** — a wrong number in the exact
quantity it exists to report.

**`de_section81_arms.py:526`**

```python
if (s_, sd, float(g["t0"])) in gen_scores:
    rows.append({...})
else:
    dropped += 1
```

in the function that builds the arms' rows, under a docstring saying the exclusion "is
COUNTED (rule 4)". **A generation whose `t0` key is absent is dropped from `rows` and
counted as dropped** — so both the population and its exclusion count move together, which
is the shape that looks internally consistent and is wrong.

**Both are worse than the three DA elementwise modules**, which mis-count a population;
these two mis-count a population *and* report the miscount as an audited exclusion.

## 4. SCOPE — WHAT I CLOSED OVER, SO THE CLAIM IS CHECKABLE

- **Files:** every `.py` in `live/pm_research`, not a subset.
- **Operations:** all four classes above. Class 1 was swept for `float(g["t0"])` in any
  membership test, lookup or dict construction; classes 2–4 by their own patterns; and a
  final closure sweep for **any** membership or subscript into `gen_scores` / `gs` /
  `scored` regardless of how the key was spelled, which surfaced no further t0-keyed
  assembly consumer beyond the ten above.
- **What that does NOT close:** a consumer that assembles the key from variables in a form
  no pattern I used would match; a consumer that receives the assembly through a wrapper and
  keys into it under another name; and anything outside `live/pm_research`. **Those are the
  residuals, and I name them rather than claim a closure I did not test.**

The closure argument is the same as REVIEW 111's: **a consumer is exposed only if it
performs one of the four operations, so enumerating the operations enumerates the
consumers** — the set is closed over the operations I swept, not over all possible code.

## 5. WHY THE COUNT WENT FROM THREE TO FIVE, SAID PLAINLY

Not because the code changed. **Because REVIEW 120 searched for the pattern I had already
seen and stopped when it found instances of it.** That is the sampling failure this
programme keeps naming in instruments, committed by a reviewer: a search that confirms a
known shape is not an enumeration. The correction is the method — enumerate the OPERATION,
not the SPELLING — and it is the same correction REVIEW 111 made for pin sites.

## 6. ROUTED

1. **DA — `da_de53_exclusion.py:33` and the three elementwise modules**: sweep the
   membership test to the per-row key shape before any of them is run against a corrected
   book. `da_de53_exclusion` first, because its output is an exclusion fraction.
2. **DE — `de_section81_arms.py:526`**, same sweep; it feeds the arms' rows.
3. **Coordinator — the number REVIEW 121 asked the build to emit
   (`n_generations_with_a_key_at_their_own_t0` per head) prices ALL FIVE at once**, and is
   still one line.
4. **The set is closed at five over the scope in §4**, with the three residuals named there.
