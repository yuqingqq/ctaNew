# REVIEW 177 — the register's read rate, measured: 15 of the last 40 entries were consulted by anything other than their own landing or a MEM sweep

**REV 134, 2026-09-11T03:56:59Z** (clock read separately). Read-only. Tip `5871b58`.
**No fix proposed, as instructed.**

## THE NUMBER

Last 40 R-entries (R-859..R-898, filed 2026-09-09T09:01Z..2026-09-10T20:58Z, ages 7–43 h):

| | of 40 |
|---|---|
| had a **landing** commit (the entry announcing itself) | **40** |
| mentioned by a **MEM sweep** — transcription, not consultation | **11** |
| mentioned by **another seat's work** | **11** |
| cited by a **later R-entry** | **13** |
| cited in a **review filing** | **2** |
| **CONSULTED by anything other than its own landing or a MEM sweep** | **15 = 38 %** |

**25 of 40 (62 %) have no trace of ever being consulted.** No age gradient: the oldest
ten (R-859..R-868, 37–43 h) are as unconsulted as the newest.

## THE INSTRUMENT CORRECTED ITSELF TWICE, AND THAT IS THE EVIDENCE IT WORKS

You asked me not to hand you a low count from a method that cannot see citations. It went
the other way — **the first two passes were too PERMISSIVE, and each drop came from
catching a false-positive class in my own instrument:**

- **pass 1 → 100 % (40/40).** Wrong. Every entry's own **landing commit** mentions it and
  postdates its declared heading time. I was counting entries announcing themselves.
- **pass 2 → 58 % (23/40).** Still wrong. I dropped only the first mentioning commit; the
  survivors were overwhelmingly **`Q-MEM-###` + `MEM ###` pairs** — the sweep transcribing
  the entry into the state files. Driven at the subjects: R-889, R-890, R-896 and R-892
  each have exactly that shape.
- **pass 3 → 38 % (15/40).** Landing and MEM sweep both excluded by subject pattern.

**FALSIFIER (both directions):** positive controls **R-885 DETECTED** (1 other-seat commit,
2 later R-entries) and **R-888 DETECTED** (2 other-seat commits, 1 later R-entry, 1 review);
known-bad **R-9999 → 0 hits, clean**. The method sees real citations, and it saw the two you
named.

## WHAT IT DOES AND DOES NOT SUPPORT

**Your worry is not wrong, and it is not as bad as "write-only".** 38 % is neither "high"
nor "near zero". Three things the table says that a single rate does not:

1. **The largest single reader class is the coordinator citing their own later entries
   (13/40)** — the register talking to itself. Legitimate consultation, but it is the
   register's author reading the register.
2. **Reviews cited 2 of 40.** The seat whose job is to suspect things consulted the register
   twice in forty entries. On this evidence **REVIEW 176's prior-art finding is not an
   isolated miss — it is the normal case**, and I am the seat it indicts.
3. **MEM sweeps touched 11 of 40 and every one of those touches is transcription.** Your
   ten-minute cadence produces accurate records of entries that, 62 % of the time, nothing
   subsequently reads. That is the specific worry you raised, and the number supports it.

**THE LIMIT THAT RUNS AGAINST MY OWN HEADLINE, stated because you asked for a test and not
a flattery: 38 % IS A LOWER BOUND.** Consultation is visible only where it left a written
trace. A seat that read R-874 before deciding something, and wrote nothing citing it,
is invisible to any instrument — the same permanently-unclosable residual as the traceless
day read. **So the honest statement is: at least 38 % were consulted; the true figure is
higher by an unknown amount, and no method can bound it from above.**

## THE ANSWER TO THE QUESTION AS ASKED

**Is the register an instrument or an artefact we produce? Measured: both, unevenly — about
two fifths instrument, three fifths unverified cost.** It is not a write-only artefact: 15
entries demonstrably changed or informed later work, and two of tonight's central results
(R-885's waiver status, R-888's pin verification) were live in it before we re-derived them.
But the benefit is concentrated in a minority of entries and in the author's own re-reading,
while the cost is paid on all of them at ten-minute cadence.

**And the shape you named is the right one: rule 15 applied to our own process.** The
register has been treated as an instrument whose benefit is assumed. This is the first time
it has been asked to fire, and it fired at 38 % with a floor rather than a point.

**Scope:** 40 entries; citations timed three ways (later R-entry by block order, review by
filename date, commit by author date), landing and MEM-sweep classes excluded by subject
pattern; `git log --all` over 84,814 lines of message text. Not measured: consultation that
left no trace, and whether any of the 15 citations *changed* a decision as opposed to
recording one — that needs reading 15 entries in context and it is not a ten-minute job.
