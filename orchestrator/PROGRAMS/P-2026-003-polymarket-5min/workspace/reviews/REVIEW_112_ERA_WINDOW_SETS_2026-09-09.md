# REVIEW 112 — the builder's era against the day verdict's

**REV, 2026-09-09T07:30Z.** Read-only: no lock taken, no heavy unit, nothing written under
`data/`. Driven in a scratch clone at the tip.

## THE ANSWER, IN THE TERMS THE ROUND ASKED FOR

**IT IS A NAMING DEFECT AT THE LABEL WITH A POPULATION CONSEQUENCE INSIDE EVERY WINDOW —
and the population it moves is not the window set, it is the GAPS.**

- **The window SETS agree**, and they cannot do otherwise: `day_selector._sel` returns
  **every** slug in `want` (`for s in sorted(want)`), so the era label filters no window.
  And **DA's side produces no window set at all** — `day_era_admission` is a DAY-level
  predicate ("is this UTC day ENTIRELY inside ONE ADMISSIBLE effective era"). So there is
  no window-for-window disagreement to find, and I say that rather than force the
  comparison into a shape neither side has.
- **But the label is not cosmetic.** It is the only argument to `flow_intensity.gaps_by_slug(era)`,
  and **the two eras' gap tables are DISJOINT**: `clob_v3_1` carries 1,143 slugs, `clob_v4_1`
  carries 727, **0 in common**. Naming the wrong era therefore hands `build_reference`
  `gaps=[]` for **every window of every September day**.

## THE MECHANISM — the builder never asks about the day

```
be_daybook_build.day_selector:  era = HER._era_or_refuse(fi, None, "be_daybook_build")
                                                          ^^^^ era=None
```

`_era_or_refuse(fi, era, what)` takes the era as its second argument and the builder passes
**`None`**, so it resolves a module default. Driven: it returns **`clob_v3_1`** with no day
in the call, and `day_selector` returns `clob_v3_1` for 09-01, 09-02 and 09-06 alike —
day-independent by construction. DA, meanwhile, reads the collector ledger's era timeline
and finds every September day **entirely inside `clob_v4_1`**, `era_pure: True`,
`era_admissible_ruled: {'clob_v4_1': True}`, `boundaries_inside_day: []`.

**DA is right and the builder is wrong**, and the disagreement is on **every day**, not only
09-07.

## WHAT IT COSTS, PER DAY, AND THE WORST DAY IS THE ONE ABOUT TO BE BUILT

Gaps carried by each day's btc windows under DA's era, against what the builder attaches:

| day | windows | gaps under `clob_v4_1` (DA) | seconds | gaps under `clob_v3_1` (builder) |
|---|---|---|---|---|
| **2026-09-03** | 287 | **160 windows** | **2,294.7 s** | **0** |
| 2026-09-04 | 288 | 52 | 438.9 | 0 |
| 2026-09-05 | 288 | 13 | 68.4 | 0 |
| 2026-09-06 | 288 | 14 | 49.1 | 0 |
| 2026-09-07 | 288 | 27 | 136.1 | 0 |

**On 09-03 — the day BE was about to build — 160 of 287 windows (56 %) carry real tape gaps
totalling 38 minutes, and the builder hands every one of them `gaps=[]`.** Named, for 09-06,
the fourteen: `btc-updown-5m-1788653100`, `…668100`, `…669900`, `…678300`, `…686400`,
`…698100`, `…711300`, `…721200` (an 11.1 s gap), `…727500` (10.9 s), `…730800`, `…734700`,
`…735300`, `…738000` (8.7 s), `…738300`.

**WHICH WAY THE DIFFERENCE RUNS: the unsafe way.** The builder under-reports gaps — zero
where the truth is 160 windows — so windows with real missing tape are built **as if
continuous**. The book is assembled over MORE content than is admissible, not less. That is
CLAUDE.md rule 5 exactly: era purity is a per-event predicate, and here it passes at the day
level (DA: pure, admissible) while being applied at the event level from the wrong era's
table.

**And it is not repaired by anything downstream.** DE's closure predicate, BE's coverage fix
and the key collision all correct how a book is built or read; this one decides what is
**inside** each window of it.

## HOW I KNOW THE COMPARISON COULD HAVE SHOWN A DIFFERENCE (rule 15)

It *did* show one, which settles it — but the discrimination is worth stating because a set
comparison that cannot fail is not a check:

- **The same comparison, on the same 288-window set for 09-06, returns 0 under one label and
  14 under the other.** One input, two labels, two answers.
- **The two tables share zero slugs** (1,143 and 727, intersection 0), so if the builder had
  named the right era the difference would have been visible in the first line of output;
  and the same probe applied to a day where both tables were empty would have returned
  0 and 0 and told me nothing — which is why I ran it against the day's real slug set and
  reported the seconds, not just the counts.
- **Cross-check on the slug set itself:** for 09-06 I derived the windows two independent
  ways — `be_daybook_build.day_slugs` (288) and the raw tape's filenames (288) — and both
  give the same 14 gapped windows. So the count is not an artifact of how I named the set.

## SCOPE AND WHAT I COULD NOT RESOLVE

`AW.supply` refuses days at or after the governed day (2026-09-02) without the blackout mask
artifact, and my scratch data root has none, so `day_slugs` resolved only 09-01 (265),
09-02 (248) and 09-06 (288). The masks exist under the real root
(`da_blackout_mask_2026090{1,2,3,4}.json`), so this is my environment, not a defect. **It
does not weaken the finding**: the era label is day-independent by construction — the call
passes `None` — so the three days that resolved establish it for all, and the coordinator's
own numbers (09-03 → 247 windows, 09-07 → 287, both `clob_v3_1`) are the same pattern from
the other side. For the per-day exposure table I took each day's windows from the raw tape,
which for 09-06 reproduces `day_slugs` exactly.

## ROUTED

1. **BE — `day_selector` must pass the DAY to `_era_or_refuse`**, not `None`, and refuse
   rather than default when the day's era cannot be resolved. One argument.
2. **BE — do not build 09-03 until it does.** The book would carry 160 windows' worth of
   missing tape as continuous, and no downstream predicate can put it back.
3. **BE/DE — after the fix, the two must be compared as sets, not labels**, at the build:
   assert the era `day_selector` resolves equals the era `day_era_admission` names for that
   day, and refuse by name on disagreement.
4. **Coordinator — the label disagreement is on EVERY day held, not only 09-07.**
