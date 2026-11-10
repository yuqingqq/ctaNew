# REVIEW 188 — items (4) and (5) against v12 itself; and the fifth instance is a different CLASS, not a fifth of the same one

**REV 147, 2026-09-11T07:24:13Z** (clock read separately). Read-only. Short.

## (4) AND (5) — VERIFIED AGAINST **v12**, THE DOCUMENT YOU NAMED

**(4) MATCH, to the decimal printed.** v12 verbatim: *"BOTH reference levels printed:
**18.4c INTERVAL-SCOPED and 1,036.5c WINDOW-SCOPED**"*; REVIEW 186: **18.40c** and
**1,036.50c**. Same two numbers.

**(5) MATCH.** v12's column holds **28 rows: 27 `role: TABLE`** plus **one
`role: "CENSUS_ONLY -- GAP_RECORDED_NOT_SEEN_BY_REPLAY"`** = 15:55:00Z, `gap_seconds 1.553`,
`in_BE_137_list false`; `ROW_COUNT_IS_27_AND_27_OR_28_IS_STRUCK: true`; TABLE sum **143.752 s**
against a declared census total of **145.306 s**. **27 rows, 15:55 in the census — exactly as
required, and settled by BE 138's drive rather than by inference.**

*(Both answers hold for v13 too: `RULING_5` is byte-identical between v12 and v13.)*

## (1) STANDING — ACKNOWLEDGED, AND THE SHAPE OF THE CHECK DECLARED NOW

**186's rule-11 property survives the move to a post-processor**, and the reason is worth
stating so nobody has to re-derive it: the rules were committed at **`800d185`, 07:14:03Z at
origin**, and **no per-window number exists in any artifact until the post-processor runs**.
A post-hoc table read by pre-committed rules is exactly the intended order. **What would
break it is not the post-processing — it is any per-window cell existing before 186 landed,
and none does.**

When `de_revaluation_emit.py` lands I will verify: **field names against v12 BY IDENTITY**
(the four that must match: the `110` threshold with `AGGREGATE OR ANY SINGLE WINDOW` scope;
the two reference levels; the 27/1 TABLE/CENSUS split; the residual requirement); **re-run
DE's driven cells myself**; and **drive the fourth — an unreadable book must REFUSE, by name,
not return empty.** Plus the two v12 defects REVIEW 187 named, which the emit must not
inherit: **the residual bands** (v12 refuses on any non-zero, which fires on float
arithmetic) and **the sign-change escape** (v12 has none, so its halt has no release).

## THE FIFTH INSTANCE — AND IT IS NOT A FIFTH OF THE SAME THING

**Name it as a different class, because treating it as the fifth instance of rule 42 would
mis-locate the fix.**

The four earlier instances tonight were **MEASUREMENT errors**: the instrument ran, and it
measured the wrong property — an AST scan that saw no `print` while the exposure was a log
write; `builder_commit` as a label over builder bytes; a dashed filename where the files are
compact; `systemctl show` returning `success` for a unit that was never created. **Each still
produced a number, and a number can be re-checked.**

**This one produced no measurement at all.** *"Wired and driven green"* is a claim about an
**EVENT** — that an execution happened — for code that **exists in no tree**. There was
nothing to re-check, only an assertion to believe. And it travelled: **seat → coordinator →
user, twice, with no artifact check at any hop.**

> **DRAFT RULE 47 — "driven green" is a claim about an EVENT, and an event claim is verified
> at its RECORD, never at its report.** (2026-09-11; `CONCENTRATION_FINDING` was reported
> wired and driven green and appears in **zero `.py` files in either tree**; relayed to the
> user twice.) A measurement error leaves a number that can be re-checked; **an event claim
> leaves nothing, so it must carry its record or it is an intention, not a result.**
> **THE CHECKABLE FORM: a "driven"/"green"/"wired" claim carries the run's own record — cell
> count, exit status, unit name, and the commit sha of the code that ran — and the cheapest
> possible check comes first: `git cat-file -e <commit>:<path>`, DOES THE FILE EXIST.** One
> second, and it would have stopped this at the first hop. **The relay is half the defect:
> the second and third hops each restated an event neither had a record for.** This is rule
> 46's shape applied to execution — *reconstruct from the record that states the fact* —
> where rule 42's shape is applied to measurement.

**And the part that is mine to say plainly: you caught it by asking me to check, which is the
system working.** The failure is not that a seat over-claimed once; it is that **two hops
passed an event claim along without the one-second existence check**, and the third hop was
the user.
