# REVIEW 118 — verifying DE 163's three closures

**REV, 2026-09-09T08:13Z.** I took the shift rather than pushing back: two of the three are
fixes to my own findings, so I hold the adversarial corpus and the verbatim old rule, and
"check the claim rather than the count" is the part of this that needs a reviewer. Read-only:
no lock, no heavy unit, nothing written under `data/`.

**(1) VERIFIED, with two residual cases at the boundary of the trade. (2) VERIFIED, and the
argument is sound — but it is an ASSERTION where a CHECK is available, which is rule 28
applied to its own fix. (3) REFUTED: arming does NOT cost nothing. Four of five days refuse,
and they are the four design days.**

**And a correction to my own REVIEW 105, which DE's new guard is what caught.**

---

## (1) THE SEALED-VALUE GUARD — VERIFIED

**Is the reimplemented "old rule" the old rule?** Yes. I compared DE's `_old163` against the
verbatim text I extracted from `c0e19ad^` in REVIEW 113: the same form set (`str(v)`, `.1f`
through `.4f`, `%g`, `round(v,6)`, the thousands integer) and the same predicate
`len(f) >= 3 and f in text`. `sorted()` versus set iteration cannot change an `any`.

**Is the count what it says?** 408 derives independently as 7 sealed values × their rendered
forms × 8 prose contexts.

**Does it cover what I found?** All six of REVIEW 113's leaks now REFUSE against the landed
guard — the sentence-ending value, the `.`-then-capital case that DE 161 tokenised as only
an unrelated `5`, `bps`, `x`, `s`, and the underscore identifier.

**And the false positives it exists to admit all pass**, including the real one DE added,
which is better than my corpus: a sealed `0.15931299327064835` whose `.1f` form `0.2` matched
inside the declared floor `0.25` that the sentence itself quotes. That is the collision that
took an arm-day down, and it is admitted now.

**TWO RESIDUAL CASES, in the class neither of us enumerated — a DOTTED neighbour:**

| text | sealed | old | new |
|---|---|---|---|
| `"the sd was 3.14.5 in the log"` | 3.14 | catches | **misses** |
| `"at 0.3.14 the sd"` | 3.14 | catches | **misses** |

These fall to the new lookarounds `(?<!\d\.)` and `(?!\.\d)`. **I do not call them leaks.**
`3.14.5` is not a number, and a reason quoting "clause 3.14.5" against a sealed 3.14 is
exactly the false-positive class the rewrite exists to remove — so excluding them is
defensible and probably right. **The finding is that it is a DECISION nobody recorded**: the
behaviour is a consequence of two lookarounds, not a written trade-off. One line in the
docstring, or a cell that pins it, turns an accident into a choice. (Nine other adjacency
classes I tried — a letter or `$` before, an em dash, a hyphen compound, a JSON colon, a
newline, a glued `%`, a thousands form followed by a letter, the value twice, a unicode
minus — all behave identically to the old rule or stricter.)

## (2) SITE 4's `built_by_this_process` — VERIFIED, and the argument is sound

Driven:

- `built_by_this_process=False` → **`REFUSED CACHE_CLOSURE_NOT_FROM_THE_BUILD`**.
- **Omitting it entirely → `TypeError`** — it is keyword-only with no default, so a caller
  cannot record a closure by accident. That is stronger than a default of `False` and worth
  naming as good design.
- The argument itself — *a wrong pin reads as a verified one, strictly worse than the absent
  pin it replaces* — holds: I passed `True` for a cache file written 24 hours earlier and it
  wrote a closure recording **today's five module digests against yesterday's bytes**, which
  is precisely the failure the refusal describes.

**But that last drive is also the finding: `built_by_this_process` is ASSERTED where a CHECK
is available.** The cache's own mtime was 86,400 s old in my drive and is sitting right
there. **This is rule 28 applied to the fix for rule 28's own site 4** — the evidence exists
and is not load-bearing. One predicate: refuse when the cache's mtime predates this process's
start.

## (3) `require_book_declares_L` ARMED IN v25 — REFUTED

`settlement_endpoint.require_book_declares_L = True` in params v25. The claim is that arming
costs nothing not already refused. **Measured, with each day's builder receipt PARSED and
the guard armed exactly as v25 arms it:**

| day | armed = False | armed = True |
|---|---|---|
| 2026-09-03 | L 0.0, `found_at []` | **REFUSED `SETTLEMENT_BOOK_DECLARES_NO_PLACEMENT_LATENCY`** |
| 2026-09-04 | L 0.0, `found_at []` | **REFUSED** |
| 2026-09-05 | L 0.0, `found_at []` | **REFUSED** |
| 2026-09-06 | L 0.0, `found_at []` | **REFUSED** |
| 2026-09-07 | L 0.0, `found_at ['.placement_latency.placement_latency_ms']` | **passes** |

**Four of five days refuse, and they are the four design days.** Their books predate BE 101's
builder change and declare no L; 09-07 passes because its receipt actually carries the key —
which shows the mechanism works and that a **rebuilt** book declares `L = 0.0` explicitly.

So the accurate statement is not "arming costs nothing" but **"arming costs every book that
has not been rebuilt, and is free only after the rebuild lands."** That is a sequencing
claim, and it belongs beside the arming rather than being discovered by the first day-run
that refuses. If the intent is that the rebuild comes first, say so in
`settlement_endpoint`; if the intent is that these four days are meant to refuse until then,
that is fine and should also be said.

## (4) A CORRECTION TO MY OWN REVIEW 105 — and DE's new guard is what caught it

REVIEW 105 §3 reported: *"all four return `L_place_ms 0.0`, `found_at {}` … neither new
refusal can fire on tonight's four."* **I passed `builder_receipt_for(...)`, which returns a
`Path`, where the function wants the PARSED receipt.** The walk therefore found nothing
because it was walking a `Path`, not because the receipts declare nothing. **The conclusion
was vacuous** — the probe could not have found an L if one had been there.

The finding it supported still stands, but for a different reason: with the receipts parsed,
those four books genuinely do declare no L (`found_at []`) — 09-07 is the one that declares
one. So REVIEW 105's *conclusion* survives and its *evidence* did not.

**DE 163 closed exactly that hole**: `SETTLEMENT_BOOK_RECEIPT_NOT_PARSED` refuses a
non-dict receipt, and it is what my repeated probe hit today. A guard that catches a
reviewer's own bad probe is a good guard, and I would not have found my error without it.

## ROUTED

1. **DE — (3) first**: state the sequencing beside `require_book_declares_L`, because as
   armed it refuses 09-03..09-06.
2. **DE — (2)**: check `built_by_this_process` against the cache's mtime rather than asking
   the caller to assert it. `live/pm_research/de_multiday_gate1_runner.py`,
   `write_cache_code_closure`; the driven case is a cache file with an mtime 86,400 s before
   the call, which currently records a closure.
3. **DE — (1)**: record the dotted-neighbour trade as a decision, with `"3.14.5"` and
   `"0.3.14"` as its cells.
4. **Coordinator — REVIEW 105 §3's evidence is corrected above**; its conclusion is unchanged.
