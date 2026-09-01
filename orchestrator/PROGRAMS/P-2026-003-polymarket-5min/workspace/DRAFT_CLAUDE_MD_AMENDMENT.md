# DRAFT-FOR-USER — two amendments to the repo's `CLAUDE.md`

**Status: DRAFT. Nothing here is applied.** `CLAUDE.md` is USER-owned; a seat
drafts the exact text and **the USER applies it by their own hand or on their
explicit apply instruction — never a seat unprompted** (R-274). Drafted by the
MEM seat, 2026-09-01T13:58Z, against `CLAUDE.md` as it stands at commit
`c31bb8f`.

Two defects, both long-standing, both flagged on the register and neither
fixable by any seat. Each hunk below carries **current text verbatim**,
**replacement text verbatim**, and the **register citation** behind it.

---

## Hunk A — the state-file ownership collision

### Why

`SEAT_PROTOCOL.md` rule 6 has carried this caveat since 2026-08-28:

> **State-file ownership**: MEM writes STATUS/HANDOFF; BE/DA commit artifacts
> and file facts (R-233). CAVEAT: CLAUDE.md still instructs every session to
> update these files — a USER-ONLY amendment is pending; until it lands, a
> fresh seat following CLAUDE.md is behaving correctly and MEM sequences
> around it.

**The USER already ruled the substance.** R-274 (2026-08-28T14:09Z), answering
Q-DA-138 through the ESCALATION-FOR-USER channel, chose option (a): *"the
multi-seat program follows SEAT_PROTOCOL.md; fresh readers land on the register
first; MEM's exclusive state-file ownership stands; CLAUDE.md's general rule
stays for single-seat programs. The coordinator DRAFTS the exact text; the edit
is applied by the user's hand."* That text was drafted then and never landed;
this is it again, current.

**The live cost of not landing it** is not theoretical: a fresh seat obeying
`CLAUDE.md` writes the same two files MEM owns, and two writers on one state
file is a collision, not redundancy. The caveat's own wording concedes the
fresh seat is *behaving correctly* — which is exactly why the instruction, not
the seat, has to change.

### Current text (verbatim, `CLAUDE.md`, section "Active program tracking (code-relay protocol)")

```markdown
## Active program tracking (code-relay protocol)

Active research programs live in `orchestrator/PROGRAMS/P-*/` (currently
P-2026-002 HF market making and P-2026-003 Polymarket 5-min). **At session
start read each active program's `workspace/HANDOFF.md`**, and **after each
completed step update that program's `STATUS.yml` (task statuses + flags) and
`workspace/HANDOFF.md` (done / in-progress / next / watch-out-for)**. State
lives there, not in conversation history — write it before context runs long,
not after.
```

### Replacement text (verbatim)

```markdown
## Active program tracking (code-relay protocol)

Active research programs live in `orchestrator/PROGRAMS/P-*/` (currently
P-2026-002 HF market making and P-2026-003 Polymarket 5-min). **At session
start read each active program's `workspace/HANDOFF.md`**, and **after each
completed step update that program's `STATUS.yml` (task statuses + flags) and
`workspace/HANDOFF.md` (done / in-progress / next / watch-out-for)**. State
lives there, not in conversation history — write it before context runs long,
not after.

**Exception — P-2026-003 is multi-seat and its state files have ONE writer.**
Read `orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace/SEAT_PROTOCOL.md`
before writing anything there, and land on that program's register
(`workspace/COORDINATION.md`) first; on any conflict, the register is the
authority. In that program `STATUS.yml` and `workspace/HANDOFF.md` are written
by the **MEM seat only**. Every other seat files its facts into
`COORDINATION.md` — R-entries and the Q-filing table — and lets MEM sweep them
into the state files. Two seats editing one state file is a collision, not
redundancy. The paragraph above stays in force for single-seat programs.
```

### Citations

| claim | where it is established |
|---|---|
| USER ruled option (a); text applied by the user's hand | **R-274**, 2026-08-28T14:09Z (answering Q-DA-138) |
| MEM's exclusive state-file ownership | **R-233**, 2026-08-28T06:24Z (MEM seat operational) |
| the caveat this closes | `SEAT_PROTOCOL.md` rule 6, standing since 2026-08-28 |
| register is the authority on conflict | `SEAT_PROTOCOL.md` header, "On any conflict, the register wins" |

---

## Hunk B — rule 9's false settlement parenthetical

### Why

Rule 9's parenthetical asserts a settlement source that **does not exist in the
data**. It was flagged FALSE on the register within the hour it was verified,
and flagged specifically as a USER-owned file a seat may not edit.

**R-253** (2026-08-28T10:20Z): *"'PM binaries settle on a Binance-derived
price' is FALSE (17,727/17,727 records name Chainlink; Binance appears in
ZERO) … CLAUDE.md's rule-9 parenthetical carries the same false claim —
flagged to the USER (their file)."* DA filed the same correction against its
own earlier rows in-band as **Q-DA-117**.

**Re-verified today at the artifact, not carried from the entry** (2026-09-01,
`data/pm_5min/markets.jsonl`): **26,099 records; 26,099 name Chainlink; 0 name
Binance.** The population has grown 47% since R-253 and the ratio is unchanged
— numbers age, ratios carry. The check asserts it read a non-empty population,
so the zero cannot be a vacuous parse.

**Rule 9 itself is untouched, and it still binds here — through a different
door.** R-253's third consequence: the tautology risk was never Binance; it is
that **`Identity` (the PM book) already prices this event**. So "report skill
incremental to the input" survives with its example corrected.

### The trap in this hunk: do NOT apply R-253's own suggested wording

R-253 suggested replacing the parenthetical with *"(PM binaries settle on
Chainlink TWAP-vs-open; Identity — the PM book — already prices the event)"*.
**That suggestion has itself been superseded** and must not be pasted in.

Later the same day **Q-DA-142 (amendment A2)** corrected the *statistic*, and
**Q-DA-146** confirmed it on a fresh, larger, non-overlapping population:

| reading | agreement with settlement | verdict |
|---|---|---|
| `S60(T)` vs `S60(t0)` — 60-second endpoints | 99.8% (n=1,465) / **99.85%** (n=8,022, 08-24..27) | **passes** the pre-registered ≥99.0% / ≥99.5% gate |
| `meanS60[t0,T]` vs `S60(t0)` — full-window mean | 86.9% / 85.2% | **refuted** |

Verified today at the artifact the filing names,
`live/pm_research/EXP_RESULTS_2026-08-20.md:10-17`: *"the averaging window is
w = 60 s, not the full 300 s range — the full-range reading scores 86.9% and
is refuted."*

**And the venue's own prose says the opposite**, in all 26,099 records read:
*"if the time-weighted average price (TWAP) of Bitcoin, generated by Chainlink,
of the time range specified in the title is greater than or equal to the price
at the beginning of that range."*

So the description says full-range TWAP and the repo's own passed
reconstruction says 60-second endpoints. **That tension is STATED, NOT
RESOLVED.** Writing either form into `CLAUDE.md` would install a contested
claim in the file whose whole purpose is rules that hold — which is how the
Binance claim got there in the first place. **This amendment therefore fixes
the venue and says nothing about the statistic.**

### Current text (verbatim, `CLAUDE.md`, reliability rule 9)

```markdown
9. **A baseline must remove the tautology.** If the target is derived from an
   input (PM binaries settle on a Binance-derived price), report skill only
   incremental to that input; skill vs base rate is meaningless.
```

### Replacement text (verbatim)

```markdown
9. **A baseline must remove the tautology.** If the target is derived from an
   input, report skill only incremental to that input; skill vs base rate is
   meaningless. (P-2026-003's PM binaries settle on **Chainlink**, never
   Binance — verified in `data/pm_5min/markets.jsonl`. The exact settlement
   statistic is contested and no form is asserted here; see R-253 and
   Q-DA-142/146. Rule 9 binds that program through a different door: the PM
   book — `Identity` — already prices the event, so skill there is reported
   incremental to `Identity`, not to a base rate.)
```

### Citations

| claim | where it is established |
|---|---|
| the parenthetical is FALSE; flagged as a USER file | **R-253**, 2026-08-28T10:20Z |
| DA's in-band correction of its own rows | **Q-DA-117** (register Q-table) |
| the statistic is 60-second endpoints; full-window refuted | **Q-DA-142** (amendment A2), confirmed **Q-DA-146** |
| the artifact behind both | `EXP_RESULTS_2026-08-20.md:10-17` (EXP-M6, n=1,465, pre-registered gate) |
| rule 9 survives through `Identity` | **R-253**, third consequence |
| population, as-of 2026-09-01 | `data/pm_5min/markets.jsonl`: 26,099 records, 26,099 Chainlink, 0 Binance |

---

## What was verified, and how

Everything above was read at its artifact; nothing was carried from a dispatch
or from memory (CLAUDE.md rule 16).

- `CLAUDE.md` — both target passages read verbatim from the file at `c31bb8f`.
- `COORDINATION.md` — R-274, R-253, Q-DA-117, Q-DA-142, Q-DA-146 read in place.
- `markets.jsonl` — counted, with an assertion that the read was non-empty, so
  "0 Binance" cannot be a vacuous parse (R-289's empty-set trap).
- `EXP_RESULTS_2026-08-20.md` — the table and its own refutation sentence read
  at the cited lines.
- The venue prose — matched in all 26,099 records, with the match count printed
  rather than assumed.
- **Both "current text" blocks were checked to anchor EXACTLY in `CLAUDE.md`**,
  so neither hunk can fail to apply against a passage that has drifted. The
  check ships both directions: a deliberately corrupted copy of hunk B's block
  is NOT found (so a match is not trivially true), and both replacement blocks
  are confirmed ABSENT from `CLAUDE.md` — which is what makes this a draft
  rather than a claim that something already landed.

## On applying it

Both hunks are independent; either can land alone. When Hunk A lands,
`SEAT_PROTOCOL.md` rule 6's caveat should lose its "a USER-ONLY amendment is
pending" clause and cite the amended `CLAUDE.md` instead — that edit is the
coordinator's, and it must not be made before the amendment is actually in the
file, or the protocol will describe a `CLAUDE.md` that does not exist.
