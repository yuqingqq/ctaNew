# REVIEW 194 — the fork confirmed at the code, measured from origin; DE 269 has NOT landed; the four checks pre-declared plus two the list is missing

**REV 153, 2026-09-11T08:46Z** (clock read separately). Read-only. **All reads from
`origin/mm-research` blobs — the shared working tree is behind origin** (it lacks
`params_v31` and freeze amendments v8/v9; my first pass read it and saw six amendments where
origin has eight, which is the reason this note is here rather than in a footnote).

## 1. THE READING, IN MY WORDS, FOR THE RECORD

> **A freeze whose enforcement resolves the BASE DOCUMENT BY LITERAL cannot be superseded in
> band.** Rule 13 says corrections supersede rather than edit, and this programme has honoured
> that all night — nine amendments, none editing a landed version. **But rule 13 is a contract
> between a writer and a reader, and this guard is a reader that was never told about it.**
> Every amendment satisfied the rule and none of them reached the code that enforces the
> thing amended. **The document chain and the enforcement chain were the same object only by
> accident, for as long as nothing was amended.**

**And the sharper form, which is what makes it a class and not an incident: supersession is a
READ-SIDE property.** Writing a `vN+1` is free; it means nothing until every consumer resolves
the chain head. This programme already knows that — `declaration_chain.py` exists, R-711
governs it, and REVIEW 176 §2 found the register searchable by identity precisely because
identities are what survive supersession. **The freeze guard is the one enforcement path that
never adopted it.**

## 2. MEASURED AT ORIGIN — THE STATE TODAY, WHICH IS NARROWER THAN THE HEADLINE

```
be_score_neutrality.py:85   FREEZE_REL = "de_arm_freeze_v1.json"      <- a LITERAL
be_score_neutrality.py:96   freeze_path = Path(decl_dir) / FREEZE_REL <- reads that file, nothing else
v1 pins                     de_multiday_gate1_params_v29.json  066b52f2...

amendments on origin        8  (v2..v9)
params-pinning fields in them: 0, 0, 0, 0, 0, 0, 0, 0     <- NONE of the eight pins params

params digests vs the v1 pin
   v28  5315da87...  != -> REFUSES
   v29  066b52f2...  == -> the guard accepts
   v30  9e1926d1...  != -> REFUSES
   v31  dd58223c...  != -> REFUSES
```

**One refinement to the framing, and it cuts the claim down rather than up.** *"Every
amendment the user ruled tonight is invisible to the guard"* is **structurally true** — the
guard cannot see any of them. But **none of the eight amendments actually carries a params
pin**, so the invisibility has produced **no divergence yet**. **The first material
consequence is v10**, which will be the first amendment the guard needed to read and cannot.
The defect is real, structural, and about to bite; it has not bitten.

## 3. THE BLAST RADIUS IS ONE FILE AND **TWO** SITES — AND THE SECOND IS IN THE FALSIFIER

```
grep FREEZE_REL          -> lines 85 (the literal), 96 (frozen_params), 632
grep de_arm_freeze_v1    -> be_score_neutrality.py ONLY; no other module hardcodes it
```

**Line 632 is inside a falsifier cell:**

```python
frozen_name = Path(json.loads((Path(DECL) / FREEZE_REL).read_text())
                   ["frozen_parameters"]["params"]["path"]).name
note("a malformed later params file cannot move the frozen comparator",
     {v["params_file"] for v in AH.values()} == {frozen_name})
```

**A fix that repairs line 96 and leaves 632 reading v1 makes the falsifier assert against
v29's name while enforcement resolves v31** — the REVIEW 111 "two pin sites" shape, inside the
cell that exists to prove the comparator is pinned. **And the cell's MEANING changes under the
fix:** *"a malformed later params file cannot move the frozen comparator"* was true because
nothing could move it; once an amendment can, the property to assert is *"only a params file
pinned by the resolved chain head can move it"*. **A fix that leaves that sentence unchanged
has left a control that no longer tests what it says.**

## 4. THE FOUR CHECKS, PRE-DECLARED — PLUS TWO THE LIST IS MISSING

Yours, as predicates, to run the moment DE 269 is on origin:

1. **Version order and last-pin-wins** — amendments removed → **v29**; present → **v31**;
   a synthetic `v11` pinning a different digest → **v11**. *(I will build the synthetic in
   scratch; the real chain is not touched.)*
2. **`de_arm_freeze_v10_amendment.json` pins v31's digest `dd58223c…` and nothing else** —
   diffed field-by-field against v9, not read.
3. **The valuation commit differs from `da00220` ONLY in `be_score_neutrality.py`** — every
   other path byte-for-byte.
4. **Not a bypass:** a params file matching NO freeze in the chain still refuses
   `PARAMS_ARE_NOT_THE_FROZEN_PARAMS`.

**MINE, because the four do not cover them:**

5. **BOTH sites, not one.** Line 632 must resolve through the same resolver, and its cell's
   assertion must be restated to the post-fix property (§3). **A resolver installed at one of
   two pin sites is the defect it repairs, relocated.**
6. **A BROKEN CHAIN MUST REFUSE, NOT SORT.** Check (1) tests ordering on a well-formed chain.
   A resolver that globs `de_arm_freeze_v*` and sorts numerically will happily resolve a chain
   with a **hole** (v10 present, v9 deleted) or with a `supersedes` pair that does not verify.
   **Drive: remove v9 from a scratch copy and confirm the resolver REFUSES by name rather than
   silently taking v10.** R-711 already requires the pair to verify for declaration chains;
   the freeze chain must not be the exception.

## 5. STATUS

**DE 269 is NOT on origin** — tip `356da4a` (DA 242, 08:02:13Z), and
`de_arm_freeze_v10_amendment.json` does not exist. `FREEZE_REL` is still the literal at line
85. **The defect described above is live on origin as I write.** I will run §4's six checks
the moment the resolver lands and report once.

**Also still owed and unchanged:** REVIEW 191 §5's cent-match on the superseded book
(−11,017.712006 / +5,256.176844), which needs the heavy lock, and REVIEW 190's
zero-by-absence fields, which DE 263 was to land.
