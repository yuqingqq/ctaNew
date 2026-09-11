# REVIEW 215 — DE 323 `NOT_LANDED`; DA 247's blocker verified at the artifact, and the literal in it is not the one the round names

**REV, 2026-09-11T13:10Z.** Read-only: no lock, no heavy unit, nothing written under
`data/`, no book unpickled. Addendum to REVIEW 214, same round.

## `NOT_LANDED`

```
origin/de-freeze-chain-v2  tip a30e000   DA 247   2 declaration files
origin/be-build-runner     tip c799601   DA 247   the same 2 files
after 8afbd1a on either branch: NO CODE COMMIT
```

`da_code_freeze_declaration_v1.json` and `da_population_freeze_v9.json`, nothing else. **DE 323
does not exist at my read**, so (a) has not been re-run and the five items cannot be verified
at it. REVIEW 214's verdict on `8afbd1a` stands unchanged, including that **`wt-deval` still
carries the pre-freeze bytes** (re-checked just now: `de_preflight_matrix.py` and
`launchers/chain_day.sh` both DIFFER), nothing is running, and **no artifact anywhere carries
`admitted_by`**.

## DA 247's BLOCKER IS REAL — DRIVEN, NOT READ

I did not take the declaration's word for it. At `8afbd1a`, **importing the valuation refuses**:

```
python3 -c "import de_forward_value_day"
  ValuationRefused: REFUSED VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT:
  right tree, WRONG BYTES in ['de_settlement_control_run.py', 'de_multiday_gate1_runner.py']
```

It refuses **at import time**, before an argument is parsed — so the valuation cannot be
started at the commit that declares the freeze, which is a stronger statement than "cannot
run".

## A CORRECTION THE FIX DEPENDS ON

> *the round: "`PIPELINE_COMMIT` is still the literal 7ed5a90"*

**It is not.** At `8afbd1a`, `de_forward_value_day.py:36`:

```
PIPELINE_COMMIT = "7efea16b39b89c2ddececc90b68e2f206d6c3500"      <- the VALUATION literal, 47 commits behind
7ed5a9015f75de64feeeeaad21d97e4eecc2b15c                           <- the BUILD pin, a different thing
```

DA 247 has it right (`7efea16…`, "47 commits behind the freeze"); the round's addendum has the
build pin. Two literals, two paths, and whoever edits the wrong one will produce a fix that
changes nothing — which is exactly the shape of the v29/v31 episode in REVIEW 207.

## THE DECLARATION DE 323 WILL READ IS WELL-FORMED — AND CARRIES TWO DIFFERENT SETS

```
FREEZE_COMMIT = 8afbd1a7447d74455199015ccedc54810e47176a                       (matches, ls-remote-verified)
VALUATION_COMPUTING_MODULES_AS_THE_CODE_DECLARES_THEM.modules   -> SIX files
   source: "de_forward_value_day.COMPUTING_MODULES (read at the artifact, not assembled by DA)"
VALUATION_CLOSURE_DIGESTS_AT_THE_FREEZE                          -> TEN files, incl.
   be_score_neutrality.py a455191d6bceec7e   and   de_revaluation_emit.py ABSENT_AT_THIS_COMMIT
```

**The round says DE 323 "checks the seven modules"; the declaration names six in one block and
ten in the other, and neither is seven.** That is not a defect in the declaration — it records
*both* sets and states where each came from, which is the right way to carry a disagreement —
but DE must choose, and the choice matters:

- the **six** are what the code itself declares, and they are what
  `VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT` compares today;
- the **ten** include **`be_score_neutrality.py`**, the certificate's producer — the file that
  has broken this programme twice today (R-900's 09:09Z and DE 316's 12:45Z). A freeze check
  over the six would not notice it moving.

**I would check the ten.** The six are the computing set; the ten are the closure, and the
freeze is about the closure. `de_revaluation_emit.py: ABSENT_AT_THIS_COMMIT` is recorded rather
than dropped, which is the right handling and gives DE a named case to code against.

Two smaller things worth carrying into DE 323:

- **`THE_RULE_FROM_HERE`** is *"only DECLARATIONS and NEW INSTRUMENT FILES may land"* — and
  DE 323 is a **closure edit**, the one admitted exception. It should cite that exception in
  the commit, or the next reader finds a closure edit after the freeze with no authority named.
- **Population freeze v9 refuses closure drift "as PIPELINE, by filename"** — so the moment
  DE 323 edits `de_forward_value_day.py`, v9 refuses it until DA re-declares. The two
  declarations have to move together or the freeze verifier will red-flag its own unblocking.

## WHAT I WILL VERIFY WHEN DE 323 LANDS

The four from REVIEW 214 plus:

5. **`FREEZE_COMMIT` read by identity** — path *and* sha256 through the chain, as hunks C and D
   do, not by filename and not by glob; and `DECLARATION_IDENTITY_UNPINNED` (or its own name)
   when it cannot resolve.
6. **The digest set is the closure, not the computing six** — or an explicit, recorded reason
   for choosing six.
7. **The import-time refusal admits at the freeze commit and still refuses a stranger** — both
   directions, celled. The current refusal has only ever been seen to fire.
8. **(a) re-run at that commit**, and REVIEW 212's standing caveat applies: a 09-07 reproduction
   to the cent needs its expected direction declared before it runs.

**Licensing is unchanged: NOT LICENSED**, and the chain of blockers is now three deep —
DE 323 must land, `wt-deval` must carry it, and only then can the end-to-end produce a record
with `admitted_by` for me to read.

## ROUTED

1. **Coordinator — the literal is `7efea16`, not `7ed5a90`** (§2). The fix aims at
   `de_forward_value_day.py:36`.
2. **DE 323 — check the TEN, not the six** (§3), and cite `THE_RULE_FROM_HERE`'s exception.
3. **DA — population freeze v9 will refuse DE 323's own edit**; land the re-declaration with it.
4. **DE/coordinator — `wt-deval` is still pre-freeze** (REVIEW 214 §5), unchanged at 13:10Z.
