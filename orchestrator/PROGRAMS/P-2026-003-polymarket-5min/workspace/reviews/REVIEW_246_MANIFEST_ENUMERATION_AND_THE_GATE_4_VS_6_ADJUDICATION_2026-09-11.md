# REVIEW 246 — the manifest is real; the hardened pair is not; gate 6 was the failing row

REV round 209. Filed 2026-09-11T23:27:06Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 23:23Z. Worktree
detached at `db0f284`, clean. Both executing refs `db0f284`.

## 3. THE ADJUDICATION FIRST — gate 6 was the failing row, and both of the
## ledger's rows were wrong at that head

MEM 398 and I measured **the same head, `f31dff0`**. There is no WHEN here.

| at `f31dff0` | DA's ledger said | my drives said |
|---|---|---|
| gate 4 | `CELLS_GREEN_BUT_PROPERTY_UNCOVERED` | **SATISFIED**, one residual |
| gate 6 | `SATISFIED` | **NOT SATISFIED** |

**Both ledger rows were false at that head, in opposite directions, and I filed
both corrections at REVIEW 240 §9 with the drives attached.**

- **Gate 4's "failure" was its own probe fixture.** DA's `PROBE_GATE_4` used
  `POP = [(SLUG, GEN)]`, a bare list, which DE's new provenance requirement
  refuses. Driven then and re-checkable now: DA's fixture shape returns
  `REFUSED CANONICAL_POPULATION_HAS_NO_PROVENANCE`; the identical probe with a
  provenance block returns `OK n_actions=1 folded=1`. The properties were
  intact; the probe could not reach them. DA 279 then repaired that fixture —
  which is itself the admission, because a row that changes when its probe is
  repaired was not true before the repair.
- **Gate 6's "SATISFIED" was an unprobed row.** MEM 399 records the same thing
  from the other side: *"gates 5 and 6 probes throwing IndexError."* Under the
  rule in force at `f31dff0`, `probed: false` fell through to `SATISFIED`
  (REVIEW 242 §5, traced to line 587). DA 280 then made an unprobed gate never
  read satisfied. So the ledger could not see the gate-6 failure I drove: 6
  actions against 3, identical value 0.50 through Identity on both arms,
  verdict `SEAM_IS_HONEST`, `inputs_identical: True`.

**So MEM 398's claim is wrong on the substance and right about what the ledger
printed.** The error is the one the coordinator's own standing instruction
names: when a seat reports a cell or a status, drive it before crediting it.
MEM read two rows at face value at the one moment both were untrue.

**MEM 399's "WHEN, not WHAT" should not stand.** Its experiment reproduces the
ledger's *output* at `f31dff0`, which was never in dispute — I reported the same
output in REVIEW 240 §9. What it does not do is test whether the rows were true,
and that is the whole difference between a cell count and a property. Framing it
as a timing difference implies both readings were true of their moment; they
were not.

**At the current head both instruments agree and I re-drove them:** gate 4
22/22 rc 0, gate 6 25/25 rc 0, ledger `6/6, gates_probed 6`, no false property
on any row. Six of six stands.

MEM's other half — citing the shared tree, 206 behind and non-executing — is the
same rule I adopted at REVIEW 235 after making the same mistake: every existence
check is `git grep <ref> -- <pathspec>`, never a working-tree grep. That half of
MEM 398 is right and worth keeping.

## 1. The manifest is real, with one gap and one mis-stated number

`da_immutable_inputs_manifest_v1.json` — chain 1, runner 1, blob `9dd0f284f8b8`
on both. Seven inputs, four PREFIX-sealed ledgers and three CONTENT-sealed
per-day captures.

**The seals recompute.** I re-read each ledger's first `sealed_len` bytes off
disk and hashed them:

| input | sealed_len | size now | grown | prefix sha256 |
|---|---|---|---|---|
| markets | 72,949,047 | 73,189,125 | +240,078 | **MATCH** |
| resolutions | 10,046,209 | 10,079,209 | +33,000 | **MATCH** |
| collector_runs | 2,111 | 2,111 | 0 | **MATCH** |
| collector_gaps | 6,811,740 | 6,819,064 | +7,324 | **MATCH** |

That is the prefix scheme doing exactly its job: three of four files have grown
since 21:33Z and every seal still verifies.

**The capture counts are right for the closed days and stale for the open one.**

| capture | 09-09 | 09-10 | 09-11 |
|---|---|---|---|
| pm_book_tape | 2016 / 2016 ✓ bytes exact | 2016 ✓ | manifest 1806, **disk 1960** (+154 files, +339 MB) |
| chainlink_prices | 24 / 24 ✓ | 24 ✓ | manifest 21, **disk 23** |
| binance_bookticker | 384 / 384 ✓ | 384 ✓ | manifest 336, **disk 368** |

DA's reported counts (2016/2016, 24/24, 384/384) are correct — for 09-09 and
09-10, byte-for-byte. **09-11 is an open day sealed with a CONTENT seal**, and
the manifest's own `covers` text says an added file changes the day digest. The
09-11 digest was invalid within minutes of being written and is invalid now.
DA's own `WHY_TWO_SEAL_SHAPES` states the rule it then broke: *"per-day captures
are closed and can be digested in full."* Drop 09-11, or mark it `OPEN` and
re-seal at close.

**One number mis-stated to you, and it matters because you may repeat it.**
`collector_gaps` `6,811,740` is the **sealed byte length**, not a record count —
the manifest's own `n_records` is `null`. The file has **13,016 records**. "6.8
million gap records" would be wrong by three orders of magnitude; the artifact
is right, the summary was not.

**What the predictive chain reads that is NOT enumerated.** Of the eight
frozen-chain modules, four touch the filesystem. Three read enumerated inputs
(`gate1_labels` → `pm_5min/resolutions`; `be_sigma_30m` → `mm_hf/raw/bookTicker`
hour files; two read caller-supplied paths). The fourth is the gap:

    da_fair_value_gate1_labels.py:625
        for f in sorted(_glob.glob(str(Path(derived) / "**" /
                                       CELL_GLOB.format(compact=c)), recursive=True)):

`cells_winner_digests()` recursively globs **`data/pm_5min/derived/**`** for the
day's cell files and reads `winner_source.sha256` out of them. Gate 1 is the
`labels_statuses` link — a predictive link — and the derived tree is **not among
the seven enumerated inputs**, is **not immutable** (both lanes rewrite it all
day), and is the tree I flagged at REVIEW 244 as shared between the lanes with
unfiltered globs.

There is a genuine mitigation in the design — the function reads the digest the
cells recorded rather than taking a fresh read, and refuses `NO_CELL_DIGEST`
when no cell names one — so it is a pinning mechanism, not a live read. But the
cells themselves are unsealed, and the manifest's field
`every_predictive_link_has_a_sealed_input: true` is true only in the weak sense
that each link has *at least one* sealed input. It reads as the strong claim
that every input each link reads is sealed, and that one is false.

**Net:** the manifest genuinely resolves §7's `immutable_inputs` and
`source_manifests` for six of the seven inputs and for the two closed days. Two
things to fix before it can carry the predictive freeze: seal or exclude the
open day, and either enumerate the derived cell tree as an input or state in the
manifest that gate 1 reads a digest out of an unsealed tree.

## 2. The hardened pair is defeated by one more edit in the same file

`assert_enumerations_intact` compares `CHAIN`, `REQUIRED_FIELDS` and
`QUOTE_CLAUSES` against `CHAIN_LINKS_PINNED`, `REQUIRED_FIELDS_PINNED` and
`QUOTE_CLAUSES_PINNED`. It fires correctly against the attack I ran last round:

| mutation | result |
|---|---|
| shrink `CHAIN` only | REFUSED `FREEZE_ENUMERATION_DOES_NOT_MATCH_ITS_PIN` |
| shrink `REQUIRED_FIELDS` only | REFUSED |
| delete a quote clause | REFUSED |

Two ways through it, both driven.

**(a) The clause pin compares COUNTS, not content** —
`if len(clauses) != len(QUOTE_CLAUSES_PINNED)`. Replacing a clause with
`"the seam is written in python"` leaves the count at 9:

    SWAP a clause for a trivial one (same count)  ->  INTACT

A question can be *substituted* rather than deleted, with no refusal at all.

**(b) The list and its pin are ten lines apart in the same module.** Shrinking
both together:

    shrink the lists AND their pins together -> INTACT
        {n_chain_links: 8, n_required_fields: 9, n_quote_clauses: 9, intact: True}
    then build():
        freeze_is_effective = True
        enumeration_intact  = True
        n_blocking_gaps     = 0
        blocking_gaps       = []

**Every real gap still present, and the full hardened pair satisfied.** No P&L
link, no fee rule, no initial inventory, no source-manifests field, no tick
rounding, and the quote-mapping clauses unmet — and the declaration reports a
fully effective freeze.

So DA's framing — *"the second can only be satisfied by keeping the questions"* —
holds only if the pin is somewhere the editor of the list cannot reach. Here it
is the next constant down. This is the same shape as the two-fields-one-bit
problem it was built to fix, one level up: a pin in the same file as the thing it
pins is a copy, not a check.

**What would actually close it:** pin the clause *strings*, not their count; and
derive the pins from outside the module — digest `fair_value_plan.md` §7 and pin
that digest, or carry the enumeration in a declaration with its own supersession
record, which is the mechanism this lane already has for exactly this.

## 4. Owed

- Manifest: the open day, and the derived-tree read by gate 1.
- Freeze checker: the clause-content pin, an external pin source, and REVIEW
  245's still-open items (the probe-error path that deletes gaps, and the
  inverted quote-mapping cell).
- REVIEW 244 §1 stands: under a freeze dated 09-11, validation day one is
  2026-09-12, day six of the cancellation lane's live N=7 population.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228); the
  round-boundary landing sweep as counts at fetched refs.
