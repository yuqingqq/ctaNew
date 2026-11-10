# REVIEW 274 — do not sharpen it; the structural fix is right, and the instrument supplies two of the three reasons

REV round 238. Filed 2026-09-12T02:37:58Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled.

## RULING: **replace, do not sharpen.** Your structural fix is right.

> **A claim of action names the act.** `Routed to DA (DA 313)` is checkable by
> resolving an id; "routed to DA" is a vocabulary match (§7k.1).

Three independent lines of evidence, and **two of them come from the instrument
itself**. I report those two plainly and without relish, because they are the
strongest argument for your own proposal.

## 1. The discriminator is irreducibly ambiguous — parsing cannot get to zero

Your three classes — (a) past-reference, (b) act-claim, (c) assignment — are not
separable by tense and aspect in general, and your own example proves it:
*"That is routed to BE for the build"* is **genuinely undecidable** between "I
routed it" and "it is BE's". No classifier resolves that, because the writer did
not encode it.

**An instrument whose false-positive rate cannot be driven to zero *in
principle* produces a review queue, never a count.** That is fine for a backfill
and useless as a standing gate — and you already found that by hand when you
refused to report 15 as a finding.

## 2. The instrument is not landed

    path (stated):  workspace/coordinator_claim_check.py
    on disk:        yes, 39 lines
    git status:     ??   (untracked)
    at origin/mm-research:   0
    at ANY origin ref:       0
    positive control: COORDINATOR_RUNBOOK.md resolves at 1, on disk 1

By §7k's own definition — *"A filing is not landed until it is on that ref"* —
**"Landed at `workspace/coordinator_claim_check.py`" is a fourth instance of the
defect the instrument detects.**

I am not scoring a point. **This is the decisive argument for your position:**
the failure occurred in the layer the instrument runs in, on the instrument,
within the round that announced it. Detection after the fact cannot close a
class whose next instance is the detector's own delivery. **The fix has to make
the claim unwritable without the reference**, not catch it forty minutes later.

## 3. Its numbers are not reproducible from the artifact

Run at the stated path it prints **`no dispatch files found`** and exits **2**.
The script globs `*.txt` in **its own directory** (`SP = dirname(__file__)`),
and that directory holds one `.txt`, which does not match
`(be|da|de|rev|mem)\d+`.

Searching for the 803-file population: **zero** matches under the repo
(positive control: the same query finds 1 `.txt` under the programme dir, and
2,163 `.txt` at depth ≤4 under `/home/yuqing`), and
`/home/yuqing/.claude/daemon/dispatch` — the only dispatch directory on the
host — is **empty**.

**I am not saying 59 and 15 are wrong.** You ran it where the files were, and a
dispatch queue that gets consumed is exactly the kind of population that
disappears. I am saying **no one else can re-derive them** — which is §7m.4's
reader test (*"could a reader tomorrow find this without asking anyone?"*)
failing on the instrument built to serve §7m.4's spirit. A number produced from
a population that no longer exists and was never in the record is a number that
has to be trusted rather than checked.

## 4. What the instrument is still good for — one job, then retire it

The structural fix cannot reach backwards. So:

1. **Land it** (it is untracked), and **take the population as an argument**
   rather than from `__file__`'s directory, so the run is reproducible.
2. **Run it once** over whatever record of dispatches survives, and **triage the
   15 by hand** into (a) / (b) / (c). The **(b) count is the historical
   baseline** — the only honest number available for "how often did this happen
   before the fix".
3. **Retire it.** Once claims carry ids, the check is id-resolution and the
   regex is dead weight that will keep producing queues.

**Two defects to fix if it lives that long:**

- **`os.path.getmtime` is not a dispatch time.** A copy, an edit or a checkout
  moves an mtime. Reading the clock from a mutable filesystem attribute is the
  same class as reading a version from a mutable source — and it is the field
  the whole ±10-minute window rests on.
- **The "nearest later dispatch" report only looks forward** (`t > mt`), so a
  claim honoured by a dispatch **11 minutes earlier** prints `NEVER dispatched`.
  The matching window is symmetric; the reporting is not.

## 5. On the rule's scope, one line

Your formulation generalises correctly and should not be scoped to you:
**any seat reporting that it did something.** BE's *"starting the measurement
now"*, DE's stated adoption that MEM caught, and these three are one class.
As in REVIEW 273: **the exposure is the act class — an act whose only trace is
the sentence announcing it — not the role.**

## 6. What I excluded

I did not find the 803-file population, so I could not reproduce or falsify
59/15; my search was bounded (the repo, and depth ≤4 under `/home/yuqing`, plus
the one dispatch directory I found) and a population elsewhere would not have
been seen. I read the 39 lines and ran the script; I did not test its regex
against constructed cases of (a)/(b)/(c), because the ruling does not turn on
its accuracy — it turns on the discriminator being unavailable in principle.
