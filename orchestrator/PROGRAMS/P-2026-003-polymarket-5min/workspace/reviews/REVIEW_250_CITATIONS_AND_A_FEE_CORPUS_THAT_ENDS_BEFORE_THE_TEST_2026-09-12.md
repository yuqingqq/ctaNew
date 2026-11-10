# REVIEW 250 — neither location fixes it, and the fee corpus ends before the test begins

REV round 213. Filed 2026-09-12T00:00:02Z. Read-only; no lock; no heavy unit;
nothing written under `data/`; no book unpickled. Fetched at 23:58Z.

---

# PART 1 — I argue for NEITHER of your two options

You offered: mirror the evidence into the canonical ref, or name a second
location in §7k. I measured before answering, and **neither addresses the actual
failure**, so I am arguing for a third thing and showing the numbers that decide
it.

## 1.1 What the register's citations actually do

`COORDINATION.md`, counting only real artifact names (`da_`/`de_`/`be_`/`ev_`/
`p003_`/`exp_` prefixed `.json`):

| | count |
|---|---|
| distinct artifacts cited | **162** |
| resolvable at the canonical ref | 119 |
| **on the chain but NOT canonical** | **1** — `da_shared_tree_non_executing_v1.json` |
| only under `data/` — gitignored, can never be at any ref | **24** |
| at no ref and not on disk (dangling) | 18 |
| **carrying a ref qualifier** | **0** |
| **carrying a blob digest** | **0** |

## 1.2 Why mirroring is the wrong fix

**Mirroring the 14 declarations into the canonical ref would repair exactly one
citation out of 162.** And it would do so by creating fourteen second copies of
artifacts that are still being superseded — `da_fair_value_progress_ledger`
reached v7 tonight, the freeze declaration recomputes on every run, the manifest
seals a day that was already stale when written.

That is the defect this programme has spent the night removing, three times
over: DA's v5 declaration deliberately stopped carrying row statuses because *"a
dumped status is a copy that ages"*; the state backup created a third incomplete
location rather than a canonical one; the ledger was re-keyed because a copied
status contradicted the blob. Mirroring evidence into the record would
re-introduce that class at scale to fix one line.

## 1.3 Why naming a second location is also not enough

It is not wrong — it is just not the binding constraint. **24 cited artifacts
live under `data/`, which is gitignored by project policy** (CLAUDE.md, "large
data files — use `.gitignore`"). No number of named locations makes those
resolvable, and they include the fee audit this whole evening turned on.

## 1.4 What is actually broken: a citation with no locator

**Zero of 162 citations carry a ref, and zero carry a digest.** A bare filename
is unresolvable no matter how many locations are declared, and resolvable at any
number of them once a locator is attached. The programme already knows this
everywhere except the register: DA's ledger pins `blob_sha256_16_at_ref` per
row, the manifest pins prefix digests and `sealed_len`, the freeze declaration
records `declared_at_ref` and `ref_head`. **The register is the one document
that cites evidence without saying where it read it.**

**My recommendation, and I will argue for it against the more complicated
option:**

1. **One canonical location for the record** — reviews, `HANDOFF.md`,
   `STATUS.yml`, `COORDINATION.md`. Unchanged. Do not mirror evidence into it.
2. **Evidence stays where the executing code reads it**, and a citation carries
   its locator:
   - git-resident: `<ref>:<path>@<blob16>`
   - `data/`-resident: `<path>@<sha256-16>` plus the unit that produced it —
     which is what the declarations already record internally.
3. **§7k gains one checkable rule**, not a second location: *every evidentiary
   citation in the register resolves, and resolution is demonstrated by
   re-reading it at the locator it carries.* That is driveable the way DE's
   property-to-cell map is driveable, and it would have caught all 18 danglers
   and both of tonight's "read at an incomplete location" errors — yours and
   mine.

The one chain-only citation gets a `origin/de-freeze-chain-v2:` prefix and is
done. The 24 `data/` citations get digests, which makes them *verifiable* even
though they are not *fetchable* — which is strictly better than a location that
cannot hold them.

---

# PART 2 — noted, no action

Agreed, and it is the right call: splitting §8 from §9 amends a user-authored
plan rather than executing it, and that is the user's to do. I put it forward as
available, not as advisable, and I withdraw it as a recommendation.

One thing the clock has changed while we worked: **it is now 2026-09-12 UTC.**
Under REVIEW 247's R1–R2, a freeze going effective today yields **D1 =
2026-09-13**, which is still inside the cancellation population `09-07..09-13`,
so R3's refusal still fires. The window first comes clear at `T_eff ≥
2026-09-13T00:00Z`, giving D1 = 09-14. The collision has not aged out; it has
moved by one day.

---

# PART 3 — UNTARGETED: the fee corpus ends before the population begins

I went back to the file rather than to my own filing, and found two things,
one of which corrects me.

## 3.1 The corpus has no observation from any forward-test day

`p003_da_onchain_fee_audit__20260905T155346Z.json` is stamped **2026-09-05T15:53:46Z**,
so every block it contains predates that instant. It is the only fee audit in
the tree — `ls … | grep -ci fee_audit` returns **1** — and the newest fee
artifact of any kind is from **2026-09-06**.

**The forward-test population is 09-07..09-13.** The fee corpus therefore
contains **zero** observations from any day in the population it would be
declared over, and zero from any day the fair-value §9 clock will run on.

That is not fatal on its own — a fee schedule is not expected to change daily —
but it changes what the declaration can honestly say. "Zero, measured on
1,046/1,056 maker legs" is a statement about a corpus ending 09-05. If the
qualified zero is adopted, the receipt must say so, and DA's address query
(REV 212) should be run on a corpus that **includes** the population, not the
one that predates it. Re-running the audit forward is a cheap way to turn a
six-day-old negative existence claim into a current one — and rule 8's
as-of requirement asks for it anyway.

## 3.2 A correction to my own REVIEW 249, and a new finding inside it

The audit declares its own formula in a top-level field I did not read before
filing:

    formula = C * 0.07 * p * (1-p)

**It is `p·(1−p)`, not `min(p, 1−p)`.** I scaled with `min(p, 1−p)` in REVIEW
249 §2.2(b). Corrected multipliers, at the same rates:

| price | `p(1−p)` | 9.9% tier | vs observed 0.099 c/share | 49.5% tier |
|---|---|---|---|---|
| 0.99 | 0.0099 | 0.098 c/share | 1× | 0.490 c/share |
| 0.90 | 0.09 | 0.891 c/share | 9× | 4.455 c/share |
| 0.75 | 0.1875 | 1.856 c/share | 19× | 9.281 c/share |
| **0.50** | **0.25** | **2.475 c/share** | **25×** | **12.375 c/share (125×)** |

So my "50× and 250×" should read **25× and 125×**. The finding is unchanged —
the sensitivity is still one to two orders of magnitude light, and DE's
respecification should use `p(1−p)`, not `min(p,1−p)` — but the numbers in
REVIEW 249 §2.2(b) are superseded by these.

**And the reason I could not tell from the data is itself the finding.** At
p = 0.99 the two candidate schedules are `0.0099` and `0.01` — a **1%**
difference. **All ten charged legs are at exactly 0.9900.** So the corpus cannot
distinguish `p(1−p)` from `min(p,1−p)`, and the two disagree by a **factor of 2**
at p = 0.5 where the strategy quotes. The functional form of the fee is
unidentified from the evidence, not merely its rate — and the audit's `formula`
field asserts one of the two without the data being able to separate them.

That compounds with the control I flagged last round: the formula reproduces
**12.21%** of taker fees where fees are 901/901 certain. A formula that is
unvalidated on the certain side and unidentifiable on the observed side is not
a basis for a worst case. **DE's sensitivity should be specified over both
candidate forms and report the larger**, which costs one extra column and
removes the choice.

## 4. Owed

- REVIEW 247 Part 1 awaits a resolver; REVIEW 246 and 245 items remain open.
- Standing: one review per day for the 09-09..09-13 records (REVIEW 228) —
  09-11 is now closed and unread by me; the round-boundary landing sweep as
  counts at the canonical ref, stated as such.
