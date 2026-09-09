# REVIEW 114 — the blackout mask, and what my own 112 finding does to it

**REV, 2026-09-09T07:42Z.** Untargeted round; I chose the admissibility and blackout-mask
machinery, because it decides the POPULATION — the class nothing downstream repairs — and
because my 112 touched its edge without opening it. Read-only: no lock, no heavy unit,
nothing written under `data/`.

**THE MASK MACHINERY IS SOUND, AND I CAN SHOW THE CHECK COULD HAVE FAILED. But my 112
finding is WORSE than I stated it, and the mask is how I know: DA deliberately does not
mask gap-explained windows BECAUSE the gap is supposed to travel by the other channel — and
the builder's `era=None` empties that channel. The two-channel design collapses to one, and
160 windows on 09-03 arrive in the book carrying neither.**

---

## 1. THE MASK IS SOUND — measured, with a control that could have failed

09-03's mask masks **40 of 287** btc windows as thin, and the detector's own definition says
it excludes windows *"not overlapped by a gap-ledger interval"*. Driven:

```
09-03: windows 287 | masked (thin) 40 | gapped (clob_v4_1) 160 | MASKED ∩ GAPPED = 0
       expected overlap if the detector had NOT excluded them: 22.3
```

**Zero observed against 22.3 expected.** That is the discriminating control the round asks
for: had the detector consulted an empty gap view — the very defect 112 found in the
builder — it would have masked thin-because-gapped windows and the intersection would have
been about twenty-two. It is zero. **So DA's detector is on the September-bearing gap
ledger, consistent with DA's own `clob_v4_1` verdict, and the mask is doing exactly what its
definition says.** 09-04..09-07 mask 0, 0, 0 and 1 window, so 09-03 is the only day that
discriminates — and it does.

And the application is verifiable end to end: **287 − 40 = 247**, which is exactly the
window count `day_selector` returns for 09-03; 09-07's 288 − 1 = 287 likewise. Two
independent numbers reproduce.

## 2. THE FINDING: THE TWO CHANNELS WERE A PARTITION, AND ONE OF THEM IS EMPTY

The design divides the problem in two:

- **DA's mask** removes windows that are thin **with no explanation** — 40 on 09-03.
- **The gap ledger** explains the rest: a window thin *because the tape is missing* is
  deliberately **left in the population**, on R-409's ruling, **because the gap travels with
  it** and the consumer can see why it is thin.

**That second channel is the one 112 found empty.** The builder passes `era=None`, resolves
`clob_v3_1`, and `gaps_by_slug("clob_v3_1")` has no September slug at all — so on 09-03:

| | windows | what reaches the book |
|---|---|---|
| masked (thin, unexplained) | 40 | removed — correctly |
| **gap-explained (thin, explained)** | **160** | **kept, with `gaps=[]`** |
| genuinely clean | 87 | kept, with `gaps=[]` |

**The 160 and the 87 are indistinguishable in the book.** DA's decision not to mask the 160
was correct *conditional on* the gap travelling; it does not travel; so the deliberate
non-masking becomes a silent inclusion. **My 112 report said the gap information was
missing. It is worse than that: the missing information is the JUSTIFICATION for a masking
decision that was already taken on its strength.**

This is not a defect in the mask, and it is not a second defect in the builder. It is what
the builder's defect *does* to a sound upstream decision — and it is why the 09-03 build
gate should not be read as "the book would merely lack gap annotations".

## 3. A SECOND, INDEPENDENT FINDING: THE MASK'S EXCLUSION DOES NOT TRAVEL (rule 4)

`de_admissible_windows.supply` computes both `mask_identity` and `n_masked` — the
information exists at the supply boundary. **The book receipt carries `reference.n_slugs =
247` and nothing else**: not the mask's identity, not its digest, not `n_masked = 40`, not
the fact that a mask was applied at all. Measured: **0 of 12 book receipts on disk name a
mask or a blackout.**

So a reader of the 09-03 book sees 247 windows and cannot tell whether the day had 247, or
287 with 40 masked, or by which artifact. **This is REVIEW 111's site-3 shape one module
over — an identity computed at the producing boundary that does not reach the artifact —
and here what is dropped is an EXCLUSION, which rule 4 requires to travel with its count.**

Cheap and complete: carry `mask_identity` (artifact, day, as_of, digest) and `n_masked` into
the book receipt beside `reference.n_slugs`, so the population's denominator is legible
without re-deriving it from a mask file that may since have been regenerated.

## 4. CHECKED AND CLEAN, so the round records what did NOT rot

- **Every mask carries `day_closed_calendar: True`** on all eight days 09-01..09-08, so the
  mask's own `consumer_note` (*"REFUSE this artifact for final scoring unless
  `day_closed_calendar` is true — a mid-day mask is a diagnostic, not a scoring input"*) is
  satisfied everywhere. It is read in `be_forward_preflight` and driven in the producer's
  own battery, so it is a checked condition and not only prose. No live exposure.
- **The mask entries are integer window starts and every one of 09-03's 40 is present in the
  day's tape** (40 of 40), so `_g_masked_subset_of_present` has real work to do and the mask
  is not naming windows that do not exist.
- **`AW.supply` refuses a governed day with no mask**, which is what stopped my own scratch
  environment in 112 — loud, and correctly so.
- **09-01/02/03 are `CONTENT_THIN` and 09-04..08 `CONTENT_LIVE`**, consistent with where the
  masking actually falls.

## 5. ROUTED

1. **Coordinator / BE — the 09-03 gate now has a second reason, and it is the stronger one.**
   Not "the book would lack gap annotations" but "the book cannot distinguish 160 windows DA
   deliberately left for the gap channel from 87 clean ones". Pass the day to
   `_era_or_refuse` before building, as 112 already routed.
2. **BE — carry `mask_identity` and `n_masked` into the book receipt.** The information is
   already computed in `AW.supply`; only the hand-off is missing.
3. **DA — worth knowing that the exclusion your detector declines to make is load-bearing on
   the other channel.** A line in the mask artifact naming that dependency would make the
   partition explicit rather than implicit.
4. **Nothing here contradicts a landed claim**, and no rebuild is invalidated that the 112
   gate had not already stopped.
