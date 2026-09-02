# Review — DA round 6: blackout mask producer + complement block (R-409)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `181b4fa`** (Q-DA-205).
**Request of record:** `REQUEST_DA_MASK_PRODUCER_2026-09-02.md` (at `434dc5d`).
**Composed 2026-09-02T10:17:20Z.** One filing, per R-377. RR8-1 is ruled (R-412) and
is not re-found here.

Executed in `~/ctaNew-wt-rev` at `--detach 181b4fa`; the filing lands in the shared
tree under R-387's pathspec discipline (R-397 item 5).

---

## FIRST, AN INCIDENT OF MY OWN: I overwrote two canonical verdicts

Reviewing point 6 I ran `da_midnight_verify.sh` with `OUTDIR=` and `LOG=` set. **Those
are not the script's variable names.** It reads `DA_MIDNIGHT_OUTDIR` / `DA_MIDNIGHT_LOG`,
saw neither, and did what it is built to do: a production run into the canonical
directory. It rewrote `da_dayverdict_20260901.json` and `da_dayverdict_20260902.json`
at 10:16:16Z / 10:16:19Z.

**Damage assessment, measured:**

- both artifacts carry `write_reason: "UNATTRIBUTED hand run of da_midnight_verify.sh"`
  and a `supersedes` block naming what they replaced — the instrument attributed my
  write correctly and loudly;
- **09-01: `all_pass` true → true, `race_accrual_eligible` true → true** (the
  `supersedes` block records the replaced values, and they match);
- **09-02: `all_pass` false → false, accrual false → false**; it is now a 10:16Z
  snapshot of an open day rather than a 00:06Z one, and tonight's closing verdict
  supersedes it regardless;
- independently, I had already established that **zero governing fields differ**
  between the pre-batch and post-batch code on 09-01, so nothing of record moved.

No governing number changed and the write is self-declaring. It was still my error,
and two findings below (RR9-3) come directly out of it — including one that says the
provenance note on those artifacts is not true.

---

## Verdict

### RELEASED. The mask exports rather than redefines, the complement states its denominators, the pins hold, and both RR7 closures verify.

Three findings, none holding: **RR9-1** (the export contract is data-dependent and the
suite cannot see a redefinition), **RR9-2** (one bar constant unstated), **RR9-3** (the
rehearsal guard and the provenance note, both surfaced by my incident).

---

## 1. Export, not redefinition — the equality is real, its coverage is not

The equality is against the **frozen module's own number**: `build_mask` imports
`da_content_liveness_rule` and compares `len(masked)` with
`CLR.measure_day(...)["coins"][c]["n_invisible_thin"]` — not against a copy inside the
builder. The window list is re-derived with `CLR.THIN_FRAC` and v1's gap-overlap
exclusion.

| mutant | suite | real build |
|---|---|---|
| the equality check removed | **KILLED** — *"a mask/L1 disagreement must REFUSE"* | — |
| the UNJUDGEABLE/UNRESOLVED refusal removed | **KILLED** — *"an UNJUDGEABLE day must refuse a mask"* | — |
| **gap-overlap exclusion dropped** | **17/17 PASSES** | 09-01 **REFUSED**; **08-26 BUILDS** |
| **thin fraction 0.05 → 0.06** | **17/17 PASSES** | 09-01 **REFUSED**, 08-26 **REFUSED** |

### RR9-1 — MEDIUM — the export guarantee is data-dependent, and the suite is blind to it

The equality can only fire when the two definitions **disagree on that day's data**.
Both redefinition mutants pass the suite; on real data the guard catches them on
09-01 — but **dropping the gap-overlap exclusion still builds 08-26 successfully**,
because no thin window on that day overlaps a gap row. So a redefined mask ships
silently on any day where the change happens not to matter, which is precisely the day
nobody would look.

DA's red-first control perturbs the **frozen count** (`n_invisible_thin += 1`), which
proves the comparison is plumbed; nothing pins the **mask's own rule** to v1's.

**Closure — one fixture, two rows:** a day containing (a) a thin window that overlaps a
gap-ledger interval and (b) a window between `0.05×` and `0.06×` the median. Either row
alone kills both mutants at the suite level and turns a data-dependent guarantee into a
structural one.

## 2. The complement — the open-day control fires and the arithmetic is exact

**Executed known-bad:** reverting the complement to `range(288) - masked` →
**KILLED**, naming it: *"with 119 windows present and 10 masked the complement is 109 —
NOT 278."* DA's own defect is genuinely closed by a control that discriminates.

Recomputed by hand from the emitted block (09-02 btc, my run at 10:1xZ):

| quantity | emitted | my recomputation |
|---|---|---|
| present − masked = unmasked | 122 − 40 = **82** | 82 |
| `P1_lost_s_per_UNMASKED_hour` | **91.25** | 623.5 / 6.833 = 91.25 |
| `P1_lost_s_per_CALENDAR_24h` | **25.98** | 623.5 / 24 = 25.98 |
| `complement_fraction_of_PRESENT` | 0.6721 | 82/122 = 0.6721 |
| `complement_fraction_of_288` | 0.2847 | 82/288 = 0.2847 |

(The filing's 93.01 / 25.51 are the same pair measured earlier in the open day; the
structure is what matters and it holds.) **Both denominators are carried for P1, both
for P2, and P3 carries the frozen bar beside its value.**

**P3's reading is defensible.** The frozen bar is "worst rolling 60-minute lost
seconds ≤ 900". The complement version keeps the **calendar** 3600 s width and excludes
loss inside masked windows — which is the frozen statistic applied to the complement's
loss. The alternative (compressing time) would change the window width and invent a
statistic the frozen bar has no counterpart for, and the block says exactly that in
`denominator_note`. One property worth naming: excluding masked loss while keeping the
width means the complement's P3 can only be **≤** the frozen P3 — it can never be the
more conservative of the two, which is fine for a reported number and would not be for
a governing one.

## 3. Rule 9 — the tautology is flagged and unconsumed

`L1_over_complement: 0.0` with `L1_over_complement_is_TAUTOLOGICAL: true` and a note
saying it is arithmetic, not evidence. **Grep across `live/pm_research/*.py`: the only
file mentioning `L1_over_complement` is the producer.** Nothing downstream reads it.

## 4. The pins — recomputed, not read

- **Accrual.** The definition is `q_ok AND a_ok AND era_admissible AND day_closed`.
  Recomputed from 09-01's emitted booleans: `day_quality_pass` T, `post_freeze_pass` T,
  `era_admissible` T, `day_closed` T → **`race_accrual_eligible` True**, matching. The
  string `blackout` does not appear anywhere inside `verdict_split`; the mask block sits
  at top level with **`governs: false`**.
- **The veto pin:** `content_thin_vetoes_HEALTHY: false`, `ruled_by: "R-409"`.
- **The v2 seam:** `status: INERT_PENDING_USER_FREEZE`, `frozen_by_user: false`,
  `refuses: true`. Called directly, `v2_mask_windows('20260901')` raises
  `MaskRefused` — *"the v2 absolute-floor mask is DRAFT and NOT USER-FROZEN"*. **It
  cannot return a mask.**

## 5. `day_closed_calendar` — the flag discriminates

`20260901 → true` (141 masked), `20260902 → false` (243 masked at my run time). The
field is at top level, which is where a consumer can reach it before reading any coin
block — the schema makes BE's pending consumption easy. Under R-412 the producer's
envelope is the contract, so this is the field BE's next batch reads.

## 6. Launcher path — one key added, zero governing fields moved

Rather than trust the launcher output alone, I compared `verify_day('20260901')` at the
**pre-batch parent** (`f24a11c`) against the tip:

- **keys only in POST: `['blackout_mask_and_complement']`** — exactly one;
- **governing fields differing: none** (`all_pass`, `race_accrual_eligible`,
  `verdict_split`, `bar_regime`, `day_bar_v2`, `predicates`, `windows_gap_affected`,
  `per_coin`, `era_admission` all identical).

The launcher itself ran rc 0 end to end (and, per the incident above, into the canonical
directory rather than my intended scratch one).

## 7. RR7-1 / RR7-2 — both closed, verified by mutation and by recomputation

- **RR7-1:** replacing `binance_hf`'s shipped regex with `hyperliquid`'s →
  **KILLED**: *"the SHIPPED regex matches a REAL line from that venue's OWN log."* The
  suite now exercises each venue's real parser, and the cross-venue rejection leg is
  present.
- **RR7-2:** recomputed on 08-26 — **7 status-string differences, exactly 1
  `verdict_changed` (hype)**, matching the filing, with
  `n_coin_days_verdict_changed` aggregated and a note telling readers to compare that
  rather than the two status strings.

## 8. The escalated constant — honest, and one companion is unstated

`ESCALATION_no_minimum_complement_size` is the right escalation and states the real
risk: *"the frozen bars were pre-registered against a 288-window day; applied to a
small complement they are being read on a population they were not registered for."*

Hunting for a quiet second one: the calendar constants (288, 24 h) are **named in the
field names themselves** (`_of_288`, `_CALENDAR_24h`) and paired with their
unmasked-denominator forms, and P3's rolling width and exclusion rule are stated in
`denominator_note`. One is genuinely missing:

### RR9-2 — LOW — P2's material-span (75 s) is not stated in the block

The block emits `P2_frozen_bar_share: 0.05` but not `P2_MATERIAL_SPAN_S = 75.0`, which
is the other half of P2's definition — the rule deciding which windows are material at
all. In a block whose thesis is *"every denominator is stated"*, the numerator's
threshold should be too; a reader of the artifact alone cannot reconstruct P2.

## RR9-3 — MEDIUM — two operational findings my own incident produced

**(a) The rehearsal guard cannot catch a wrong-name override.** The launcher refuses
when `DA_MIDNIGHT_LOG` and `DA_MIDNIGHT_OUTDIR` are overridden *singly* — a good pair
guard against half-isolation. But supplying neither, which is what a **mis-named**
override produces, reads as an ordinary production run and writes to the canonical
directory. Mis-naming is the likelier operator error, and it is the one shape the guard
cannot see. **Closure:** either require an explicit `DA_MIDNIGHT_MODE=production` for
canonical writes (so a rehearsal that names nothing cannot become one), or refuse any
non-tty invocation whose outdir is canonical and whose `--write-reason` would be
`UNATTRIBUTED`.

**(b) The `supersedes` note points at provenance that does not exist.** Every replaced
verdict carries: *"its bytes remain in git history, which is the provenance."* For
these two artifacts that is **false**: `data/` is gitignored, and while three
dayverdicts (08-28/29/30) were force-added, **09-01 and 09-02 are untracked** — the
bytes I replaced are gone. The note is true for three days, false for the rest, and a
reader cannot tell which from the artifact. **Closure:** make the note conditional on
the file actually being tracked, or have the canonical write copy the prior bytes
beside the new one.

---

## Executed evidence

At `181b4fa`, as of 2026-09-02T10:17Z:

| check | result |
|---|---|
| `da_blackout_mask.py --selftest` | **17 checks**, rc 0 |
| `da_forward_day_verify.py --selftest` | **229 checks**, rc 0 |
| equality target | `CLR.measure_day(...)["n_invisible_thin"]`, frozen module imported, not copied |
| mutant: equality removed / UNJUDGEABLE refusal removed | **both KILLED** |
| mutant: gap-overlap dropped / thin_frac 0.06 | **both PASS the suite**; real build refuses 09-01, **08-26 still builds under the first** — RR9-1 |
| mutant: complement → `range(288) − masked` | **KILLED**, naming 109 vs 278 |
| complement arithmetic (09-02 btc) | **five quantities recomputed, all exact** |
| `L1_over_complement` consumers | **none outside the producer** |
| accrual conjunction | recomputed T∧T∧T∧T → **True**; mask absent from `verdict_split` |
| veto pin | `false`, `ruled_by: R-409` |
| `v2_mask_windows` | **refuses by name**; cannot return a mask |
| `day_closed_calendar` | 09-01 **true** / 09-02 **false** |
| pre-batch vs tip on 09-01 | **one key added, zero governing fields differ** |
| mutant: hf regex ← hl's | **KILLED** — RR7-1 closed |
| 08-26 v2 comparison | **7 string diffs, 1 verdict change (hype)** — RR7-2 closed |
| mutants executed | **7 — 5 killed, 2 survived (RR9-1)** |
| my incident | two canonical verdicts rewritten at 10:16Z; **no governing value changed**; both self-declare as unattributed hand runs |

---

## Disposition

- **RELEASED:** DA round 6. The mask exports v1's numerator under a contract that
  refuses disagreement, the complement is computed over windows that exist and states
  its denominators in pairs, the tautology is flagged and unconsumed, the accrual
  conjunction is untouched, the v2 seam refuses, and both RR7 findings are closed.
  **No hold from this seat.**
- **FILED, not holding:** RR9-1 (one fixture makes the export contract structural
  rather than data-dependent), RR9-2 (state P2's material-span), RR9-3 (the rehearsal
  guard's blind spot and the false provenance note).
- **Recorded against this seat:** I wrote to the canonical verdict directory while
  reviewing the launcher. The instrument's own attribution is what made the damage
  assessable in one read — which is an argument for the `write_reason`/`supersedes`
  discipline, and a reason to close RR9-3(b) so the same read is possible for the bytes
  themselves.
