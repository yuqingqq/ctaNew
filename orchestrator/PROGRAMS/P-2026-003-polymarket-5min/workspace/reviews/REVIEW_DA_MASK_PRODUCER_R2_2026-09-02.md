# Review — DA round 7 (incident recovery + RR9 closures + R-412 producer obligation)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `770e5ee`** (batch `84ec1a1` + `770e5ee`).
**Request of record:** `REQUEST_DA_MASK_PRODUCER_R2_2026-09-02.md`.
**Composed 2026-09-02T11:02:31Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 770e5ee`. The only launcher runs were the
three **refusal** shapes and two **scratch rehearsals** with both `DA_MIDNIGHT_OUTDIR`
and `DA_MIDNIGHT_LOG` set, inside the research slice. `DA_MIDNIGHT_MODE=production` was
never set; nothing was written under `data/pm_5min/derived/`; the canonical shas are
byte-identical before and after this review.

**This round closes findings of mine and repairs an incident of mine.** I have tried to
make the verification independent of both: every sha below I re-derived, and the two
findings I file are ones the batch's own tests do not cover.

---

## Verdict

### RELEASED. The recovery is sound, RR9-1/2/3 are closed, and the launcher now refuses the exact shape that caused the incident.

One finding, **RR12-1 (MEDIUM)**: the producer's provenance and the launcher's default
verifier path are **blind to per-seat worktrees** — the tree the programme adopted three
rounds ago. Both legs executed. Plus one method note that nearly cost me a false finding.

---

## 1. The recovery, re-derived from the log without using DA's line numbers

I located the run by its `======== fired 2026-09-02T00:06:00Z` marker and extracted the
echoed JSON blocks by brace-matching — the extraction landed on lines **5302–5868** and
**5913–6388**, the ranges DA cites, without being told them.

| block | my sha256 (text, no trailing newline) | artifact's `recovered_content_sha256` |
|---|---|---|
| 09-01 | **f18724e37d8f1e3f…** | f18724e37d8f1e3f… |
| 09-02 | **b1d67fcd9b189489…** | b1d67fcd9b189489… |

Comparing each restored canonical file to its recovered block:

- **keys only in the restored file: `['restored']`**; keys only in the block: **none**;
  shared keys whose value differs: **`['supersedes']`** — i.e. **exactly the two keys**,
  for both days;
- as-of `00:06:01.284484Z` / `00:06:03.718835Z`; `write_reason: "scheduled unit run,
  da-midnight-verify.service"`; all_pass/accrual **T/T** and **F/F**;
- `restored.original_supersedes_preserved` **equals** the 00:06Z block's own
  `supersedes` (both days) — the original chain is preserved, not overwritten;
- `supersedes.sha256` **equals** the sha of the `.superseded_…` sibling on disk
  (`d071030d…` and `71001e12…`) — the bytes my run replaced exist as named artifacts.

**Is restore-with-provenance the right reading of rule 13? Yes, and byte-identical would
have been the wrong one.** A byte-identical restore would have erased the only evidence
that the file was replaced and restored — the reader would see 00:06Z bytes and no
incident. Rule 13 asks for corrections *in band*: the current file carries the recovered
content **plus** a `restored` block naming what it replaces and what it recovers from,
the superseded bytes sit beside it under their own as-of, and the original `supersedes`
is preserved inside the new one. That is the shape that lets a future reader reconstruct
the whole sequence from the artifacts alone.

**Nothing in the accrual chain resolves differently.** 09-01 is all_pass True /
accrual True and 09-02 False / False — the same values in the 00:06Z bytes, in my 10:16Z
overwrite, and in the restoration. R-395/R-396 stand unchanged.

## 2. RR9-3(a) — the launcher refuses the incident shape

Pre-state recorded, three shapes run, post-state compared:

| shape | rc | expected |
|---|---|---|
| bare run | **6** | 6 |
| `OUTDIR=/tmp/x LOG=/tmp/y` (the mis-named override — **my incident exactly**) | **6** | 6 |
| one-of-pair (`DA_MIDNIGHT_OUTDIR` only) | **5** | 5 |

**Nothing was written:** both canonical verdict shas *and* the log's sha and mtime are
identical before and after (the log still ends at my 10:16:19Z run). No
`======== fired` header appears in any of the three — **the refusal precedes the log
header**, which is this script's own standing lesson.

Unit state: installed unit **≡** repo unit; `Environment=DA_MIDNIGHT_MODE=production`;
`DropInPaths=` empty; no `~/.config/systemd/user/da-midnight-verify.service.d`; timer
next elapse **Thu 2026-09-03 00:06:00 UTC**.

### The two-leg admission, weighed as asked

The guard admits a canonical run on **identity** (cgroup is
`da-midnight-verify.service`) **or** **declaration** (`DA_MIDNIGHT_MODE=production`).

**What the two-leg form admits that the single-leg form refused, other than the unit:**
a hand-made transient unit deliberately given that exact name (e.g.
`systemd-run --user --unit=da-midnight-verify …`), and a **system-scope** unit of the
same name — I confirmed `0::/system.slice/da-midnight-verify.service` matches. Both
require someone to name themselves the unit, so neither is an accident; and each still
has to reach the canonical directory to matter.

**Can the `case` pattern match a unit that is not `da-midnight-verify.service`?** No, not
by name confusion. Tested against candidates: `foo-da-midnight-verify.service`,
`xda-midnight-verify.service`, `da-midnight-verify.servicex` and
`da-midnight-verify.service.d` all **refuse** — the leading `/` in the pattern blocks
prefix collisions. It matches that exact unit name in any scope, and any child cgroup
beneath it (which is the unit's own processes).

**My view:** the deviation is defensible and I would keep it. Leg (1) is an *identity*
(what the process is) rather than an *assertion* (what it claims), and DA's availability
argument is real — a single-leg form makes every future 00:06Z verdict refuse silently
if the unit's `Environment=` is ever lost. One improvement, in the batch's own idiom:
when admission rests on **identity alone** (cgroup matched, `DA_MIDNIGHT_MODE` unset),
say so in the log. Today that case — the declaration having gone missing — is
indistinguishable from a normal run, and it is precisely the state someone should see.

## 3. RR9-1 — both my survivors now die, and the fixture cannot pass while testing nothing

| mutant (my round-6 survivors) | result |
|---|---|
| gap-overlap clause dropped | **KILLED** — *"mask/L1 disagreement for 20260404 btc — the mask lists 2 windows while the frozen detector counts 1"* |
| `THIN_FRAC` → 0.06 | **KILLED**, same guard, same day |

19 checks under both launchers.

**Rule 16, tested rather than read.** The fixture asserts its own essential property —
*"row (a) really is gap-covered and row (b) really is not — a fixture that did not carry
both shapes would kill neither mutant"* — and when I removed the gap row from the
fixture (leaving everything else), the suite **FAILED** at the structural check. The
fixture cannot quietly stop discriminating.

## 4. RR9-2 and RR9-3(b)

**RR9-2 — my LOW finding is closed.** `P2_material_span_s` sits beside
`P2_frozen_bar_share` at both emission sites and is read from `D.P2_MATERIAL_SPAN_S`,
not restated: moving the module constant to 99.0 moves the emitted value to **99.0**;
restored, both read **75.0** / **0.05**.

**RR9-3(b) — my finding is closed, and closed at the fact rather than the sentence.**
`_is_tracked` uses `git ls-files --error-unmatch`, and I exercised both branches:
**True** on a tracked prior, **False** on an untracked scratch prior. The note text
follows the fact (`"in git history AND …"` vs `"NOT in git (this path is gitignored) but
ARE …"`), and `preserve_prior_bytes` is called at line 4343 while the verdict is written
at 4369 — **the prior bytes are copied beside before the write**, which is why the
`.superseded_…` siblings I checked in §1 exist at all.

## 5. R-412 — the producer obligation

The scratch rehearsal produced **exactly four artifacts** — two verdicts, two masks —
plus its log, all in scratch, with the canonical files unchanged. Source order confirms
the sequence: `preserve_prior_bytes` (4343) → `emit_mask_artifact` (4365) → the verdict
write (4369), so **the mask is emitted before the verdict is serialised** and lands in
the verdict's own directory.

**The NOT_WRITTEN path**, driven by injecting a producer failure:

```
{"status": "NOT_WRITTEN", "why": "producer failed (injected)",
 "consumer_note": "a scorer MUST refuse a day whose mask is NOT_WRITTEN
                   rather than assume an empty one (R-412)"}
```

No file written, no exception escaping, the verdict unaffected — the reason is a
**status**, which is the whole obligation.

**The open-day mask must not score, and does not.** The 09-02 mask as rehearsed today
carries `day_closed_calendar: false` (246 windows). Through BE's adapter it **loads**
(246) and is then **REFUSED by `apply_blackout_mask`** naming the partial-mask reason —
while the closed 09-01 rehearsal mask is **accepted** (141). RR8-3 holds across the seam,
in the direction that matters.

## 6. `770e5ee` — the regenerated 09-01 mask

141 windows; per-coin **btc 23, eth 22, sol 23, bnb 22, doge 22, xrp 20, hype 9** —
identical to the previous artifact. `producer.module_sha256_prefix` = **d191695dcff0546e**
= the sha of `da_blackout_mask.py` at `770e5ee`; `carrying_commit` = **84ec1a1…**;
`tree_dirty_on_producing_files: false`. Both consumers accept the extended envelope:
BE's adapter loads **141**, DE's supplier emits **1,875**.

### RR12-1 — MEDIUM — the provenance block and the launcher are blind to per-seat worktrees

The request asked me to confirm `tree_dirty_on_producing_files` is a real predicate by
dirtying the module in scratch. **It stayed `false`** — and the reason is structural:

- **Leg (a).** `da_blackout_mask.REPO` is **hardcoded** to `/home/yuqing/ctaNew`, while
  `module_sha256_prefix` is computed from `__file__`. Run from `~/ctaNew-wt-rev` I
  measured `_head_commit()` → **`d8aaa714…`** (the *shared* tree's HEAD) while the
  worktree's HEAD was **`770e5ee…`**, and dirtying the worktree's module left the flag
  **false**. So a mask produced from a seat worktree records **another tree's** commit
  and **another tree's** cleanliness. The field's own docstring says *"a `carrying_commit`
  recorded over a dirty tree points at bytes that did not run"* — in a worktree it can
  point at a different tree entirely.
- **Leg (b).** `da_midnight_verify.sh` defaults
  `V="/home/yuqing/ctaNew/live/pm_research/da_forward_day_verify.py"`. My rehearsal ran
  from the worktree and therefore exercised the **shared tree's** verifier and mask
  producer: a mutation I applied to the worktree copy had no effect on the rehearsal,
  while the same mutation applied in-process produced NOT_WRITTEN immediately. (The
  script does guard `DA_MIDNIGHT_VERIFY_BIN` to isolated rehearsals, so the override
  exists — it is the *default* that crosses trees.)

Nothing shipped is wrong: the committed artifact was produced from the shared tree, where
both agree, and I verified `module_sha256_prefix` equals the module at its own
`carrying_commit`. But the programme adopted per-seat worktrees at R-397 item 5, so this
bites the first time DA produces a mask from `~/ctaNew-wt-da`, and it bites silently.

**Closure:** derive `REPO` from `__file__` (as `da_forward_day_verify` already does) and
default `V` to `$(dirname "$0")/da_forward_day_verify.py`; then add the one check that
makes the pair self-verifying — assert `module_sha256_prefix` equals the module blob at
`carrying_commit`, and record the mismatch as a status if it does not.

## 7. Rule 10 / rule 14 sweep

- `disposition_rule: "R-409"` and `disposition_text` **cite** the USER ruling and state
  *"This artifact REPORTS the mask; it decides nothing (rule 14)"* — a citation, not a
  disposition.
- The only booleans are `day_closed_calendar` (a calendar fact) and
  `agrees_with_frozen_L1_numerator` (a checked property whose failure refuses). Neither
  reads as an entitlement.
- `consumer_note` instructs a consumer to refuse an open-day mask — guidance, and the
  consumer's own guard enforces it independently (§5), so it gates nothing by itself.
- No printed conclusion beside a predicate in the new code; the single `print("FAIL: …")`
  is the selftest's own reporter.
- **No new number.** `ESCALATION_no_minimum_complement_size` remains the only standing
  escalation.

### Method note, recorded because it nearly cost me a false finding

A mutate-run-restore cycle completed **inside one second** on a file whose edit was the
same length (`75.0` → `99.0`) left a `__pycache__` entry Python still considered valid:
the source read 75.0 while the imported module reported 99.0. Clearing the cache resolved
it. For anyone doing mutation work in these trees: same-size edits reverted within the
same second can leave stale bytecode, which reads exactly like a mutant that survived.

---

## Executed evidence

At `770e5ee`, 2026-09-02T10:43–11:02Z:

| check | result |
|---|---|
| 00:06Z blocks located structurally | lines **5302–5868**, **5913–6388** |
| shas re-derived | **f18724e3…**, **b1d67fcd…** — both match |
| restored vs recovered | differs by **exactly** `restored` + `supersedes`, both days |
| `original_supersedes_preserved` | equals the block's own `supersedes`, both days |
| `.superseded_…` siblings | present; shas equal `supersedes.sha256`, both days |
| bare / mis-named / one-of-pair | **rc 6 / 6 / 5**, nothing written, no log header |
| canonical shas before vs after this review | **unchanged** |
| unit ≡ repo, MODE, drop-ins, timer | ≡ / `production` / none / 2026-09-03 00:06:00 UTC |
| cgroup pattern vs 8 candidate names | matches only the exact unit name (any scope) |
| RR9-1 mutants | **both KILLED by name**; 19 checks both launchers |
| fixture with its gap row removed | **FAILS** — the fixture asserts its own property |
| `P2_material_span_s` follows the constant | 99.0 mutated / 75.0 restored |
| `_is_tracked` both branches | **True** / **False** |
| preserve → emit → write order | 4343 → 4365 → 4369 |
| scratch rehearsal | **4 artifacts**, none canonical |
| NOT_WRITTEN path | status + reason, no file, verdict intact |
| open-day 09-02 mask through BE | loads 246, **refused at scoring**; closed 09-01 accepted |
| regenerated mask | 141, per-coin exact, module sha = blob at `770e5ee`, `carrying_commit` 84ec1a1 |
| both consumers | BE **141**, DE **1,875** |
| worktree provenance | `_head_commit()` **d8aaa714** vs worktree HEAD **770e5ee** — RR12-1 |
| worktree after the review | clean |

---

## Disposition

- **RELEASED:** DA round 7. The recovery is independently reproducible from the log, the
  restoration is the right reading of rule 13, RR9-1/RR9-2/RR9-3(a)/(b) are all closed
  with controls that can fail, R-412's producer obligation is honoured including its
  NOT_WRITTEN status path, and the open-day mask is refused across the seam. **No hold.**
- **FILED, not holding:** **RR12-1** — the producer's `REPO` and the launcher's default
  verifier path are hardcoded to the shared tree, so a run from a per-seat worktree
  records the wrong tree's provenance and executes the wrong tree's code. One check
  (`module_sha256_prefix` vs the blob at `carrying_commit`) makes the pair self-verifying.
- **Recommendation on the two-leg admission (not a finding):** log when admission rests
  on cgroup identity alone, so a lost `DA_MIDNIGHT_MODE` is visible rather than silently
  normal.
- **On the incident:** the bare run that caused it now exits 6 without touching a byte, I
  re-derived the recovery shas myself rather than accepting them, and the bytes I
  replaced exist on disk under their own as-of. The loop is closed at the artifacts.
