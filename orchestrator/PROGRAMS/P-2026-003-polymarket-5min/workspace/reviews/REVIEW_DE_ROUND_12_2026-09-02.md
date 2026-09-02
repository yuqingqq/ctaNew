# Review — DE round 12 (DE10-R1: every temporal comparison parsed, not sorted)
reviewer: claude (pm-codex seat) · round opened by the coordinator (pm-co)

**Pinned tip executed: `9dbaa5a`** (Q-DE-30).
**Request of record:** `REQUEST_DE_ROUND_12_2026-09-02.md`.
**Composed 2026-09-02T12:05:43Z.** One filing, per R-377.

Executed in `~/ctaNew-wt-rev` at `--detach 9dbaa5a`. Read-only under `data/`; register
fixtures on copies in temp dirs — `COORDINATION.md` never written. No timer, no service,
no launcher. DE11-R1 is out of scope here and is not re-filed.

---

## Verdict

### RELEASED. DE10-R1 is closed in both directions, and closed at the root rather than at the symptom.

Two findings, neither holding: **DE12-R1** — the `stamped_at` branch asymmetry the
coordinator observed (I confirm it, and it is wider than stated; **I set it above LOW**
and say why) — and **DE12-R2**, which I did not expect to find: **`null` is not the only
open-ended spelling**, and one of the others is an empty value.

---

## 1. DE10-R1 closed, and nothing temporal is still sorted

**Scope confirmed:** the commit touches `de_ratification_check.py` and the register only.
Seam and supplier are unchanged (69 / 53 checks, matching my round-11 figures).

**Both directions, by field and value:**

| field | permissive-sorting garbage | restrictive-sorting garbage |
|---|---|---|
| `now_utc` | `'zzzz'`, `'aaaa'` → **REFUSED**, naming the field, the value and the accepted formats | `''` → **REFUSED** |
| `scope_to` | `'aaaa'` → **REFUSED** | `'not-a-date'` → **REFUSED** |
| `scope_from` | `'zzzz'` → **REFUSED** | `'not-a-date'` → **REFUSED** |
| superseder heading | `2026-99-99T99:99Z` (well-shaped, impossible) → **REFUSED**, naming the entry |

Non-strings refuse as refusals: `now_utc=123` → *"is 123 (int), not a string. A TypeError
from a comparison is…"*; `now_utc=['x']` likewise. `scope_to: null` still reads
open-ended and `20260930` still bounds.

**The direction the round could have missed — every temporal comparison in the module,
with its operands:**

| line | expression | operands |
|---|---|---|
| 433 | `d < parse_day(fields["scope_from"], …)` | datetime vs datetime |
| 440 | `d <= parse_day(to, …)` | datetime vs datetime |
| 509 | `t > stamp` | `superseder_times` values are `_norm_ts(...)`; `stamp` is `_norm_ts(...)` — **both datetimes** |
| 514 | `t <= stamp` | same |
| 633 | `day_end_instant(day) <= now_dt` | datetime vs datetime (`now_dt` is `datetime.now` or `parse_instant`) |

Verified at runtime: `_norm_ts`, `parse_instant`, `parse_day` and `day_end_instant` all
return `datetime`. **No temporal field is compared as a string anywhere in the module.**
`day_end_utc` survives only for rendering, and its docstring says so — *"nothing compares
this string any more"*. The fix is at the root (`_norm_ts` now parses, and its docstring
records that padding-and-comparing "is the DE10-R1 defect"), not at the five call sites.

## 2. DE12-R1 — the `stamped_at` asymmetry: CONFIRMED, and wider than the observation

| ref | `stamped_at` | result |
|---|---|---|
| **R-418** (superseded) | `'not-a-time'` | **REFUSED** by name |
| R-418 | `123` | **REFUSED** — "(int), not a string" |
| R-418 | `''` | **REFUSED** |
| **R-419** (not superseded) | `'not-a-time'` | `verified True`, emission carries `'not-a-time'` |
| R-419 | **`123`** | `verified True`, **emission carries `123`** |
| R-419 | `''` | `verified True`, emission carries `''` |

So it is not only unparsable strings: a **non-string** stamp is accepted and echoed on the
un-superseded branch, while the identical value refuses on the other. `stamped_at` is
parsed at `:501-508`, inside the superseded branch, and nowhere else.

**Severity: I read it above the coordinator's LOW — call it MEDIUM-LOW, low impact today
and medium in the failure mode it creates.** Three reasons:

1. **It is DE10-R1's own shape, one branch over.** Round 12's achievement is that a
   temporal value is validated where it is *accepted*, not where it happens to be
   *compared*. `stamped_at` is the one field still validated where it is compared.
2. **The failure is deferred onto a future reader.** A receipt stamped today with `123`
   verifies clean and stores `stamped_at: 123`. The day a superseder appears — which is
   the day the stamp matters — that same receipt refuses. The defect surfaces against an
   artifact whose producer is gone, at the moment provenance is being audited.
3. **The cost of closing it is one call**: parse `stamped_at` when supplied, before
   branching. There is no case where accepting an unparsable stamp is useful.

Routing to DE round 13 is right; the framing should be "a stored provenance field that
the checker will later refuse to read", not "a cosmetic echo".

## 3. `now_utc_source` is honest

`now_utc=None` explicitly → `now_utc_source: wall_clock`, `day_closed True`; an injected
value → `injected`. The source records what happened, not whether a kwarg was truthy.

## 4. DE12-R2 — `null` is **not** the only open-ended spelling, and one of the others is empty

`SCOPE_OPEN_TOKENS = ('null', 'none', '')`.

| `scope_to` | reading |
|---|---|
| absent | **REFUSED — MISSING** (round 10's closure, correct) |
| `null` | open-ended, verified — the declared spelling |
| `none` / `None` / `NULL` | **open-ended**, verified — case-insensitive synonyms |
| **`` (empty after the colon)** | **open-ended, `verified True`, `unverifiable []`** |
| `~` (YAML's null) | **REFUSED** by name |

The empty case is the one that matters: `scope_to:` with nothing after it — an ordinary
editing slip — silently becomes an **unbounded** ratification, and the emission gives no
sign of it. That is the same family as DE10-R1: an ill-formed value *interpreted* rather
than refused, and interpreted on the permissive side. It is a deliberate constant rather
than an oversight, which is why I file it as a design question and not a bug: the block
format declares `null` as the open token, and an empty field is absence-in-place, which
the module already refuses when the line is missing entirely.

**Closure:** drop `''` from `SCOPE_OPEN_TOKENS` and refuse an empty value by name (the
MISSING/VALUE vocabulary already exists); keep `none` only if a synonym is wanted.

## 5. The control that ran nothing — the class is closed structurally

I walked the selftest's AST for zero-assertion shapes: no empty literal iterables, no
`ok()` inside a comprehension, and **no `if` whose body holds the only assertion** (0 of
them). Five loops, each over a non-empty literal.

More usefully, the backstop is verified rather than assumed: I emptied one selftest loop
(`for _val in ("zzzz","aaaa")` → `[]`) and the suite **failed** —
*"check count asserted at run time: 82 == 84"*. A loop that runs zero times cannot pass
silently, because the count assertion converts "ran nothing" into a failed check. That is
what closes the class, not the absence of suspicious shapes.

## 6. The audit at 19 — what the number means

19 paths, `survivors: []`. The five new ones are `unparsable_now_utc`,
`non_string_now_utc`, `unparsable_scope_to`, `unparsable_scope_from`,
`unparsable_stamped_at`.

I traced which raise site each reaches:

| audit path | raise site |
|---|---|
| `unparsable_now_utc`, `unparsable_stamped_at` | `parse_instant:170` |
| `non_string_now_utc` | `parse_instant:160` |
| `unparsable_scope_to`, `unparsable_scope_from` | `parse_day:188` |

**Five cases, three raise sites.** So "19 audited refusal paths" counts
(input, expected-refusal) pairs, not 19 independent guards — and for this class that is
the right design, not laundering. The five cases prove the shared parser is **wired at
five call sites**, which is exactly what DE10-R1 demanded; a separate raise per field
would duplicate the parser and invite the drift round 12 removed. The refusals name the
field (`now_utc`, `stamped_at`, `block.scope_to`, `block.scope_from`), so the wiring is
per-field even though the refusal is shared. Worth stating in the count's own words so a
reader does not read 19 as 19 guards.

## 7. Nothing under review moved

`R-419` on 09-01 → `verified_for_new_run True`; `R-418` stamped 10:30Z → `provenance
True`; the seam emits **1,875** specs on the real 09-01 supply; `de_admissible_windows`
**53** checks, seam **69** — unchanged from my round-11 figures. Checker: **84 / 84**
under both launchers from the repo root, rc 0.

## 8. Rule 10 / rule 14

Refusal messages compute what they print — each names the field, the value it received
and the formats it accepts, and the day-closed refusal prints the instant it compared.
The emission still carries `decides: "nothing -- this reports; admission is the
coordinator's act and accrual is decided elsewhere"`.

---

## Executed evidence

At `9dbaa5a`, 2026-09-02T12:02–12:05Z:

| check | result |
|---|---|
| scope | `de_ratification_check.py` + register only |
| suites | **84 / 84** both launchers; seam 69, supplier 53 |
| temporal operand types | `datetime` at all five comparison sites |
| `now_utc` garbage / non-string | refuse by field and value / "not a string" |
| `scope_to` / `scope_from`, both sort directions | **all refuse by field and value** |
| `now_utc=None` | `wall_clock` |
| `stamped_at` on R-419 | `'not-a-time'`, **`123`**, `''` all **accepted and echoed** — DE12-R1 |
| `stamped_at` on R-418 | all three **refuse** |
| `scope_to` empty | **open-ended, silently** — DE12-R2 |
| `scope_to: ~` / absent | refuses / MISSING |
| selftest AST scan | 0 zero-assertion shapes |
| a selftest loop emptied | **suite FAILS** on the count assertion (82 == 84) |
| audit | 19 paths, 0 survivors; the five new ones reach **3** raise sites |
| R-419 / R-418@10:30Z / seam specs | unchanged: True / True / **1,875** |
| register file | never written; worktree clean |

---

## Disposition

- **RELEASED:** DE round 12. DE10-R1 is closed in both directions and at the root — the
  parser changed meaning once and every call site followed. **No hold.**
- **FILED:** **DE12-R1** (`stamped_at` parsed only when superseded — confirmed, wider
  than observed since non-strings survive too; **MEDIUM-LOW**, and the framing matters:
  a stored provenance field the checker will later refuse to read) and **DE12-R2**
  (`SCOPE_OPEN_TOKENS` makes an **empty** `scope_to` silently open-ended).
- **Noted, not a finding:** the audit's 19 is 19 cases over fewer raise sites; for a
  shared parser that is call-site coverage and is the right design — worth saying so in
  the count.
