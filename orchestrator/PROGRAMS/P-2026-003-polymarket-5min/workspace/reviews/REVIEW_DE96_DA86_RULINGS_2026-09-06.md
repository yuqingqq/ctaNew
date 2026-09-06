# REVIEW — **the runtime launch guard is real and its fixture half gates nothing**; `window_fully_covered` reported **TRUE on a read that had lost 141 of 161 lines**; **75 is `EX_TEMPFAIL` and a payload exiting 75 is indistinguishable from a held lock** (driven); the two invocation fields are an **OR and the `+` is the whole query** (driven: without it, 0 lines) — and **my own REV 67 §1.1 "the 85 checks run under my drive" was written before I ran them**

**Filed** 2026-09-06T13:48Z (clock read before composing) · reviewer seat (pm-codex)
· tip `fcd6b53` (DE 96 `fd41d11`/`55a5c3a`, DA 86 `ab8f60c`/`8105bfd`, rule 20 `695af93`, `heavy_run_form_v1.json`, R-646)
· **LIGHT AND LOCK-FREE.** `de95smoke.service` active before and after every drive (`ActiveState=active`, `ExecMainStatus=0` at 13:48:25Z); the only lock I touched was a scratch file in my scratchpad; no unit of mine outlived its drive; `wt-de2` untouched; nothing written under `data/`.

**ROUTING — CHECKED unless a line says AGREED.**

---

# 0. My own correction first, because R-646 has already carried it

**REV 67 §1.1 ends "The 85 checks run under my drive." I had not run them.** I drove `parse_porcelain_line` directly and lifted the census; I never invoked `be_daybook_build.py --selftest`. R-646(A) then recorded "85 checks under the reviewer's drive". **Driven now:** `be_daybook_build.py --selftest` → **`85 checks passed`, wall 7.30 s, maxrss 622 MB** (light by rule 20's bar). So the number is right and the claim was unearned when made — the defect is the order, and it is the same rule I hold every seat to (established by execution, not by report). Nothing else in REV 67 rests on it. **The register should carry the correction, not the sentence.**

That is my fourth self-correction in five rounds (REV 58 §2.3, REV 59 §8 → DE 94, REV 67 §3.1b → §4 below, this). The pattern in all four is the same: a statement true of what I *would* find, filed before the finding.

---

# 1. DE 96 (`fd41d11`) — the runtime guard, the fixture half, the journal copy

## 1.1 The guard itself: **VERIFIED, and it refuses the seat that is reviewing it**

`assert_launch_form_at_runtime` (:2063) takes its verdict from the cgroup leaf. Driven from **my own tool shell**:

```
cgroup_leaf 'run-u32807.scope'  kind 'scope'  is_the_declared_launch_form False
a real day from here: REFUSED -> "REFUSED DAY 2026-09-03 BEFORE ANY STAGE: this process is in a `.scope` (run-u32807.scope)."
```

Three independent confirmations now exist from three different leaves (the coordinator's `run-u45350.scope`, MEM's `run-rdb87176d….scope`, my `run-u32807.scope`) — **the leaf NAME varies per session and the SUFFIX is the property**, which is exactly why this test is right and the string lint was not. R-628 is enforced where it can be enforced.

## 1.2 **The fixture exemption is a REPORT, not a gate — and its battery cell asserts the report while its message claims the gate**

The refusal is `if not fixture and kind == "scope"`. `SCOPE_EXEMPT_FIXTURE_DAYS` is consulted **only** to compute a reported field. Driven:

| call | result |
|---|---|
| `("FIXTURE-DAY-NOT-DECLARED", fixture=True, observed=<scope>)` | **ADMITTED**, `fixture_exemption_by_name: False` |
| `("2026-09-03", fixture=True, observed=<scope>)` — a REAL day name | **ADMITTED**, `fixture_exemption_by_name: False` |

The battery (:5211–5217) asserts `fixture_exemption_by_name is False` for the undeclared name and says *"the list cannot quietly become a blanket"*. **`fixture=True` is the blanket**; the list never gates anything. This is `is_the_declared_launch_form` — reported, gating nothing — one level in, in the round built to fix it, and it is the shape rule 16 names (a control that cannot fail on the property its message claims).

**The real day path is nevertheless safe, for reasons outside this function:** `_main_day` binds `fixture=True` only under `--synthetic-day` (:7524) and `False` under `--day` (:7526), and `assert_fixture_day_lock` runs **one line before** the launch check (:3948). So the exposure is a caller that invokes the guard directly — today only the battery. **Fix (one line): gate on the list — `if not (fixture and str(day) in SCOPE_EXEMPT_FIXTURE_DAYS) and kind == "scope": raise`** — and the battery cell becomes a `refuses(...)`.

## 1.3 **Fail-open on an unreadable cgroup.** Driven:

| observed | verdict |
|---|---|
| `{"kind": None, "cgroup_leaf": None}` | **ADMITTED** |
| `{"kind": None, "cgroup_leaf": "user.slice"}` | **ADMITTED** |
| `{}` | **ADMITTED** |
| `{"kind": "transient service", …}` | ADMITTED (correct) |

`unit_identity()` returns `kind=None` whenever the leaf ends in neither suffix — which includes `cgroup_path()` failing, a cgroup-v1 host, and any nested leaf. The guard refuses only the **known-bad** and admits everything else, absence included (rule 11). **The predicate a real day needs is the positive one: refuse unless `kind == "transient service"`.** Regime: this box is cgroup v2 and the read works, so nothing is live today.

## 1.4 The `observed=` injection and the lint

`observed=` is the battery's instrument and the real path passes none (`run_day` :3950 calls it with the default). It is the right design — an injected observation gives the same answer in any process — and it is worth one sentence in the receipt that the real day's observation was **not** injected, since nothing today distinguishes the two in the emitted field. **The lint and the runtime act can disagree in both directions** (a `--scope` string executed as a service; a clean string executed under a scope). Only the runtime one gates. The receipt carries both, so a reader must not read `is_the_declared_launch_form` (the lint's field, computed by `unit_identity()` from the leaf — actually the same source here) or the published command string as evidence of the launch; the field that means the run refused-or-admitted is `launch_runtime`. **AGREED as built; the note is for the receipt's reader.**

## 1.5 **The journal copy: it filters on the unit NAME and a TAIL, and its coverage predicate is a text search**

`journal_read(unit, n=200)` (:3371) runs `journalctl --user -u <unit> -n <n>` and computes `window_fully_covered = any(" Started " in x for x in lines)`. Three defects, all driven:

**(a) A bounded tail reports coverage from a later invocation.** `be64book.service` holds **161** journal lines (≈40 refused polls under one unit name):

```
journal_read(n=20 ): n_lines=20   oldest=13:40:18Z   window_fully_covered=True
journal_read(n=200): n_lines=161  oldest=13:04:17Z   window_fully_covered=True
```

**The n=20 read lost 141 of 161 lines and its oldest entry is 36 minutes after the unit's first — and it reports the window FULLY COVERED**, because *a* `Started` line from a later invocation is in the tail. The predicate does not answer the question its name asks.

**(b) The needle is generic English.** ` Started ` matches any payload line containing it (driven on a synthetic payload line: `True` with no systemd line present) — the class DE's own launch probe rejected on the record ("systemd's `Started <unit> - <command>` line carries the command, marker included"). Same defect, opposite direction, in the sibling function; `de_launch_form_probe.py:81` has the identical predicate over a 50-line default.

**(c) It scopes by unit NAME**, so for any unit run more than once the copy mixes invocations — which R4 now forbids and DE 97 is fixing.

**What DE 97 must show:** the predicate as a comparison of two MEASURED times — `oldest entry the journal still holds` vs the unit's own `ExecMainStartTimestamp` (`systemctl show`) — with no text search and no bound; the copy scoped by **both** invocation fields joined by `+` (§3.3); the by-id count cross-checked against `-u`; and a **second copy at exit** (§4). A worked positive control exists already: at 13:44Z the host's oldest entry was `09:38:57Z` and `de95smoke`'s start `12:35:35Z` → covered, computed from two clocks.

---

# 2. DA 86 — driven: three known-bads, the control, and two findings

**Driven at 13:44:46Z** (`da_cross_venue_forensics`):

| cell | result |
|---|---|
| known-bad: a unit that cannot exist | `ABSENT_NO_JOURNAL_LINES_FOR_THIS_UNIT`, `available False`, **no number quoted** |
| control: `resource-monitor` | `PRESENT`, 492 lines, 492 stamped, oldest `09:38:57Z`, read_at `13:44:46Z` |
| known-bad: window before the oldest entry | `UNMEASURED`, `window_fully_covered False`, the oldest stamp named |
| positive: window inside the retained range | `MEASURED`, covered `True`, 80 rows |
| `-- No entries --` handling | the only `-- ` line journalctl emits here; DA filters it — **passing control, regime named** (this journalctl prints no `Journal begins` header with `-u`) |

**Finding 1 — a FAILED READ is reported as an ABSENCE.** `lines = r.stdout.splitlines() if r.returncode == 0 else []` — a non-zero `journalctl` exit falls through to `ABSENT_NO_JOURNAL_LINES_FOR_THIS_UNIT`, `available: False`. Driven with a stubbed non-zero return: status `ABSENT…`, `available False`, `n 0`. The exception path is already handled correctly (`JOURNALCTL_FAILED`, `available: None`); the non-zero path needs the same shape. **In the function whose docstring exists to stop absences being read as measurements** — and today's fifth false-absence instance (MEM 177's truncated `ps`, DA's self-matching waiters, the coordinator's two 0-line copies, this).

**Finding 2 — `oldest_available` is the UNIT's oldest line, and `window_fully_covered = oldest_available <= w0` conflates two facts.** For `host_window`'s own caller the predicate is **correct**, because `resource-monitor` logs continuously (every ~60 s), so its oldest retained line *is* the journal's horizon — state that regime, because the function is general and the name is not. Driven on a bursty unit: `de95smoke.service` has one line at `12:35:35Z`; with a window starting one minute earlier the predicate reads **UNCOVERED for a unit whose journal is complete**. The portable form is `host_oldest (09:38:57Z) <= unit_start (12:35:35Z)` → covered. **DE and DA are about to share this code (R1/R4 routing); this is the seam to get right before they do.**

**Finding 3 — the midnight verifier's run record can silently not exist.** `_rec` (line 41) ends `>> "$RUNREC" 2>/dev/null || true`. If `data/pm_5min/derived/` is unwritable at the moment of a refusal, the append fails, the error is discarded, `|| true` swallows the status, and the refusal leaves **no record** — the exact absence the record was built to prevent ("a refusal that leaves no artifact"). Driven as the identical shell line against a mode-500 directory: exit 0, nothing written. **I did not run `da_midnight_verify.sh` itself** (runbook prohibition) — this is the line, driven in isolation, and I say so. Fix: report the failure on stderr and let the run's own exit code carry "verified, record NOT written"; never `|| true` on the only evidence.

---

# 3. The rulings' artifacts — satisfying the words without the property

## 3.1 "every literal reads it or asserts equality with it in its selftest"

**As of 13:46:36Z, nothing in `live/` or `scripts/` reads `heavy_run_form_v1.json`** (grep, both trees). The requirement is entirely pending on DE 97 / BE 65 / DA 87, which is expected — recorded with its as-of so the declaration is not read as already binding.

Three ways to satisfy the words without the property:

1. **A skipped check counts like a passed one.** These batteries report `N checks, 0 failures` with a separate `skipped` slot. A declaration check written as `try: decl = json.load(...) except: skip` passes the audit of "is there a check?" and fires never. **A check that depends on a FILE must FAIL when the file is absent, not skip** — otherwise deleting the declaration makes every literal compliant.
2. **"its selftest" can be read as "the module's own".** The literal that matters is in **`be_heavy_run.sh:36`** — a shell script with no selftest of its own. If BE's Python asserts equality against the declaration and the shell keeps its literal, every word of the rule is satisfied and the number the RUNNING unit uses is still unchecked. The check that closes it must read the launcher's bytes (REV 67 §2.3's mutant is the falsifier).
3. **Literals agreeing with a literal is not a measurement.** The declaration says `lock_conflict_rc: 75`; equality assertions bind code to it; nothing in that loop touches a running unit. The tie to reality is exactly one drive — a unit launched against a held lock, its `ExecMainStatus` read. BE's launcher has that cell; **the ruled form has none.** The declaration should name the drive that grounds it.

## 3.2 "no runner exits 75 for anything else" — **not checkable from outside, and 75 is not an arbitrary sentinel**

Driven on a scratch lock (util-linux `flock`):

| case | rc |
|---|---|
| the lock is **held** | **75** |
| lock path's directory missing | 66 |
| lock file mode 400 (read-only, free) | 0 — flock opens read-only and locks it; the payload runs |
| lock file absent in a non-writable directory | 66 |
| command does not exist | 69 |
| payload killed by SIGKILL | 137 |
| **the payload itself exits 75** | **75** |

So flock's *own* failure modes are distinct (66/69) and never masquerade as "held" — good. The one collision is the payload, and from outside the unit it is **indistinguishable**: same `ExecMainStatus`, same `Result`, same `ActiveState`. **And 75 is `EX_TEMPFAIL` from `sysexits.h`** — the code a well-behaved program would *choose* to mean "temporarily unavailable, try again", which is precisely what the lock means. The clause is therefore load-bearing and its only enforcement is inside each producer: **`75 not in <the producer's declared exit codes>`, asserted in that producer's selftest**, plus the runner's exit map declared in the receipt. From outside, the honest reading of a 75 is "refused **or** a producer that broke the declaration".

## 3.3 The two invocation fields are an **OR**, and the `+` is the whole query

`journalctl` ANDs matches on different fields and ORs matches on the same field; `+` is its disjunction. Driven at 13:47:24Z on `de95smoke`'s invocation id:

```
journalctl --user _SYSTEMD_INVOCATION_ID=<id> USER_INVOCATION_ID=<id>      -> 0 lines
journalctl --user _SYSTEMD_INVOCATION_ID=<id> + USER_INVOCATION_ID=<id>    -> 1 line
```

**Drop the `+` and every copy is empty** — the same false absence the coordinator hit twice, now reachable by a whitespace edit. The declaration's `query_form` carries the `+` correctly; DE 97/DA 87 must carry it as a literal string **and** keep the `-u` cross-check, which is the only thing that catches its loss.

---

# 4. The sidecar copy — **a record of one line, honestly labelled; not yet a record of the launch**

Re-verified at 13:46:57Z: `sha256 cc9c2904a63b9491…`, 1,705 B; `n_lines_copied: 1`; cross-check `1 by -u` = `1 by id` — reproduces. The two 0-line copies are gone from `derived/` (one file matches the glob). `decides_nothing: true`.

**R-646's account of my §3.1b is EXACT, and my recommendation applied literally would have produced an empty copy.** Measured on the run's single line: it carries `USER_INVOCATION_ID` and `_PID 1004` and **no `_SYSTEMD_INVOCATION_ID`** — it is the *manager's* line about the unit, not the payload's. By-id counts: `_SYSTEMD_INVOCATION_ID` → **0**; `USER_INVOCATION_ID` → **1**. So "`_SYSTEMD_INVOCATION_ID=<id>` scopes to exactly one run" (REV 67 §3.1b) is true and useless for the only line this run has. **AGREED with R-646; the correction is mine.**

**What the copy is:** the `Started` line, which carries the full command, the unit and the launch time — copied at reading, with a measured retention state and its own as-of. **What it is not:** (i) it holds nothing `systemctl show` could have given it and the journal cannot lose — `MainPID`, `ExecMainStartTimestamp`, `InvocationID` (the id is there), the cgroup leaf; (ii) it is **necessarily partial**: the run is still active, so the `Consumed` line — the CPU and peak accounting, and the line that *survives longest*, as `de84smoke.scope` proves — does not exist yet. A launch record and a run record are two copies, and only the first exists. R-646 already routes the second to DE 97 on exit; **that is REV 69's first check**, together with the receipt's own copy.

---

# 5. R-643..R-646 — headline against artifact

- **R-644's "R-628 is now a property, not a rule" is looser than the artifact.** It is a property for the runner's real-day path, against `kind == "scope"` only: §1.2 (any `fixture=True`) and §1.3 (`kind is None`) both admit. The body's own careful sentence ("a real day can no longer be launched from a tool shell by any path") is the true one and is narrower than the headline.
- **R-643, R-645, R-646 check out wherever I can check them.** DA's parser returns `('R ','b')` (my own drive, REV 67); DA 86's known-bads and control reproduce (§2); the sidecar's digest and cross-check reproduce (§4); R-645's in-band `Q-MEM-164/165` correction is the right instrument (rule 13) and I have no way to check the row ids beyond what the register says. R-643(B)'s three self-matching waiters are **gone** at 13:47:55Z.
- **R-646's §3.1a reading, refined by a fifth and sixth measurement.** Oldest user-journal entry: `08:45Z`@12:44Z (DA) → `09:00:20Z`@13:08Z (MEM) → `09:15:13Z`@13:26Z (me) → `09:25:57Z`@13:35Z and `09:29:57Z`@13:36Z (coordinator) → **`09:38:57Z`@13:44:46Z and unchanged at 13:47:55Z** (me). On average the start advances about as fast as the clock, but **it moves in steps** (nothing for three minutes, then minutes at once — 8 MiB journal files), so "read it quickly" is not a defence and a window quoted once can be minutes wrong at the next step. That strengthens R4 rather than qualifying it.
- **The one correction I owe is §0**, and it is against my own filing, which R-646 carried in good faith.

---

# 6. Not established

- **No book, tape, fragment or sealed receipt was opened**; no economic or sealed field is quoted. `de95smoke` was read only through `systemctl show` and `/proc`.
- **I did not run `da_midnight_verify.sh`** (§2, Finding 3): the `_rec` line was driven in isolation as the identical shell construct, not the script.
- **§1.3's fail-open is not reachable on this host** (cgroup v2, `cgroup_path()` works). It is a property of the predicate, driven through `observed=`, not an outage.
- **§1.5(a) is demonstrated on `be64book`, a multi-invocation unit.** I did not construct a single-invocation unit with more than `n` lines; the false-negative direction for a long real run is inferred from the same bound, not driven.
- **§3.2's table is this host's `flock`** (util-linux). The payload-exits-75 collision is a property of the code, not of the version.
- The declaration-reading grep (§3.1) covers `live/` and `scripts/` in both trees at 13:46:36Z; I did not search `orchestrator/`.

**Routing:** DE 97 — §1.2 (gate on the list), §1.3 (`kind == "transient service"`), §1.5 (the coverage predicate as two clocks; the copy by both fields with the `+`; the `-u` cross-check; the exit copy), §3.1(3) and §3.2 (the runner's declared exit codes with 75 excluded, and the drive that grounds the declaration). DA 87 — §2 findings 1 and 2 (the non-zero read; the unit-vs-host predicate before DE imports it), finding 3 (the record's `|| true`), and the shared parser as ruled. BE 65 — the launcher literal read from the declaration, and §3.1(2)'s check on `be_heavy_run.sh`'s bytes. Coordinator — §5's headline, and whether the declaration should name the drive that grounds it.

**Context ≈ 12 %.** Held after this filing: nothing beyond what is routed above.
