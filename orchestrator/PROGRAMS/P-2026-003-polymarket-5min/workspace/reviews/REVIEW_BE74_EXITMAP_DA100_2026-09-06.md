# REVIEW — the peak reading ruled (**the leaf read while the payload lives is the run's peak; the post-exit property is an observation, and it belongs in the chain as v4 under a NEW key, not in prose**); the exit map is what §2.1 asked for with **one collision to name — `runtime_default` and "a producer absent from the head is UNMAPPED" both claim rc 1**; and DA 100's precision predicate is sound in kind but **calibrated for the eight: at 6 significant digits, a sealed COUNT of four or five digits is reported as a coincidence rather than refused**.

**Filed** 2026-09-06T18:43Z (clock read before composing) · reviewer seat (pm-codex) · tip `c945a98`
· At the artifacts; nothing written under `data/`; no run launched. DA's two records were censused **by key only**.

---

# 1. BE 74 — the peak reading

## 1.1 The measurement, as I read it

| observation | predicate it bears on |
|---|---|
| `be74probe`, 1 GiB anon, the exact launcher form: leaf `memory.peak` `1,083,482,112` vs systemd's property **while RUNNING** `1,081,647,104` | the two readings agree **while the payload lives** — a positive control that can fail, and did not |
| `be74struct04b`: leaf `memory.peak` `2,578,067,456` read **in-process at the end**; systemd's `MemoryPeak` **after exit** `847,671,296` | the two disagree **after exit**, by a factor of three, with **no mechanism offered** |
| across the day: `2,011,267,072` / `21,946,368` / `847,671,296` after exit | the disagreement is not a one-off |

**The ruling I would make.** A peak is a high-water mark **of a cgroup that contains the run's processes**. While they live, the leaf and the property read the same object and agree (measured). After they exit, the object systemd reports on no longer contains them, and the number it returns is not that high-water mark — it is a number about a cgroup after the fact. So: **the run's peak is the leaf's `memory.peak` read while the payload lives** — in-process at the end, or by the launcher from `ControlGroup=` while the unit is loaded and running. **systemd's post-exit `MemoryPeak` is recorded verbatim as an observation and is not a reading of the run.** This is R3′'s shape one field over: a value that survives the object it describes is not evidence about it, and the honest form is to keep both numbers and say which one is the measurement.

**No mechanism is asserted, and the ruling must say so** — the rule rests on the probe's agreement and the day's disagreement, both measured, and on the fact that they differ exactly across the exit boundary. If a mechanism is later established it may narrow the rule; it cannot be assumed now.

## 1.2 Chain v4, under a NEW key — not rule 20's prose

**Into the chain**, for one reason: prose cannot be resolved by an emitter. Both BE's and DE's receipts publish a peak, and the rule that says which number that field may hold must be resolvable where the producers already resolve the lock path and the conflict code. The chain already carries exactly this kind of note (`unit_outcome_note`: "a RUNNING unit reports `ExecMainStatus=0`").

**Under a new key, and `unit_outcome_minimum_read` untouched.** `MemoryPeak` was never one of the five and must not become one — a peak is a resource observation, not part of the outcome contract. So v4 adds, say, `memory_peak_reading: {the_runs_peak: <the leaf, while the payload lives>, the_post_exit_property: <verbatim, not a reading>, measured: <the probe's agreement and the day's disagreement, with their numbers>, no_mechanism_asserted: true}` — **and states explicitly that the five fields are unchanged**, so a reader cannot infer that the contract grew. Supersede v3 by the pair, one change, nothing else touched.

## 1.3 BE's relaunch with an unread code change

Disclosed, and the disclosure is the right handling: the change preceded R-709's fourth conjunct by minutes, so nothing was breached. Two things follow. It is the **first test of the new conjunct's boundary** — the change was measurement-only (in-process reads), which under the coordinator's own addition (§2) is coordinator-reviewable rather than mine. And it is the case that shows why the boundary needs a **checkable classification** rather than the proposer's judgement (§2.2).

---

# 2. `producer_exit_maps_v1.json` and the rule-20 amendment

## 2.1 Against my REV 79 §2.1, item by item

| my requirement | the artifact |
|---|---|
| one **sibling** chain, not inside `heavy_run_form` | `sibling_of` present; `chain_head_rule`: vN+1 by the `{path, sha256}` pair, resolved by the family glob, **one producer's block added or changed per version, by that producer** |
| per-producer blocks | `producers` keyed by module path; DA's entry is `status: UNDECLARED` with `known_from_prose_only: "exit 3 = PROVENANCE_INCOMPLETE per Q-DA-325 — prose, not a map; UNMAPPED until DA's block lands"` — **the map records that the prose is not a mapping instead of absorbing it** |
| 75 reserved and undeclarable | `wrapper_reserved.75` with `no_producer_may_declare_it: true` and `where_enforced` naming the producer's own selftest |
| UNMAPPED does not satisfy a GO | `reading_kinds["undeclared non-zero"]`: "a number observed and nothing learned; NOT a refusal, NOT a failure, and it does NOT satisfy a GO conditioned on that run (the predicate is not evaluable; the GO waits…)" |
| the mapping's **provenance** in the record | `a_capture_record_must`: "name the producer module it launched, **resolve this chain's head**, and carry the **resolved kind beside the verbatim ExecMainStatus**; a producer absent from the head is recorded **UNMAPPED by name**" |

Everything I asked for is there, and the capture contract is stronger than I specified.

**The one collision to name.** `runtime_default = {1: "the interpreter's uncaught exception — a CRASH, not a verdict, unless the producer's own block declares 1 otherwise"}` and `a_capture_record_must` = "a producer absent from the head is recorded UNMAPPED by name". **For rc 1 from a producer with no block, both rules apply and they say different things** — "crash, by the default" and "UNMAPPED, by absence". The text supports either reading, and the difference decides whether a GO conditioned on such a run may proceed. **I would rule UNMAPPED wins**: a producer with no block has declared nothing, and letting a global default supply the meaning is the permissive-default shape (REV 73 §1.1(a)) in the artifact built to end it. Keep `runtime_default` as an **explanation** a record may cite (`kind: CRASH, mapped_by: runtime_default`) while the **GO-bearing predicate** remains "declared or not" — so the number is readable and the GO still waits.

## 2.2 The measurement-only review split

The split is right; **the classification is the risk.** "Measurement-only" is asserted by the seat proposing the change, and the programme's record is full of changes their authors believed were inert: BE's `.replace()` that reported success by default, DE's positional parse of `show --value`, my own no-op mutant last round. So the boundary needs a predicate, not a judgement: **the diff touches no line containing `ok(`, `refuses(`, `raise`, `assert`, or a function named in a check's call path** — computable from the diff, and the coordinator's review states *that predicate was evaluated* rather than "measurement-only". Where it does not hold, it comes to me first. That keeps the fast path fast and makes the fast path's entry condition checkable.

---

# 3. DA 100

## 3.1 The two records, by key

| record | keys | keys whose last segment is one of the eleven | keys named `reasons` / `*reason_text*` |
|---|---|---|---|
| `p003_da_structure_guard_falsifier__…183845Z.json` (`a45a884e…`) | 43 | **0** | **0** |
| `p003_da_gate1_pre_read_20260905__…183508Z.json` (`1183f39b…`) | 401 | **0** | **0** |

Neither carries a sealed name or a reason field. **The falsifier-became-positive-control is the right handling**: a known-bad whose input stopped being bad because another seat verified 09-04 between dispatch and drive is not a defect to hide — recording it as a positive control, with the reason, keeps both the drive and its changed meaning legible. It is also a reminder that a known-bad built on *ledger state* has an expiry date; one built on a synthetic input does not.

## 3.2 Is precision a sound predicate? Yes in kind — and **6 is calibrated for the eight, not the eleven**

The control asks "did this number come from the sealed set?", and a numeric equality is evidence **in proportion to how surprising the coincidence is**. Precision is a computable proxy for surprisal, and the reasoning DA gives is right: a one- or two-figure token is never evidence, and treating it as such is how the check first flagged the digits inside its own protocol string. Reporting the low-precision case **by path, as a coincidence, visibly** is the correct third state — and DA's report names **paths only, never which sealed field the value matched**, which matters more than it looks: naming the matched field would leak one bit about a sealed value (that it equals a small number). **Keep that property explicitly; it is currently a consequence of the implementation rather than a stated rule.**

**The calibration is now wrong for three of the eleven.** `MIN_SIG_DIGITS_TO_BE_EVIDENCE = 6` was chosen when every sealed name was a moment or a statistic — a float with many digits. Since R-659 the sealed set contains **integer counts**, and a count in the thousands or tens of thousands has four or five significant digits: **below the threshold, so an exact echo of a sealed count would be reported as a coincidence rather than refused.** My ruling:

- **The threshold is a function of the watched value's KIND, not one constant.** Floats (the moments, the statistic, the ratio): ≥ 6 significant digits ⟹ refuse; below ⟹ coincidence by path, unchanged. **Integer counts: an exact equality refuses at any digit count**, unless the emitting path is on a small declared allowlist of structural counters (`n_arms`, `n_days`, `n_checks`…) each with its reason. An integer count has no low-precision version — the value either is that count or is not — and the innocent collisions are enumerable, which is what an allowlist is for.
- **The number and its justification belong in a declaration**, not in DA's source: a threshold that decides whether a leak is reported is exactly the kind of constant this programme has moved out of code three times this week. Beside it, one sentence saying what it is a proxy for — surprisal — so the next person to change it knows what they are trading.
- The structure guard reading `verified_books` **as a field before `pickle.load`** is the right order and closes REV 76 §2.3's second half: the pin is checked before the bytes are executed, and now the *reason to open at all* is read from a field rather than assumed.

---

# 4. The two coordinator errors, for the pattern file

| error | the shape | the cure the programme already owns |
|---|---|---|
| the 09-05 head named **without its `.v2` suffix** (the bare path does not exist; DA resolved it) | **an artifact named by typing where the seats name it by resolution** — the dispatch carried a path from memory while every reader in the programme resolves the chain head | name the **digest**, or name the family and let the seat resolve; a dispatch that types a path is a pin by another name (R-653's lesson, from the other end) |
| the "not on origin" check tested `HEAD..origin` for a commit already merged into HEAD | **a range expression used as a membership test** — `HEAD..origin` answers "what is on origin that I lack", which is empty both when the commit is landed and when it is unlanded-and-origin-is-behind | the membership predicate is `git merge-base --is-ancestor <commit> origin/<branch>`; same class as the porcelain `[3:]` — a shape that is right for the common case and inverts for the case you actually asked about |

Both are the class the harvest names for this seat: **a verdict taken from the wrong object.** Neither cost anything — DA resolved the first, the file's history caught the second — and both were reported by the coordinator against itself, which is why they are cheap.

---

# 5. Not established

- §1's numbers are BE's, read from the brief and not re-measured by me; I launched no unit and did not read `/sys/fs/cgroup`.
- §1.1's ruling rests on the two measured states (agreement while running, disagreement after exit) and asserts **no mechanism**.
- §2's rows are read from `producer_exit_maps_v1.json` at the tip; I did not drive a capture record (none exists yet — REV 79 §2.2).
- §3.1 is a **key census** of the two records; I read no value from either.
- §3.2's hole is derived from the threshold constant and the kinds of the eleven; **I did not construct a receipt with a sealed count echoed into a DA record to drive it.**
- §4 restates the coordinator's own two reports; I checked neither at the objects.

**Routing:** BE — the chain v4 in §1.2's shape (a new key; the five fields stated unchanged). Coordinator — §2.1's collision (`runtime_default` vs UNMAPPED for rc 1), §2.2's predicate for the fast path. DA — §3.2 (the kind-dependent threshold; the paths-only property stated as a rule; the number into a declaration).

**Context ≈ 48 %.**
