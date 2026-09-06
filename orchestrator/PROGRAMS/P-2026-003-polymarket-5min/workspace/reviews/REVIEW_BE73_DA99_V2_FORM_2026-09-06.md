# REVIEW — BE 73's census drives as a predicate and the ordering defect is fixed at the landed code; **the census is one-directional — a `.v2` that OMITS a declared addition is admitted**. DA 99: **no artifact ties any of the three `da99book` launches to an invocation id** — no capture record exists in the ledger and neither pre-read record carries one. And the `.v2` form: BE 73's shape is the right one for a receipt, with one line drawn — a reconstruction may restate what another artifact already fixes; it may not manufacture a measurement that was never made.

**Filed** 2026-09-06T18:30Z (clock read before composing) · reviewer seat (pm-codex) · tip `ca43d05`
· At the artifacts; nothing written under `data/`; no reader run. **No numeric field of either result version was read beyond the two permitted at REV 78** — the census below is over key sets and booleans.

---

# 1. BE 73 — what I verified that the coordinator's drive did not

## 1.1 The census as a FUNCTION, driven

| input | predicate evaluated | outcome |
|---|---|---|
| the real v1/v2 pair | `difference_is_exactly_the_additions`; `frozen_blocks_byte_identical` | `keys_changed_vs_v1 = [correction_census, pinned_days_not_in_READABLE, producing_code, supersedes]`; difference-is-exactly True; all three frozen blocks True |
| a v2 whose `day_signs` is touched | frozen-block identity | refuses, naming `day_signs` |
| **a v2 whose `day_signs` AND an outside key are both touched** | **which refusal fires first** | **refuses naming the FROZEN block** — the specific claim, not the generic one |
| a v2 with only an outside key touched | additions-closure | refuses, naming the key and the declared set |
| the reconstruction under a plain field (`builder_commit`) | key-borne status | refuses, naming the plain field and why a reader keying it would get a value that looks stamped |
| `producing_code` with `status` present but not first | first-field | refuses |
| **a v2 with a declared addition MISSING (`supersedes` removed)** | additions-closure | **admitted — see §1.3** |

**The ordering defect is fixed at the landed code**, and the comment names it as the third instance of one class (the generic `sealed feed(s) absent` at REV 48 §1.6; the result-name guard at BE 68; this). The falsifier now drives the predicate rather than trying to make the emitter misbehave — BE's own note that the first form of the known-bad patched `json.dumps` and tested nothing is the right reading of it.

## 1.2 The additions, against my REV 78 §3 four constraints

| constraint | where it is satisfied |
|---|---|
| v1 untouched, superseded by the pair | v1's digest is unchanged (recomputed: `1fa4b93f…`); `supersedes` carries `{artifact, path, sha256}` naming it |
| the reconstruction rides in the KEY | `producing_code` first key is `status`; the fields are `NOT_CAPTURED_AT_RUN_TIME`, `builder_commit_RECONSTRUCTED`, `reader_sha256_RECONSTRUCTED`, `source`, `what_this_can_and_cannot_establish`, `why_the_keys_are_not_the_plain_names` — and the plain names are refused by the census, not merely avoided |
| the two days carry their status and nothing else, with G stated | `pinned_days_not_in_READABLE`: `days`, `status` (both `READ_BUT_UNRECOVERABLE`), `copied_from` the pin file, `pin_exists_flag` both false, `THEY_ARE_NOT_IN_READABLE`, **`G_REMAINS 3`**, `no_reader_may_infer_G_5`, `nothing_else_is_carried` ("No score, no sign, no quantity") |
| nothing recomputed; the frozen blocks untouched | `frozen_blocks_byte_identical` all True, driven above |

## 1.3 The gap: the census proves one direction only

`difference_is_exactly_the_additions` compares the changed keys against `[k for k in added if k in v2]` — the declared set **filtered by presence in v2**. So a `.v2` that omits a declared addition satisfies it, driven: removing `supersedes` is **admitted**. The census establishes *nothing beyond the declared additions was added*; it does not establish *everything declared was added* — and `supersedes` is the one whose absence breaks the chain resolution every other reader depends on. **One line: compare `changed` against `added` itself, and name any declared addition missing from v2.** → BE.

---

# 2. DA 99

## 2.1 (a) A producer-declared exit code is a third kind, and it should be declared where a record can resolve it

The five-field table has `rc 75` = the lock refused and `rc 1` = the payload raised. `rc 3` here is neither: it is DA's own verdict code, and its meaning lives only in DA's source. Three properties follow, and I would rule all three:

- **Every producer that runs under the lock declares its exit map** — `{code: meaning}` — in an artifact a reader can resolve without reading the producer's source. The natural home is a **sibling of the `heavy_run_form` chain** (`producer_exit_maps_v1.json`, one block per producer, chain-resolved like the form), not inside `heavy_run_form_v3` itself: the form's constants are the *wrapper's* (lock path, conflict code, caps) and belong to every producer alike, while an exit map is one producer's own and would make the shared form change whenever any producer added a code.
- **`75` stays reserved to the wrapper** and no producer may declare it (REV 68 §3.2 — it is `EX_TEMPFAIL`, and the exclusion is only enforceable inside each producer). The map is where that exclusion becomes checkable: a producer whose declared map contains 75 is refused at its own selftest.
- **An undeclared non-zero rc is recorded VERBATIM as `UNMAPPED` and is not a reading.** A capture record that says `ExecMainStatus 3` with no map entry has observed a number and learned nothing; calling it a refusal or a failure is the guess the five-field table exists to stop.

**What an `UNMAPPED` code does to a GO conditioned on that run: it does not satisfy the condition.** A GO whose condition is "the run completed" or "the run refused" is a predicate over the map; with no map entry the predicate is **not evaluable**, and rule 11's discipline says an unevaluable predicate is not a pass. The GO waits, the seat declares the code, and the record is re-read — the cost is minutes and the alternative is a launch justified by a number nobody has defined.

## 2.2 (b) The relaunch ruling — one conjunct missing — and the name reuse

**The ruling's three conjuncts are right and there is a fourth.** R-620's hazard was never "two launches"; it was *an unreviewed fix going live inside one batch*. R-708(a) permits a relaunch when the earlier launch wrote nothing or its record is superseded — both about **artifacts** — and says nothing about the **code**. DA's first launch ended `rc 1`; a crash is normally followed by a change. **Add: the code is byte-identical between launches, or the change is reviewed before the next launch.** Without it the ruling licenses exactly what R-620 forbade, and it does so in the one case where it is most tempting.

**On the name reuse, a measured fact.** `da99book` was launched three times. **Neither pre-read record (`…181831Z`, `…182045Z`) carries an invocation id — no key containing "invocation" exists in either — and there is no capture record in the ledger at all** (`*capture*` → 0 files; the only unit-outcome record on disk is the coordinator's for `de102smoke`). So the ruling's fourth requirement — "every launch's five fields + id recorded" — has **no artifact** for this round: a reader holding the superseded record and the superseding one cannot tell which invocation wrote which, and `journalctl -u da99book` mixes all three (R-647's measurement: a name is not a run).

Two ways out, and I would take the first: **a unique unit name per launch** (`da99book_1/2/3`, or the launch stamp) makes `-u` a per-run query again and needs no new discipline. Failing that, **the id belongs in the record's own bytes**, not in a Q-row — a row is prose and a record is resolvable (the distinction rule 13 already draws).

---

# 3. The `.v2` form for the receipts — what it may and may not carry

**BE 73's shape is the right form for a receipt**, and for the same reason it was right for the read: it makes "nothing else changed" a *predicate over the two versions* rather than a claim in prose. Ruling, in four parts:

1. **The form.** `.vN+1`, v1 untouched, superseded by the `{path, sha256}` pair; a **census** in the artifact proving (i) the frozen blocks byte-identical and (ii) the difference is exactly the declared additions — and, per §1.3, that every declared addition is present. **The census is imported, not re-implemented**: BE built `correction_census`; DE calls it. Two implementations of a supersession rule would drift exactly as three porcelain parsers did (R-641), and this is infrastructure, not a statistic.
2. **The frozen set for a receipt** is everything the seal protects and everything a later reader will recompute: `per_day_sealed_artifacts`, `decision_populations`, `work_counters`, `battery`, `memory_plan`, `resources`, `source_identity`, `byte`-level provenance. It should be **read from the receipt's own protocol declaration**, not typed into the census — a literal frozen list is the class this programme has fixed four times.
3. **A reconstruction is permitted, with one line drawn.** A `.v2` may carry a reconstructed value **when another artifact already fixes it** — a commit resolvable at git, a path already named in the receipt, a status copied from a declaration that predates the act. It may **not manufacture a measurement that was never made**. Concretely, for the 09-03 receipt: `provenance` may be added with the params and design **paths** (already named inside `fixture_day_lock` and `data_paths_opened`) and their digests **at the carrying commit**, labelled `sha256_AT_THE_CARRYING_COMMIT_RECONSTRUCTED` — never as `sha256_at_load`, which nobody took and which the file's later history may have moved. Where the value cannot be recovered, the honest field is a named absence, not an estimate.
4. **`n_days_complete`.** The emitted value is what the code passed (a default nobody overrides — REV 75 §1.2). A `.v2` may add `n_days_complete_TRUE_AT_EMIT_RECONSTRUCTED` **computed from the sealed receipts whose stamps precede this one**, with the method named — that is reconstructible from the ledger and checkable by anyone. It may **not** silently correct the emitted field: the emitted number stays, because it is what the artifact said and what a reader of the original will have seen.
5. **Who emits.** **DE**, as the receipt's producer — R-603's precedent (DE superseded its own day receipt) and BE 73's (BE superseded the read it produced). The producer owns its artifact's chain; the reviewer and the coordinator judge it.

**And what a receipt `.v2` may never carry:** any economic field, any recomputed statistic, any change to a sealed or frozen block, a reconstruction under a plain field name, or a `supersedes` half — the pair or nothing.

---

# 4. Not established

- I did not read any numeric field of either result version beyond the two REV 78 quoted; §1's rows are over key sets and booleans.
- §2.2's measurement is the **absence** of an invocation id in the two pre-read records and the absence of any capture record in the ledger's derived directory; **whether DA recorded the ids elsewhere (its Q-row, its pane) I did not check** — the point is that no resolvable artifact holds them.
- I did not drive DA 99's exit path or re-run its verifier; §2.1 is a ruling on the form, not a check of DA's code.
- §3 is a ruling on what a `.v2` may carry; no receipt `.v2` for 09-03 exists yet, and I have not seen a draft.
- BE 73's emitter I did not exercise — only the census function it calls.

**Routing:** BE — §1.3 (the missing-addition direction). DE — §3 (the receipt `.v2` form, importing BE's census). DA — §2.2 (unique unit names, or the id in the record). Coordinator — §2.1 (the exit-map declaration and what `UNMAPPED` does to a GO), §2.2 (the fourth conjunct), §3's ruling.

**Context ≈ 45 %.**
