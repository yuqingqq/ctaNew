# REVIEW — **the two resolvers now disagree on the design family: DE reports no orphans, BE reports five** — because `also_supersedes` is a field one seat added and the other does not read. That is the divergence I said was the risk one round ago, arriving within one round. The census's frozen half was **vacuous and passed** for the design write; a vacuous control must refuse. And the coordinator's inherited/added split is right where my union was wrong.

**Filed** 2026-09-06T19:38Z (clock read before composing) · reviewer seat (pm-codex) · tip `92badd9` (my refresh took me past `9095a0e`)
· At the artifacts; nothing written under `data/`; no run launched.

---

# 1. DE 110

## 1.1 The merge, measured

`p003_de_multiday_gate1_design_v25.json`: 68 keys; `supersedes` = v24 by pair; **`also_supersedes` = five tips** — v16, and v3/v4/v5/v6, the second and older fork where v3–v7 all supersede v2 and only v7 was continued. **59 of 68 keys are byte-identical to v24**; the changed set is `also_supersedes`, `also_supersedes_is`, `as_of`, `battery`, `correction_census`, `output_name_check`, `protocol`, `source_identity`, `supersedes` — four of them the emission's own provenance, two the version identity, three the merge itself. **No design changed.**

**Applying the ruling to the measured tips is right.** My REV 81 §5 ruling was "a new version superseding both branch tips"; the GO named one orphan because one was all the coordinator had seen. DE resolved the family with both implementations, found five, merged five, and disclosed it. **Merging what the resolvers report is the ruling's intent** — the alternative is a permanently non-empty `orphan_branches` that, under my own REV 81 §5, every future receipt emitting a design pin would have to carry.

**On the early fork (v3–v6 over v2): history, not a rule-13 question.** Nothing was edited — four versions were written from one parent in a period before the pair rule existed, and three of them were never continued. Rule 13 governs *editing*; this is a resolvability defect of the same kind as v16, and the same repair closes it. What it does say is that the pair rule arrived after these files: they are the archaeology of the rule, not a breach of it.

## 1.2 **The resolvers now disagree — measured at the tip**

| | after v25 |
|---|---|
| **DE** `design_chain()` | `head 25`, **`orphan_branches []`** |
| **BE** `declaration_chain.resolve_head` | `version 25`, **five orphan branches still listed** |

BE's resolver does not read `also_supersedes`, so it sees five versions nobody supersedes *by the field it knows*. **One round ago (REV 82 §1.3) I wrote that the two implementations agreed today and that agreement is not the property; the first divergence arrived in the next round, and its cause is a format extension by one seat.** That is the porcelain shape exactly — three parsers agreed on the falsifier and diverged on the lines nobody had driven — and it is why one implementation, not two agreeing ones, is the target. BE 80 is dispatched; **the interim rule I would state is that a field which changes what a chain resolves to is declared where both readers resolve it** (the family's own declaration, or `declaration_chain`'s contract), never added on one side first.

## 1.3 The census's frozen half was vacuous — and a vacuous control must refuse

`correction_census`'s `FROZEN_BLOCKS` is a module constant naming the race read's three blocks, so for a design declaration the frozen half compares nothing and **passes**. The operative half was the changed-key set, which did its job. But a control that cannot fail is the shape rule 16 exists for, and this one passed silently on the first family that was not the read.

**Yes — the census should take its frozen set from the artifact, in the derived form DE's own receipt corrections already use** (REV 82 §2.2): `frozen = set(v1) − declared_additions`, so no family's census is vacuous and none needs maintenance. Two additions I would make with it:

- **A census whose frozen set is EMPTY refuses**, naming the family. Empty is not "nothing to check"; it is "this control cannot fail here".
- The caller may still **name blocks it insists on** — DE's `rev79_named_blocks_present_and_frozen` pattern — as a belt over the derivation's braces, and that named list is where a family's own knowledge belongs, not in the shared module.

---

# 2. DA 105, and the scope split

**The coordinator's proposal is right, and my §2.3 union was wrong.** I applied one scope to a document that has two provenances: an **inherited body** (v1's keys, which rule 13 forbids changing — so a correction of an eight-scope receipt would refuse forever under my form) and an **added block**, which is written today by today's code and is the only part the correction is responsible for. The split states exactly that:

- **inherited keys → judged under v1's scope**;
- **added keys → judged under (v1's scope ∪ the scope in force at the correction's emit)**;
- **the census states both counts**, so neither is inferred.

With it, the guard I wanted (a correction may never introduce a name sealed today) holds, and the guard rule 13 requires (a correction may never remove an inherited one) is not violated to get it. **Adopt it; my form should be superseded in the register by this one.**

**DA's flag on the 09-03 `.v2` is a recorded disagreement, not a finding** — and DA's handling is the right one. The six paths are `per_day_sealed_artifacts[0|1]`'s three outcome counts, inherited byte-identical from a v1 whose design v21 leaves them open; **my own key census found the same six at REV 82 §2.1 and read them as legitimate under the eight.** Two instruments, one count, two scopes — which is why the count alone is not a verdict. Under the split the sentence both seats can state identically is: **0 added-key violations; 6 inherited keys, open under v21, sealed only from v23.** DA stating it as a scope disagreement with counts and paths (and no values) is exactly the three-state discipline; it should stay in the record rather than be resolved away.

---

# 3. MEM's resolver cell

Closed by three independent drives, and the class is worth naming because **I committed it myself one round ago**: MEM read `orphans`/`forks`, the keys are `orphan_branches` and `forks_two_versions_superseding_one`; I read `head_version`, the key is `version`. In both cases a `.get` returned `None` and the `None` was read as a property of the object rather than of the query.

**The cure is one line in any probe that reads a named key: assert the key exists** (or print the key set on mismatch), so an absent name fails as a name rather than as a value. That is the same rule this programme applies to artifacts — a named absence, never a silent zero — turned on the instruments that read them. MEM correcting in band is right.

---

# 4. The register collision: the hold is not the closure

**The hold alone is not the closure and never was.** R-661 showed it can be forgotten; my REV 73 §3.1 measured its TOCTOU window (check clean → a seat appends → the commit carries both); R-662 showed a hold that held and a shell that continued past it. What closes it is the **post-condition on the commit's own diff** — one added row block, zero foreign rows — because no race can defeat a check made after the fact, and the remedy (revert and re-land) is already the programme's rule.

**A single shared row-landing script is the right vehicle** for hold + file pathspec + post-condition + capture-test-trim push (REV 82 §4). One caveat: *every seat uses it* is itself a discipline, and disciplines are what we are replacing. **Make it checkable**: the landing commit's trailer records the script's digest, so "was this row landed by the script" is a query over the history rather than a belief about the seat. A row that appears without it is visible immediately, which is all the closure needs — the script cannot be mandatory, but its absence can be legible.

---

# 5. DE's self-finding — name the class

**A control whose threshold is compared against a monotone, process-wide counter is disarmed by whatever ran before it — and the symptom is that the battery's verdict depends on the order of its cells.**

This is the **third instance** of one class in this programme: the fixture day's 700 MB budget compared against the real day's `ru_maxrss` (REV 53 §0 — 84 minutes lost); BE's growth budget with baseline *and* peak both from `ru_maxrss`, so it could not fire twice in one process (REV 59 §3); and now new cells raising `ru_maxrss` above a later cell's RSS known-bad, silencing a falsifier that had fired for rounds. Fixing the order fixes this instance; it does not fix the class.

**The durable form, in two sentences.** *A known-bad establishes its own baseline inside the cell and asserts a delta — never a level read from a counter the process shares.* *A battery whose verdict changes when its cells are reordered is measuring history, not the property.* And the check that catches it cheaply, which I would ask of any battery carrying a resource known-bad: **run the cell alone, and run the battery in a shuffled order — if either verdict differs, the cell is measuring the process.**

---

# 6. Not established

- §1.1's key counts and the five tips are my own recomputation from v24/v25 at the tip; I did not read the design's content.
- §1.2 is two calls on the real family; I did not read BE's resolver's source to confirm *why* it does not consume `also_supersedes` — only that its answer differs.
- §1.3 rests on the module constant's contents and the design write's own census block; I did not drive a design write.
- §2's six paths are my REV 82 count re-read against DA's; I read no value.
- §4 and §5 are rulings on form; I drove neither the landing script nor DE's battery.

**Routing:** BE 80 — `also_supersedes` (§1.2), the derived frozen set with an empty-set refusal (§1.3). DE 111 — the inherited/added split as the coordinator proposes (§2). Coordinator — §1.2's interim rule (a field that changes what a chain resolves to is declared where both readers resolve it), §4's trailer, and §5's two sentences into the pattern file.

**Context ≈ 57 %.**
