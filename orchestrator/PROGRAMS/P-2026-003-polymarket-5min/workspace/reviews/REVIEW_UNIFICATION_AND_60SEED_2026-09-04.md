# Review — one defect or three? and what 60 seeds can license

reviewer: pm-codex · filed 2026-09-04T12:26Z · tip `60ecc21` (BE's prose scan at `fd0995c`)
executed in `~/ctaNew-wt-rev`. **No race seal opened.** Nothing written under `data/`.

**Independence declaration, as asked: I have not read BE's filing on the unification question.** I read the *code diff* of `fd0995c` — the artifact under review — and the one-line commit subjects in `git log`. I did not open Q-BE-264's row, or any BE prose answering the question. My answer below was formed from the three predicates as they stand in code.

DE's 60-seed round and DA's round have not landed at this tip; item 2 is answered in advance, as a specification of what I will accept when it does.

---

# 1. **One defect. Three checkers. They must not merge — and the reason is the oracle.**

## The defect is one, and it is counterfactual dependence

Take a **claim** C, and a **token** T in an artifact that a reader takes as evidence for C. The defect in all three cases is the same:

> **T would have been produced identically if C were false.**

That is not a family resemblance; it is the same sentence three times.

| | claim C | token T | would T look the same with C false? |
|---|---|---|---|
| **codomain** | *this window is dark* | the number `0` from `uncompressed_size` | **yes** — `0` is produced by an empty file *and* by an unreadable one |
| **citation** | *R-232 rules `clob_v3_1` inadmissible* | the string `R-232` beside the value | **yes** — the string was typed; it reads the same whatever R-232 contains |
| **prose** | *the ordinary gate still refuses without the admission* | the sentence in `scope` | **yes** — and BE's own replacement comment says so: *"it would have kept reading true after any widening of the gate"* |

So the coordinator's instinct is right and I am not going to argue him out of the unification **at the level of the defect**. I would put the operational form of it in the runbook as a question, not a program:

> **For every token a reader will take as evidence: could this token have been produced with the claim false?** If yes, it is not evidence — whatever it is made of.

That question covers all three, and it is the thing worth making standing practice.

## But the checkers are three, because the oracles are three

What separates them is not the shape of the defect. It is **what you must consult to settle it** — and no single program spans these:

| | the oracle | decidable from | cost |
|---|---|---|---|
| **codomain** | the function's own paths | the **source alone**, statically — compare the value set reachable on error paths with the set reachable on success paths | an AST pass, seconds |
| **citation** | **another artifact** | the citing artifact **plus the cited one**; nothing in the citing file can settle it | fetch and match, cheap but two-document |
| **prose** | **a running system** | neither source nor documents — only **execution** | run the behaviour, arbitrarily expensive |

A checker that reads source cannot fetch a register entry. A checker that fetches documents cannot know whether a gate refuses. A checker that runs a gate cannot tell you that `0` means two things. **Any one of these implemented as "the" checker misses the other two thirds, which is exactly the failure the coordinator anticipated.**

## The discriminating cases, one each

Each has a case the other two are structurally blind to:

- **Codomain only.** `uncompressed_size` returning `0` on `OSError`. There is no reference to fetch and no sentence to run — only a value with two production paths. *Citation and prose checkers have nothing to look at.*
- **Citation only.** `"clob_v3_1": False,  # R-232`. The value is a bool from a bool-valued table with no error path — codomain-clean. There is no assertion about behaviour. *Only the referent is wrong, and only fetching R-232 shows it.*
- **Prose only.** `scope: "…still refuses this verdict when called without the admission"`. No error path, no citation. A true-or-false statement about a running system. *Only calling the gate settles it* — which is precisely what `_ordinary_gate_outcome` now does.

## What the unified view misses, and this is the part I would not have found by agreeing

If the three are unified into one *discipline*, the discipline is sound. If they are unified into one *checker*, it misses the two thirds above — and it also hides that **there is a fourth oracle nobody has instrumented.**

The four kinds of claim this programme actually makes:

1. claims about a **value's production** → codomain checker · **instrumented**
2. claims about **another document** → citation checker · **instrumented, on one table only**
3. claims about **behaviour** → execution · **instrumented as of `fd0995c`**
4. claims about a **population** → *nothing*

The fourth is live and I have filed an instance of it: **DE53's "1,309 of 31,122 generations (4.21 %) excluded"** is an honest, reconciling status — and the claim a reader takes from it is that the arms' numbers describe the population. Nothing checks that. The token (a reconciling count with a reason) would read identically whether the exclusion is ignorable or wildly selective, because the oracle is a *statistical comparison between the excluded and retained sets* and no such comparison exists. Same defect, fourth oracle, zero coverage.

A fifth is adjacent and I would name it rather than instrument it: claims about a **human ruling** (*"the USER decided X"*). The citation checker can verify the ruling exists; it cannot verify it means what is claimed. That is exactly the `clob_v4` / R-340 residue DA reports and correctly does not act on.

## So my answer, stated as a ruling would need it

**One defect; four oracles; three instruments; keep them separate.** Build the question into the protocol, not the program. If a single artifact is wanted, make it a **router**: classify each claim-bearing token by which oracle settles it, and refuse a token whose oracle is *none* — because a claim no oracle can settle is precisely the one that gets believed.

---

# 2. The 60-seed result — what I will and will not accept when it lands

I have no artifact to review: **no `de_section81_arms__*.json` exists on disk and none has ever been committed**, so the numbers currently live in a commit message — the same shape as the source-comment `496` they replace, and DE is right to be re-running and committing. What follows is the specification, fixed before I see the numbers.

## 60 seeds do **not** license "the achievable set excludes the target"

Taken as a support claim from a finite sample, 0 of 60 draws at or below the target bounds the per-draw probability at roughly **3/60 = 5 %** (the rule of three, one-sided 95 %). That is a statement about **draws**, not about the **set**, and 5 % is not exclusion. And the extrapolation asked for is not small: the target is 333, the observed minimum 412, and the observed range is 111 wide — so bracketing requires a draw about **0.7 range-widths below the smallest of sixty**. Not absurd; not established.

**`target_bracketed: false` is therefore a fine observation and a bad conclusion.** As an observation — *no draw in 60 bracketed the target* — it is exactly right and should be emitted in those words. As a conclusion — *the target is not bracketable* — it is a budget claim in better clothes, which is the coordinator's own suspicion and I share it.

## What *does* license the strong reading is the mechanism, not the count

The claim becomes something other than induction only if the sign is **predicted** rather than **observed**. It is: a random draw does not reproduce the treated arm's suppression clustering, so the control realises a systematically higher fraction of above-events (28.9 % against ~43 %). Under that mechanism, all-positive over 60 is **confirmation of a predicted sign**, and the count is corroboration rather than the argument.

Two things follow, and I will look for both:

1. **The per-stratum result is the stronger evidence and should be the headline.** "Control-OVER on *every* stratum" is many signs pointing one way, not one aggregate; an aggregate gap can be produced by a few strata, a universal per-stratum sign cannot. If the artifact reports the aggregate as the finding and the per-stratum result as detail, the ordering is backwards.
2. **The gap's dependence on the suppression rate should be shown, not assumed.** If the mechanism is right, strata with more non-acting above events should show larger control-over. That is a computable prediction the same 60 seeds already contain, and it converts a repeated observation into a tested mechanism.

## DA's limit is right and must be quoted, not paraphrased

**UNREACHABLE FOR THIS CONSTRUCTION, never NEVER.** Everything above is conditional on the draw rule, the stratification, and the frozen protocol's single matched quantity. A different construction is not excluded by any of it, and the artifact should say so in a field rather than in a filing — otherwise the next reader inherits "no matched floor exists" when what was shown is "no matched floor exists *under this draw*".

## And the provenance census applies to this artifact like any other

When it lands I will run it and state the verdict before commenting on the numbers: **produced_by**, **producing_code**, **carrying_commit**, and a committed producer with a callable entry point — the last of which is still open as DE55-R2, since the current runner executes on import.

---

# 3. Standing

**DE55-R1 remains open at this tip.** `d["VALID_AS_A_CONTROL"] = True` and four other spellings still pass the two-string substring needle.

---

## Findings

| # | sev | |
|---|---|---|
| — | — | **Item 1 answered: one defect, three checkers, four oracles.** The defect is counterfactual — *T would have been produced identically if C were false* — and it genuinely unifies the three. The checkers cannot merge because the oracle differs (own source / another document / a running system), and each has a case the other two are blind to. **A fourth oracle — claims about a population — is live and uninstrumented**, with DE53's 4.21 % exclusion as the standing instance |
| — | — | **Item 2 specified in advance**: 0/60 bounds a per-draw rate at ≈5 %, which is not exclusion; `target_bracketed false` is an observation, not a conclusion; the mechanism licenses the strong reading and the per-stratum universality is the stronger evidence; DA's *for this construction* limit belongs in a field. The provenance census runs first |
| — | — | **DE55-R1** still open |

## Disposition

Nothing to release or hold this round — DE's and DA's work has not landed. The one thing I would act on now is the fourth oracle: it is the only one of the four with no instrument, it has a live instance already filed, and unlike the other three nobody has yet claimed it is covered.

I estimate; the coordinator routes; the USER decides (rule 14).

## Discipline record

Executed at `60ecc21` in `~/ctaNew-wt-rev`. **No race seal opened**; `be_forward_day` never run — `fd0995c` was read as a diff. Nothing written under `data/`. `~/ctaNew-wt-be`, `-da`, `-de` never read. **I did not read BE's filing on the unification question**; my answer was formed from the three predicates as they stand in code, and the only BE text I opened was the `fd0995c` code diff and one-line commit subjects. Worktree clean.
