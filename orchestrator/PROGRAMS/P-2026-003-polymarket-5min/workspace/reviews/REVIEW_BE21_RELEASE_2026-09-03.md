# Re-review — BE round 21: the release decision

reviewer: pm-codex · filed 2026-09-03T08:34Z · pinned tip **`a0b0ebb`** (BE21 at `a27ccfc`, row `Q-BE-246`)
executed in `~/ctaNew-wt-rev` at the tip. **No seal opened.** `be_forward_day.py` read, never run. Nothing written under `data/`.

# **RELEASE.**

**What is released: scoring a forward day on this path.** The blocker I named at BE19 is closed, and I settled it by driving the production chain rather than reading it:

```
op_declaration_for("btc") -> require_operating_point -> require_fenced_op(rows=…) -> increment
  PASSED, with nothing added at the call site
  the fence itself carries:  coin True | verification True | verification_fetched_by_the_fence True
  theta 0.7230267681941027 (the recomputed one) | token_recomputed True
  declared_split_does_not_intersect_scored_population True | increment_cents 18.0 | causal True
```

Suites at the tip, both launchers, rc 0: `be_forward_metric` **99**, `be_operating_point` **19**, `be_forward_recon` **27**, `harmful_forward_scorer` **75**, `be_forward_family` **23**.

Three things remain open and none of them blocks. They are BE21-R1, R2 and R3 below, and R1 is the one I would want landed before a second day is scored.

---

## (1) The fetch — a genuine narrowing, not yet the principle

**Refused, correctly:** an INLINE `verification` block. *"The fence fetches its own evidence; it does not accept evidence handed to it by the caller."* That is the right sentence and the right refusal.

**But the caller can still choose what the fence sees.** Driven at the tip:

| attack | result |
|---|---|
| INLINE `verification` block | **REFUSED** |
| a **fabricated verification FILE I wrote in a temp dir**, `verification_ref` naming it with its own true sha, thetas swapped for the scored rows' cutoff, declared split relabelled to match my fabricated `declared_days` | **PASSED** — through `require_operating_point`, `require_fenced_op` and `increment`, returning the decision metric at **theta 0.99** |
| a **symlink** pointing at a file the caller controls | **PASSED** — the path is not resolved or constrained |
| an **empty** file whose sha matches | REFUSED, but by `JSONDecodeError` — see BE21-R3 |
| a **truncated** file whose sha matches | REFUSED, same shape |

### BE21-R1 — MEDIUM — where the trust boundary now sits, and why it should move one step further

It sits at **any path the caller can name that exists at fetch time.** The sha check gives **integrity** — the bytes are the bytes named — and never **authenticity** — the bytes are a real recomputation. `verification_ref.path` is unconstrained: any existing file, including a symlink to one the caller controls.

I want to be fair about what this did buy, because it is not nothing:

* inline supply is refused, so the evidence can no longer be a literal in the caller's own dict;
* the evidence must **exist as a file** and be **named by sha before the fence runs**, which makes it auditable after the fact in a way a dict never was;
* and the production accessor already names the canonical committed artifact, so the honest path is honest.

What it did not buy is the principle I stated at BE19. *The fence must carry its own evidence* means the fence decides **where** the evidence lives. Here the caller decides, and the fence checks only that the pointer is internally consistent. Fetching a caller-named path is receiving a pointer instead of a payload — a shorter road to the same place, not a different place.

> **Clause, and it is two lines.** Constrain `verification_ref.path` to resolve (`Path.resolve()`, so symlinks are followed and compared) to the canonical committed artifact — the one `be_operating_point.VERIFICATION_PATH` already names — and refuse anything else. Then fabricating the evidence requires **committing** it: visible in git, reviewable, and falsifiable by one command over known bytes. That converts "write any file" into "make a commit", which is the difference between a shortcut and a manufactured receipt.

**Why this does not block.** The production path uses the canonical artifact; reaching a bad number now requires writing a fabricated recomputation and naming it, which is manufacturing evidence rather than taking a shortcut. And the binding itself is real: I re-ran `verify_declaration_by_recomputation()` independently last round, read-only, 805 s, and it reproduced every substantive field over 1,135,930 rows with `max_abs_difference` 0.0 on both coins.

## (2) BE19-R2 — closed, and the older known-bads still fail for their own reasons

The overlap is now computed against the **verified** days and a disagreement refuses. Driven:

```
declared split relabelled away from the VERIFIED days   REFUSED  "the split this operating point…"
declared split = the day being scored (2026-08-29)      REFUSED  same guard, before the overlap
known-bad declared_days_match_the_rows FALSE            REFUSED  "the declared days […] are not the days the rows contain"
known-bad all_coins_reproduce FALSE                     REFUSED  "the recomputation did NOT reproduce…"
known-bad verification over DIFFERENT rows              REFUSED  "the verification was run over rows … but the declaration names …"
```

Each old known-bad still fails, and each fails **with a message that describes its own reason** rather than the new label absorbing them into one. The relabelling attack that passed last round is red.

**Can the verified days themselves be wrong?** Yes — and only in one way. I set `declared_days` *and* `days_derived_from_the_rows` to a fabricated list inside a fabricated verification file, made the declared split agree, and it **PASSED**. That is not a second hole: it is BE21-R1's boundary seen from another side. Given an authentic verification artifact the days are derived by `derive_days_from_rows` from the rows themselves and cannot be wrong; given a fabricated one, nothing in the fence is true. Pinning the path closes both at once.

## (3) The frozen-contract gate

**A correction to my own working, made plainly because I nearly filed it as a finding.** My first reachability walk reported `frozen_contract_gate` as unreachable from `run_forward_day` and I was ready to call it off the run path. It is **on** the run path: `be_forward_day.py:1549`, inside `run_forward_day`, `rec["frozen_contract"] = gate("frozen_contract", frozen_contract_gate)`. My walk followed `Call.func` only and could not see a function passed **by reference**. The gate is wired.

**The derivation is computed, not listed**, and it is the right computation: an AST call-graph reachability from `run_forward_day`, then, for each reachable function that binds the manifest via `json.loads(mp…)`, every constant string used to index that binding.

```
load_bearing_keys : ['hashes', 'pin_semantics']
read_by           : {"hashes": ["materialise_frozen"], "pin_semantics": ["materialise_frozen"]}
reachable from run_forward_day : 121 functions
```

**Would the gate fire on a drift that matters?** Driven on scratch copies of the candidate and manifest — nothing under `data/` touched:

| case | result |
|---|---|
| unmutated copy (control) | **HOLDS**, `drift_touches_the_run_path` False |
| `hashes[…]` mutated — a **load-bearing** key | **REFUSED**: *"the difference touches `['hashes']` — keys the run path READS"* |
| `as_of_utc` mutated only | **HOLDS**, `keys_that_differ` `['as_of_utc','git_commit','git_dirty']`, `touches_run_path` False |

So the partition is real in both directions and the fatal half fires.

**Does the disclosure name the drifts?** Yes — names and both shas, not a count:

```
contract HOLDS | anchors_verified_at_freeze_commit 7 | all match True | n_working_tree_drift 5
  data/pm_5min/derived/harmful_exposure_rows_v3_eraB.json  bound 19a50195c34d0af2  in_tree MISSING
  live/pm_research/harmful_action_eval.py                  bound 55ea57b995afdd4c  in_tree 2c4e21936e3fc1d2
  live/pm_research/harmful_exposure_rows.py                bound 8fb34b0319b0d596  in_tree 1bbd8e7525fc27ac
  live/pm_research/harmful_hazard_model.py                 bound 0091fe75c38af79e  in_tree 58b8a2c08eea3cc9
  live/pm_research/harmful_rows_loader.py                  bound 8b90c48cfe331e71  in_tree c53c64223474d29c
materialise_frozen_sources_from_the_freeze_commit: True   (asserted from that function's own source)
```

(The first row is my worktree's mirror, which does not carry the 1.24 GB rows artifact; the four code drifts are the real ones.) And the survivability claim is not prose: if `materialise_frozen` ever stops sourcing from `_git_blob(FROZEN_COMMIT, …)`, the gate refuses instead of disclosing — a conditional the gate evaluates rather than asserts. This is the shape I would have asked for: it refuses what the run depends on and discloses what it does not, and it says which is which.

### BE21-R2 — MEDIUM — the derivation's call graph cannot see the pattern that wires the gate

The same `Call.func`-only walk that misled me is what `manifest_keys_read_by_run_path` uses. So `frozen_contract_gate` — a manifest reader wired into `run_forward_day` one line away by `gate("frozen_contract", frozen_contract_gate)` — is **invisible to the derivation's own reachability**. Measured:

```
reachable, Call.func only        : 121   frozen_contract_gate in it: False
reachable, + function REFERENCES : 250   frozen_contract_gate in it: True
load-bearing keys, Call.func only : {'hashes': ['materialise_frozen'], 'pin_semantics': ['materialise_frozen']}
load-bearing keys, + references   : {'hashes': ['frozen_contract_gate','materialise_frozen'],
                                     'pin_semantics': ['frozen_contract_gate','materialise_frozen']}
```

**The answer does not change today** — the missed reader reads the same two keys, so the key SET is identical and only the attribution differs. That is luck, not design. A future manifest reader wired by the same `gate(...)` pattern and reading a third key would be silently omitted from the load-bearing set, and a drift in that key would be **disclosed as survivable instead of refusing** — which is the one direction this gate must never get wrong.

I checked the other way the derivation could under-approximate and it does **not**: every `json.loads(x.read_text())` in the module was inspected, and the ones the `"mp"` test skips are candidate and receipt reads, not manifest reads.

> **Clause.** Count `ast.Name` arguments as call-graph edges — BE's own `gate(...)` idiom — or assert in the suite that the load-bearing SET is identical under the narrow and wide graphs, so the day it stops being identical is the day someone looks.

### BE21-R3 — LOW — a malformed verification artifact raises rather than refuses

An empty or truncated file whose sha matches the declaration reaches `json.loads` and raises `JSONDecodeError`. On the run path `gate()` records it as a refusal with its exception type, so nothing passes silently — but this programme has closed the traceback-not-a-name shape three times, and the fence's own vocabulary is available.

> **Clause.** Wrap the parse and refuse by name: *"the verification artifact at … is not readable JSON."*

---

## Findings

| # | sev | finding | blocks |
|---|---|---|---|
| **BE21-R1** | MEDIUM | the fetch moves the trust boundary from the dict to **any path the caller can name**; a fabricated verification file with its own true sha, and a symlink to a caller-controlled file, both pass. Integrity, not authenticity | no |
| **BE21-R2** | MEDIUM | the load-bearing derivation's call graph is `Call.func`-only and cannot see a function passed by reference — the pattern that wires the gate at `:1549`. Answer correct today by luck | no |
| **BE21-R3** | LOW | a malformed verification artifact raises `JSONDecodeError` instead of refusing by name | no |
| — | — | **BE19-R1 CLOSED** (production chain passes end to end, fence carries its own `coin` and `verification`); **BE19-R2 CLOSED** (day-list disagreement refuses; the three older known-bads still fail, each for its own reason); the frozen-contract gate is wired, computed, fires on a load-bearing drift, holds on a non-load-bearing one, and names its five tree drifts | — |

## The release

**RELEASED: scoring a forward day on this path.** Concretely — assembling a fenced operating point through `op_declaration_for` → `require_operating_point`, computing the decision metric through `increment` under `BY_THRESHOLD`, and running the driver's gate sequence including the frozen-contract gate. My BE13/14/15 standing rule — *no forward day scored until BEM-R1, R2 and R3 are closed* — is **lifted**. All three are closed and I have driven each of them: a bare theta is unrepresentable, a forged token refuses, an undeclared or disagreeing split refuses, the numbers are bound to bytes by a recomputation I reproduced independently over 1,135,930 rows at `max_abs_difference` 0.0, and a wrong candidate identity refuses at four production call sites that all bind `expect`.

**Open, and named so it is not mistaken for cleared:**

1. **BE21-R1** — pin `verification_ref.path` to the canonical committed artifact. I would want this landed **before a second day is scored**; it is two lines and it converts the residual from "write any file" into "make a commit".
2. **BE21-R2 / R3** — the derivation's blind spot and the unnamed parse refusal.
3. **The decision metric has never been reconciled against any published number**, and cannot be from existing artifacts — `increment()` is BY_THRESHOLD, iteration 011 is BY_COUNT. Unchanged by this round and stated by BE itself.
4. **Which artifact ought to be scored** (`PM_PLUS_FINE` / LINEAR) is a freeze-level ruling. Not reviewed, not touched.
5. **The four working-tree anchor drifts are real and disclosed, not repaired.** The run executes the freeze commit's bytes, and the gate proves that conditionally rather than asserting it — but the tree and the freeze do differ, and every receipt will say so.

Whether a forward day is now scored, and which, remains the USER's (rule 14). What I am releasing is the instrument, not the decision.

## Discipline record

Executed at `a0b0ebb` in `~/ctaNew-wt-rev`; every step under `systemd-run --user --scope --slice=research.slice -p MemoryMax=8G`. **No seal opened.** `be_forward_day.py` read, never run — the frozen-contract gate and the load-bearing derivation were called as functions. **Nothing written under `data/`**: the forgeries wrote only into temp directories, and the manifest mutants ran on scratch copies of the candidate and manifest. No unit, timer or anchor; `DA_MIDNIGHT_MODE` never set. `~/ctaNew-wt-be`, `-da`, `-de` never read. `git worktree list` **34** at quiescence, worktree clean.
