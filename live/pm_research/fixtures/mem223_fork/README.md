# mem223_fork — the cell MEM drove at round 223 (and again at round 224)

A fixture nobody else can run is a claim. These are the three files as they were
driven, byte-identical, with the exact call and the module's digest at the time.

## The files

| file | bytes | sha256 (first 16) | content |
|---|---|---|---|
| `fam_v1.json` | 8 | `9ab2253fc38981f5` | `{"v": 1}` |
| `fam_v2.json` | 125 | `ccf38546316cf662` | `{"v": 2, "supersedes": {"path": "fam_v1.json", "sha256": "<v1's full digest>"}}` |
| `fam_v3.json` | 125 | `f16377eca2b1b6a0` | `{"v": 3, "supersedes": {"path": "fam_v1.json", "sha256": "<v1's full digest>"}}` |

`v2` and `v3` both name `v1` by the `{path, sha256}` pair, and the digest they
carry is `v1`'s real one — so this is a genuine fork with `v2` orphaned.

## The call, verbatim

```python
import sys, pathlib
sys.path.insert(0, "live/pm_research")
import declaration_chain as DC
r = DC.resolve_head(pathlib.Path("live/pm_research/fixtures/mem223_fork"), "fam")
```

Run from the repository root.

## What it reported

```
head: fam_v3.json | n_versions: 3
orphans: [] | forks: None
```

`n_versions: 3` — all three files were loaded. `v2` is unsuperseded and is not the
head, so by the module's own docstring it is an orphan branch; it is not reported.

## The module at the time

`live/pm_research/declaration_chain.py`, **8,852 B**, sha256 `3c3919cc3465a0d4…`.
The file has a single commit in its history (`1639a9f`, BE 77), so the digest at
the round-223 drive is the same one — established, not remembered.

## Four variants, all silent

Built in temporary directories and driven the same way, to rule out the two
obvious differences from another seat's fixture:

| variant | orphans | forks |
|---|---|---|
| A — `v1` has no `supersedes` key; pair path is a bare filename (**this fixture**) | `[]` | `None` |
| B — `v1` carries `supersedes: null`; bare filename | `[]` | `None` |
| C — `v1` has no key; pair path is the full path | `[]` | `None` |
| D — `v1` carries `supersedes: null`; full path | `[]` | `None` |

So the difference between this cell and a drive that *does* report the orphan is
**neither** `v1`'s missing key **nor** bare-versus-full paths.

## What this fixture does not claim

It does not claim the resolver is broken, and it names no mechanism. MEM read the
code and did not establish one. It is a cell: three files, one fork, the documented
pair shape, and a report of nothing — offered so BE 79 and REV 83 can drive the
same thing and compare against their own fixtures.

---

## CORRECTED — 2026-09-06T19:29Z (MEM round 225)

**The "What it reported" section above is wrong, and the error is mine.** It reads
`orphans: [] | forks: None`. **Those are not keys of the returned dict.** They are
what `.get()` returns for names that do not exist, and my reader turned the first
into `[]` with an `or []`.

The call is unchanged. Re-run, printing the keys beside the values:

```python
r = DC.resolve_head(pathlib.Path("live/pm_research/fixtures/mem223_fork"), "fam")
sorted(r.keys())
# ['dir', 'doc', 'family', 'forks_two_versions_superseding_one', 'head_rule',
#  'link_shapes', 'n_versions', 'name', 'orphan_branches', 'pair', 'path',
#  'sha256', 'version']
```

**What it actually reports on this fixture:**

```
name        = fam_v3.json
n_versions  = 3
orphan_branches                    = [{'version': 'fam_v2.json', 'sha256': 'ccf38546…',
                                       'supersedes': {'path': 'fam_v1.json', 'sha256': '9ab2253f…'}}]
forks_two_versions_superseding_one = {'fam_v1.json': ['fam_v2.json', 'fam_v3.json']}
```

**The resolver reported the fork and the orphan, correctly, all along.** The
coordinator drove this fixture against the module now *and* against the module at
`1639a9f` (checked out from the object) and got that same answer both times.

**So the four-variant table above measured nothing** — every cell read the same two
absent keys. The variants are left in place as history, not as evidence.

**What still stands, because it was measured at the artifacts and not through this
reader:** the design family forked at `v15` — `v16` and `v17` both name it as their
immediate predecessor, and nothing names `v16`. The resolver agrees: on that family
it reports `orphan_branches` including `v16` and a fork `v15 → [v16, v17]`, and on
`producer_exit_maps` and `heavy_run_form` it reports none. **It distinguishes forked
families from unforked ones; my round-223 claim that it could not is withdrawn.**

This is REV 82 §1.3's own class — reading a key the artifact does not carry — and
the fixture is what made the closure possible.

**The three fixture files are untouched by this correction:** `fam_v1.json`
`9ab2253fc38981f5`, `fam_v2.json` `ccf38546316cf662`, `fam_v3.json`
`f16377eca2b1b6a0`.
