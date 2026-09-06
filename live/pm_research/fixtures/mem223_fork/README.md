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
