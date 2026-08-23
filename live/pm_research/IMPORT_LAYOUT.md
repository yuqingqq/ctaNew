# IMPORT_LAYOUT — the declared standard for `live/pm_research/`

**Declared per Ruling R-43 (Q-DE-9), 2026-08-23.** This converts a deployed
precedent into a decision, so the next probe inherits a rule rather than a
workaround (the pattern the coordinator review named: whatever ships first
with a citation becomes the standard by default — benign exactly when the
default can be justified after the fact, which this one can: it ships in
seven-plus probes across two planes, and changing it buys tidiness for a
seven-file touch).

## The standard

1. **Probes import each other FLAT** (`import flow_intensity as fi`),
   running from `live/pm_research/` or with that directory on `sys.path`.
2. **Package-path modules** (`tier1_pipeline` and anything importing
   `live.pm_research.*`) are reached by putting the REPO ROOT on
   `sys.path` — the bootstrap block used by `cross_window_correlation.py`
   and `layer2_v1.py`:

   ```python
   import sys
   from pathlib import Path
   repo = Path(__file__).resolve().parents[2]
   if str(repo) not in sys.path:
       sys.path.insert(0, str(repo))
   ```

3. **The known hazard, and its rule** (from the coordinator-review
   workaround hunt): a module imported BOTH flat and by package path in one
   process acquires **dual module identity** — module-level state silently
   duplicates, and the files carrying frozen constants (`FROZEN_F_LOW`,
   `SP_OPERATIVE`) are exactly where that bites. **Rule: within one
   process, import any given module ONE way.** Probes import pm_research
   modules flat, always; only `tier1_pipeline`-family modules are imported
   by package path, and they are never also imported flat.
4. New probes copy this file's bootstrap, not another probe's — cite
   `IMPORT_LAYOUT.md` in the comment, so the citation chain ends at a
   decision.

Not chosen, and deliberately: making `pm_research` a package (touches every
probe for tidiness) or a shared bootstrap module (a bootstrap that must be
importable has the same problem it solves). Revisit only if the dual-identity
hazard is ever OBSERVED (a selftest catching duplicated module state), which
would be a register-row event, not a silent fix.
