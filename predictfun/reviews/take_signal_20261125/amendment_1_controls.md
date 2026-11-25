# TAKE-SIGNAL v1 — amendment 1 (declared after the primary tables, before any control is computed)

The primary result is strongly positive (Tokyo mark_3s +2.35 ¢ per filled share, all four days, both directions). Under CLAUDE.md
("anything this strong is suspicious") the following controls are added before any interpretation. They use the SAME valuation code
(`controls.py` first re-values every kept trigger and must reproduce `takes.parquet` exactly before it values anything else):

1. **Placebo times** (falsifier): the same number of triggers per market and direction at seeded uniform random times in
   [ws, END − 30 s] (seed 20261125), no BTC condition. Expected ≈ −(half spread + fee) per share. A clearly positive placebo voids
   the primary reading.
2. **Shifted times:** every kept trigger moved +7 s (kept only if still < END). Expected ≤ 0.
3. **Opposite direction:** the kept triggers with the side flipped (sell when the signal says up). Expected strongly negative.
4. **Very slow clock:** known book at t − 200 ms, arrival t + 150 ms (worse than this US host). Shows how fast the edge decays.

All four are controls, not significance nulls: no p-value is claimed from them. Values are reported per filled share at Q = 50
with fill rates, pooled and by day, exactly like the primary table.
