"""Rehearsal of the partitioned census on the real day before the declared run: the first 4 originals per side (a subset of the frozen 256),
partitions of 2, the stored-row causality on EVERY sampled ok row; output in the scratchpad (not a receipt input)."""
import sys
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2  # noqa: E402
R2.census('20261023', per_side=4, out='/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad/smoke_r2av2', causal_every=1,
          part_size=2, tag='_smoke3')
