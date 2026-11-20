"""Resource measurement for the full-day plan (review d10d3698: revise the ~40 one-core-hour estimate from measured throughput): the same frozen
256 originals, partitions of 64, the stored-row causality OFF; output in the scratchpad (not a receipt input). Read beside the declared r2av3
run (causality on every 17th row): the difference is the causality check's cost."""
import sys
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2  # noqa: E402
R2.census('20261023', out='/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad/throughput_nocausal', causal=False,
          tag='_nocausal')
