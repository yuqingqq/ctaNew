"""Integrator check: chains_v4 transition links against touch_transition's REAL Oct 24 pilot output (all six chains are Oct 24 episodes).
Writes only to a scratch directory outside the receipt tree; asserts the aligned reader binds, de-duplicates frame pairs and matches pairs."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/reviews/maker_trace_20261122/tools')
import chains_v4 as CV4
import pandas as pd
CV4.SIBLINGS['touch_transition']['dir'] = 'chains/touch_transition_pilot_20261024_r1'
out = Path(tempfile.mkdtemp(prefix='ch4_ttcheck_')) / 'out'
r = CV4.run(mode='draft', out=out, require_freeze=False)
L = pd.read_csv(out / 'chain_links.csv', dtype=str, keep_default_na=False)
T = L[L['link'].str.upper().str.contains('TRANSITION')] if 'link' in L.columns else L
print('links', len(L), 'transition rows', len(T))
txt = (out / 'chains.md').read_text()
for key in ('matched', 'ambiguous_census_pair', 'no_census_pair', 'response_episode', 'no_response_episode', 'ambiguous_response_episode', 'PENDING'):
    print(key, txt.count(key))
assert txt.count('ambiguous_census_pair') == 0, 'duplicate frame pairs leaked'
assert txt.count('matched') > 0, 'no census pair matched'
