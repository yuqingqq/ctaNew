"""Bounded synthetic reproduction of the published Teacher/BE.pair interface.
No model fit or real tape. Only the expensive predictor is replaced by a spy:
Teacher.fn/bind and BE.pair/step/decide remain the pinned implementations.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, '/tmp/pf-review-4584a3c2/predictfun/research')
import pf_r2_teacher as T
BE = T.BE


class Spy(T.Teacher):
    def __init__(self):
        super().__init__({}, None)
        self.checked = 0
        self.disagree = []

    def decide(self, v, B):
        self.checked += 1
        if v['gid'] != B.open[v['side']]:
            self.disagree.append(dict(t=v['t'], side=v['side'], view_gid=v['gid'], bound_gid=B.open[v['side']]))
        return dict(action='WAIT' if v['state'] == 'absent' else 'KEEP')


path = BE.tape([BE.frame(0, 0, 49, 51, 0, 0)], [(150, 0, 49, 10)], 50.)
# Positive: the reviewed single-world usage binds the very branch being stepped.
B = BE.Branch(path); pos = Spy(); pos.bind(B, {})
BE.step(B, 0, pos, forced={0: 'POST'})
for k in range(1, 100):
    BE.step(B, k, pos)
good = BE.finish(B)
assert good['status'] == 'ok' and pos.checked > 0 and not pos.disagree
# Current direct paired use: same valid saved state copied into both arms,
# but the policy remains bound to that saved state, never either executed copy.
ctx = BE.fork_state(path, BE.NEVER, 0, 0)
neg = Spy(); neg.bind(ctx['state'], {})
result = BE.pair(path, neg, 0, 0, 'post', ctx=ctx)
assert result['status'] == 'ok' and result['common_prefork_identical']
assert neg.checked > 0 and neg.disagree
out = dict(target='4584a3c2696625d27cf87f193a6260997348f85e', asof=datetime.now(timezone.utc).isoformat(),
           script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), synthetic_only=True,
           positive_single_branch=dict(calls=pos.checked, mismatches=len(pos.disagree)),
           direct_pair=dict(status=result['status'], calls=neg.checked, mismatches=len(neg.disagree), first_mismatches=neg.disagree[:3]),
           interpretation='The ordinary Teacher callback is not rebound by pair to the copied arm; single-branch rollout remains valid.')
(Path(__file__).parent / 'pair_binding.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out))
