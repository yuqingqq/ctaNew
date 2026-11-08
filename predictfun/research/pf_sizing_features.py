"""Rebuild continuous basis using each raw relay event once, on the historical clock.

All other feature formulas are the SHA-pinned original builder. This models the
historical research clock, not Tokyo receipt times. No frozen files are written.
"""
import os
from pathlib import Path
from .pf_sizing_study import DAYS, HERE, context, sha

BASE_SHA='8bb95a0090f03196169d62e92a66b8a01bbdb3845021793655b08b4c0555d553'

def source():
    p=HERE/'pf_step5_features_rl.py'
    if sha(p)!=BASE_SHA:raise ValueError('wrong feature source')
    s=p.read_text()
    a='carry = None; tcarry = None\n'
    b='basis_seen = np.zeros(len(ref), dtype=bool)\n'+a
    c="rr = ref[(ref[:, 0] >= k[0]) & (ref[:, 0] <= k[-1])]"
    d="ri = np.flatnonzero((ref[:, 0] >= k[0]) & (ref[:, 0] <= k[-1]) & ~basis_seen)\n        basis_seen[ri] = True\n        rr = ref[ri]"
    for old,new in [(a,b),(c,d)]:
        if s.count(old)!=1:raise ValueError('feature anchor mismatch')
        s=s.replace(old,new)
    return s

def main():
    import sys
    if sys.argv[1] not in DAYS:raise ValueError('undeclared day')
    context();os.environ['PF_ROWS_SUFFIX']='_basis_study'
    os.environ['PF_PEND_KNOWN']='frame'
    exec(compile(source(),str(HERE/'pf_step5_features_rl.py'),'exec'),
         dict(__name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step5_features_rl.py')))

if __name__=='__main__':main()
