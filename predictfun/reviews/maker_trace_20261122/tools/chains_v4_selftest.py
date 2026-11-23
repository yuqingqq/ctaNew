"""chains_v4_selftest.py - synthetic proof of chains_v4.py (P-2026-005 MAKER-TRACE; TOUCH-TRANSITION-AND-RESPONSE v1 section 5; 2026-11-23).
In-process, synthetic inputs only: the frozen chains_v3_selftest world builder (imported, never edited: a temporary directory mirroring the real
file names, columns, receipts and parquet row groups) with the section-5 additions - queue_front_v2 levels carrying the alt-evaluation columns
(a False saved before the calculation, an evaluated False, True rows), a creation inside chain 4's post window, posting_opportunity events.csv
(identity) and a touch_transition census / episodes output in the ASSUMED interface. No real data path is opened.
  units      alt-netting states and the chain state (+ the verifier's raw predicate, written apart), the pre-state grade rule for G0..G4, the
             window position rule, the gross label, the touch_transition column resolution; each planted mutant shown LIVE at unit level.
  draft      no touch_transition: four outputs only, receipt hashes = bytes, nothing else touched; TRANSITION links PENDING with the reason; a G1 and a
             G4 post with constant features -> TIED with unresolved_outside_window (v3's ONE-HISTORY recorded beside); G2 constant -> ONE-HISTORY
             (positive control); a False saved before the calculation -> not_evaluated, the masked-netting history kept; an evaluated False ->
             excluded, ONE-HISTORY (positive control); the summary table / agreement = the graded records; 'gross c/sh' on every gross markout.
  refusals   overwrite (write-once); run without touch_transition (message); a v4 mutant outside the selftest; an unknown v4 mutant; a v3 mutant.
  mutants    v3_pre_state_rule / alt_false / summary_ungraded / v3_value_label: each refused by the verifier before any write.
  run        every sibling bound: all predicates hold, no pending link; chain 4's creation inside its window matched to improve_bid and the response
             episode with candidate_constrained (this order), TIED (joined / masked kept); a creation without a census pair keeps its status; a
             censored end at the creation frame is not a response; chain 2 TRANSITION ONE-HISTORY joined_shown_cell (positive control); a grade
             other than G2 keeps unresolved_outside_window.
  falsifiers a touch_transition file not matching its receipt and a census without its class column: PENDING with the reason in a draft, refused
             by a run before any write.
Usage (run_light, MT_RUN=v2): chains_v4_selftest.py"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[2] / 'research'
for _p in (str(RESEARCH), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import chains_v3_selftest as S3  # noqa: E402   (the frozen v3 world builder: imported, never edited)
import chains_v4 as CH  # noqa: E402

CH3 = CH.CH3
RES = []
G0, G1, G2, G3, G4 = CH.G0, CH.G1, CH.G2, CH.G3, CH.G4
DAY, WS = S3.DAY, S3.WS


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:600]))


def refuses(fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return needle in str(e), f'{type(e).__name__}: {e}'
    return False, 'did not raise'


def expect(name, fn, needle, extra=True):
    ok, info = refuses(fn, needle)
    check(name, ok and extra, info)


def section(name, fn, *a):
    try:
        fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}')


class Env:
    def __init__(self, var, val):
        self.var, self.val = var, val

    def __enter__(self):
        self.old = os.environ.get(self.var)
        os.environ[self.var] = self.val

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop(self.var, None)
        else:
            os.environ[self.var] = self.old


def Mut(m):
    return Env('MT_CH4_MUT', m)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(rp, name, path):
    """record (or overwrite) a file's sha256 in a receipt's outputs."""
    rec = json.loads(rp.read_text()) if rp.is_file() else {}
    rec.setdefault('outputs', {})[name] = dict(sha256=sha(path))
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1))


def run_q(rr, dr, out, mode='draft'):
    return CH.run(mode=mode, rroot=rr, droot=dr, out=out, require_freeze=False, selftest=True, log=lambda *a, **k: None)


# ------------------------------------------------------------------ the synthetic world (v3's + section 5) ------------------------------------------------------------------
def level_cells(lid, status):
    """the alt-evaluation cells queue_front_v2 saves for one level (its _new_life_v2 rule)."""
    if lid == 100:                 # chain 1: no earliest record - queue_front_v2 returns before the calculation, the initial False stays
        return dict(status='unknown', reason='untimed_record_only', alt_netting_possible=False, timing_basis='', n_earliest_set=0,
                    alt_sites_life=0, alt_sites_unexplained=0)
    if lid == 101:                 # chain 2: an evaluated False (earliest record, zero sites) - the positive control for 'excluded'
        return dict(status='front_candidate', reason='ok', alt_netting_possible=False, timing_basis='book_candidate', n_earliest_set=1,
                    alt_sites_life=0, alt_sites_unexplained=0)
    if lid in (102, 103, 104, 105, 403):
        return dict(status='front_candidate' if lid == 403 else 'ambiguous_order', reason='ok' if lid == 403 else 'untimed_record_could_be_first',
                    alt_netting_possible=True, timing_basis='book_candidate', n_earliest_set=1, alt_sites_life=2, alt_sites_unexplained=0)
    return dict(status=status, alt_netting_possible=False, timing_basis='', n_earliest_set=0, alt_sites_life=0, alt_sites_unexplained=0)


def v4_world(root, tt=False):
    rr, dr, F = S3.build_world(root, siblings=True)
    q2 = rr / 'chains/queue_front_v2'
    LV = pd.read_csv(q2 / 'levels.csv', dtype={'day': str})
    base = LV[LV['level_id'] == 103].to_dict('records')[0]
    ws13 = WS[13]
    recs = LV.to_dict('records') + [dict(base, level_id=403, t_open_ms=ws13 + 68_000, t_create_ms=ws13 + 68_400, end_type='zero',
                                         end_ms=ws13 + 70_100, n_members=1, creation_context='improve')]
    for r in recs:
        r.update(level_cells(int(r['level_id']), r['status']))
        r['frame'] = r['fid'] = int(r['level_id'])
    S3.write_csv(pd.DataFrame(recs), q2 / 'levels.csv')
    bind(q2 / 'receipt.json', 'levels.csv', q2 / 'levels.csv')
    iv = {int(r['level_id']): (int(r['t_open_ms']), int(r['t_create_ms'])) for r in recs}
    if tt:
        h = {s['i']: f['h'] for s, f in F}
        po = rr / 'chains/posting_opportunity'

        def ev(mid, side, tick, lid, ident, reason, cand, win, grade):
            a, b = iv.get(lid, (1, 2))
            return dict(key=f'{mid}|{side}|{tick}|{lid}', day=DAY, mid=mid, side=side, tick=tick, frame=lid, fid=lid, t_open_ms=a, t_create_ms=b,
                        status='event', identity=ident, identity_reason=reason, cand_hash=cand, cand_addr='', win_contains=win, grade=grade)
        S3.write_csv(pd.DataFrame([ev(13, 'bid', 42, 403, 'candidate_constrained', 'ok', h[3], True, G2),
                                   ev(13, 'bid', 42, 103, 'owner_unknown', 'window_not_containing', h[3], False, G2),
                                   ev(11, 'ask', 69, 100, 'candidate_scenario', 'ok', h[0], True, G4),
                                   ev(12, 'ask', 13, 101, 'owner_unknown', 'no_front_candidate:front_candidate', '', False, ''),
                                   ev(99, 'bid', 1, 7, 'owner_unknown', 'outside_population', '', False, '')]), po / 'events.csv')
        bind(po / 'receipt.json', 'events.csv', po / 'events.csv')
        td = rr / 'chains/touch_transition'
        CE = [dict(day=DAY, mid=m, upd_prev_ms=iv[lid][0], upd_ms=iv[lid][1], cls=c, sub=sb)
              for m, lid, c, sb in ((13, 403, 'improve_bid', ''), (11, 100, 'two_side', 'widen'), (12, 101, 'unchanged', ''),
                                    (12, 102, 'retreat_bid', ''), (13, 104, 'improve_ask', ''))]
        CE += [dict(day=DAY, mid=13, upd_prev_ms=ws13 + 1_000, upd_ms=ws13 + 1_100, cls='unchanged', sub=''),
               dict(day=DAY, mid=99, upd_prev_ms=1, upd_ms=2, cls='unchanged', sub='')]
        S3.write_csv(pd.DataFrame(CE), td / 'census.csv')
        st = dict(spread_bin='2', depth_own_bin='(0,50]', btc_5s_aligned='flat', fair_gap_bin='(1,3]', time_left_bin='(60,240]')
        EPs = [dict(day=DAY, mid=13, side='bid', episode_id='EP13B', start_ms=ws13 + 67_900, delta_ticks=1, end_type='response', end_lo_ms=iv[403][0],
                    end_ms=iv[403][1], exposure_ms=500, response_kind='restoring', owner_grade='candidate_constrained', owner_hash=h[3], **st),
               dict(day=DAY, mid=11, side='ask', episode_id='EP11A', start_ms=iv[100][0] - 100, delta_ticks=2, end_type='censored_other',
                    end_lo_ms=iv[100][0], end_ms=iv[100][1], exposure_ms=150, response_kind='', owner_grade='', owner_hash='', **st),
               dict(day=DAY, mid=12, side='bid', episode_id='EP12B', start_ms=iv[102][0] - 300, delta_ticks=1, end_type='response',
                    end_lo_ms=iv[102][0], end_ms=iv[102][1], exposure_ms=350, response_kind='partial', owner_grade='owner_unknown', owner_hash='', **st),
               dict(day=DAY, mid=13, side='ask', episode_id='EP13A', start_ms=ws13 + 195_500, delta_ticks=3, end_type='no_response_1s',
                    end_lo_ms=-1, end_ms=ws13 + 196_500, exposure_ms=1000, response_kind='', owner_grade='', owner_hash='', **st)]
        S3.write_csv(pd.DataFrame(EPs), td / 'episodes.csv')
        for nm in ('census.csv', 'episodes.csv'):
            bind(td / 'receipt.json', nm, td / nm)
    return rr, dr, F


def falsify(src, dst, kind):
    shutil.copytree(src, dst)
    rr, dr = Path(dst) / 'v2', Path(dst) / 'data'
    td = rr / 'chains/touch_transition'
    if kind == 'sha':                                             # a byte change the receipt does not record
        rec = json.loads((td / 'receipt.json').read_text())
        rec['outputs']['census.csv']['sha256'] = '0' * 64
        (td / 'receipt.json').write_text(json.dumps(rec, indent=1))
    else:                                                         # a schema change, bound by its receipt
        CE = pd.read_csv(td / 'census.csv', dtype={'day': str})
        S3.write_csv(CE.drop(columns=['cls']), td / 'census.csv')
        bind(td / 'receipt.json', 'census.csv', td / 'census.csv')
    return rr, dr, None


# ------------------------------------------------------------------ sections ------------------------------------------------------------------
def s_units():
    base = dict(level_id='1', status='front_candidate', alt_netting_possible='True', timing_basis='book_candidate', n_earliest_set=1,
                alt_sites_life=1, alt_sites_unexplained=0, has_evidence_cols=True)
    nf = dict(base, status='unknown', alt_netting_possible='False', timing_basis='', n_earliest_set=0, alt_sites_life=0)
    evf = dict(base, alt_netting_possible='False', alt_sites_life=0)
    inc = dict(base, alt_netting_possible='False', alt_sites_life=2)
    mis = dict(base, alt_netting_possible='absent')
    v1 = dict(nf, has_evidence_cols=False, alt_netting_possible='absent')
    rows = (base, nf, evf, inc, mis, v1)
    got = tuple(CH.alt_state(x) for x in rows)
    check('alt-netting state: True -> possible; False saved before the calculation -> not_evaluated; evaluated False (earliest record, zero '
          'sites) -> excluded; inconsistent False -> not_evaluated; missing / v1 -> not_evaluated',
          got == ('possible', 'not_evaluated', 'excluded', 'not_evaluated', 'not_evaluated', 'not_evaluated'), got)
    check('the verifier raw predicate (written apart from alt_state) agrees on the same cells',
          tuple(CH._raw_unevaluated(x) for x in rows) == (False, True, False, True, True, True) and CH._raw_evaluated_false(evf)
          and not CH._raw_evaluated_false(nf))
    rw = CH.alt_raw(dict(level_id=7, status='unknown', alt_netting_possible=np.False_, timing_basis=float('nan'), n_earliest_set=0,
                         alt_sites_life=0, alt_sites_unexplained=0))
    check('alt_raw reads a saved row (numpy bool, NaN timing basis) without interpreting it; its state is not_evaluated',
          rw == dict(level_id='7', status='unknown', alt_netting_possible='False', timing_basis='', n_earliest_set=0, alt_sites_life=0,
                     alt_sites_unexplained=0, has_evidence_cols=True) and CH.alt_state(rw) == 'not_evaluated', rw)
    check('chain state: no level -> not_evaluated; any possible -> possible; any not_evaluated keeps it; all excluded -> excluded',
          (CH.chain_alt_state([]), CH.chain_alt_state(['excluded', 'possible']), CH.chain_alt_state(['excluded', 'not_evaluated']),
           CH.chain_alt_state(['excluded'])) == ('not_evaluated', 'possible', 'not_evaluated', 'excluded'))
    with Mut('alt_false'):
        check('mutant alt_false is live (a False before the calculation -> excluded; no level -> excluded)',
              CH.alt_state(nf) == 'excluded' and CH.chain_alt_state([]) == 'excluded')
    chd = dict(chain=9, rank=1, episode_id='e9', day=DAY, mid=11, addr='0x' + 'a1' * 20, order_hash='0x' + 'ab' * 32)

    def one():
        return CH3.new_link(chd, '1', 'pre_state', evidence_grade='observed_state', alternatives=['spread=1; fair_gap=(0,1]'], n_histories=1,
                            observed_public=True, n_records=1, summary='1 saved seconds; features constant: spread', lines=['  - window seconds k = 4..4'])
    R_ = {g: CH.grade_pre_state(one(), g) for g in (G0, G1, G2, G3, G4)}
    check('pre-state rule: G0 / G1 / G3 / G4 with ONE constant vector keep unresolved_outside_window (TIED, the hidden action named); G2 stays '
          'ONE-HISTORY; each carries its grade and the v3 reading beside',
          all(R_[g]['n_histories'] == 2 and R_[g]['alternatives'][-1] == CH.UNRESOLVED and R_[g]['hidden_action'] and CH3.expected_token(R_[g]) == 'TIED'
              for g in (G0, G1, G3, G4))
          and R_[G2]['n_histories'] == 1 and CH3.expected_token(R_[G2]) == 'ONE-HISTORY'
          and all(R_[g]['evidence_grade'] == f'observed_state:post_window:{g}' and R_[g]['post_grade'] == g and R_[g]['v3_reading'] == 'ONE-HISTORY'
                  for g in R_), {g: (R_[g]['n_histories'], R_[g]['alternatives']) for g in R_})
    with Mut('v3_pre_state_rule'):
        m_ = CH.grade_pre_state(one(), G1)
        check('mutant v3_pre_state_rule is live (G1 with one vector -> ONE-HISTORY, no unresolved alternative)',
              m_['n_histories'] == 1 and CH.UNRESOLVED not in m_['alternatives'])
    pend = CH.grade_pre_state(CH3.pending_link(chd, '1', 'pre_state', 'x', 'posting_opportunity receipt absent'), G1)
    check('a pending pre-state stays PENDING and still carries the post grade', CH3.expected_token(pend) == 'PENDING' and pend['post_grade'] == G1)
    wp = tuple(CH.window_position(*x) for x in ((100, 200, 99, 150), (-1, 200, 0, 200), (100, 200, 150, 250), (100, 200, 10, 99),
                                                (100, 200, 200, 300), (100, 200, 98, 150)))
    check('window position: inside (lo <= upd(f-1) + 1 and upd(f) <= hi; unbounded lo) / straddles / before / after',
          wp == ('inside', 'inside', 'straddles', 'before', 'after', 'straddles'), wp)
    V = CH3.views(pd.DataFrame(dict(q=[10.], fee_fill=[0.], st_10s=['ok'], v_10s=[0.5], st_60s=['ok'], v_60s=[-0.2], st_end=['ok'], v_end=[0.1],
                                   st_res=['ok'], net_res=[0.3])))
    ln = CH._vline_v4('x', V)
    check("gross label: every gross per-share markout reads 'gross c/sh' (common cohort too), the resolution view 'net c/sh'; none bare after "
          "net $ (v3's line is caught by the same pattern)",
          ln.count('gross c/sh') == 4 and 'net c/sh' in ln and not CH.BARE_RE.search(ln) and bool(CH.BARE_RE.search(CH._V3_VLINE('x', V))), ln)
    with Mut('v3_value_label'):
        check('mutant v3_value_label is live (a bare c/sh after net $)', bool(CH.BARE_RE.search(CH._vline_v4('x', V))))
    r1 = CH._resolve(['day', 'mid', 'upd_prev_ms', 'upd_ms', 'cls', 'zz'], CH.TT_COLS['census'])
    r2 = CH._resolve(['day', 'mid', 'upd_ms'], CH.TT_COLS['census'])
    check('touch_transition columns resolve by their declared names; a missing logical column is None (named in the refusal)',
          r1 == dict(day='day', mid='mid', lo='upd_prev_ms', hi='upd_ms', cls='cls') and r2['lo'] is None and r2['cls'] is None, (r1, r2))


def s_draft(world):
    rr, dr, F = world
    out = rr / 'chains/chains_v4_draft'
    s0 = S3.snapshot(rr.parent)
    res = run_q(rr, dr, out)
    s1 = {k: v for k, v in S3.snapshot(rr.parent).items() if not k.startswith(str(out))}
    check('draft writes exactly the four outputs into chains_v4_draft and touches nothing else (no partial directory left)',
          sorted(p.name for p in out.iterdir()) == sorted(CH.OUT_FILES) and s0 == s1, set(s0) ^ set(s1))
    rec = json.loads((out / 'receipt.json').read_text())
    check('receipt: sha256 of every output = its bytes; CPU and VmHWM recorded; every input bound ok; the base module named',
          all(sha(out / nm) == v['sha256'] for nm, v in rec['outputs'].items()) and set(rec['outputs']) == {'chains.md', 'chain_links.csv', 'claims.json'}
          and rec['cpu_s']['process_total'] > 0 and rec['vmhwm_mib'] > 0 and all(v['binding']['status'] == 'ok' for v in rec['inputs'].values())
          and rec['base_module'].endswith('chains_v3.py'), {k: v['binding']['status'] for k, v in rec['inputs'].items()})
    cl = json.loads((out / 'claims.json').read_text())
    P = cl['predicates']
    LK = S3.links_of(out)
    md = (out / 'chains.md').read_text()
    check('six chains x eight links; every link graded in its link vocabulary and with a declared reading',
          len(LK) == 48 and P['every_link_graded'] and P['every_link_has_a_declared_reading'] and P['every_chain_has_every_link'])
    tr = LK[LK.link == 'transition']
    check('touch_transition absent: every TRANSITION link PENDING with the reason (draft)',
          (tr.status == 'pending').all() and (tr.reading == 'PENDING').all() and tr.pending_reason.str.contains('touch_transition receipt absent').all()
          and P['transition_pending_iff_sibling_absent'], tr.pending_reason.tolist()[:1])
    pre = LK[LK.link == 'pre_state'].set_index('chain')
    check('pre-state G1 (chain 3) and G4 (chain 5) with constant features: TIED with unresolved_outside_window (v3 rule: ONE-HISTORY, recorded beside)',
          all(pre.loc[k, 'reading'] == 'TIED' and CH.UNRESOLVED in str(pre.loc[k, 'alternatives']).split('|') and pre.loc[k, 'v3_reading'] == 'ONE-HISTORY'
              and int(pre.loc[k, 'n_histories']) == 2 for k in (3, 5)) and pre.loc[3, 'post_grade'] == G1 and pre.loc[5, 'post_grade'] == G4,
          pre[['reading', 'n_histories', 'post_grade', 'v3_reading']].to_dict())
    check('pre-state G2 with constant features (chains 2, 4): ONE-HISTORY allowed (positive control); chain 1 (G4, spread changes) TIED with unresolved',
          pre.loc[2, 'reading'] == 'ONE-HISTORY' and pre.loc[4, 'reading'] == 'ONE-HISTORY' and pre.loc[1, 'reading'] == 'TIED'
          and CH.UNRESOLVED not in str(pre.loc[2, 'alternatives']).split('|') and CH.UNRESOLVED in str(pre.loc[1, 'alternatives']).split('|'))
    check('every pre-state carries the post grade of its regrade row (evidence observed_state:post_window:<G>)',
          all(pre.loc[k, 'evidence_grade'] == f"observed_state:post_window:{pre.loc[k, 'post_grade']}" for k in pre.index)
          and [pre.loc[k, 'post_grade'] for k in range(1, 7)] == [s['grade'] for s in S3.SPEC])
    check('claims predicates: no ONE-HISTORY pre-state unless G2; the unresolved alternative unless G2; the grade carried',
          P['no_one_history_pre_state_unless_g2'] and P['unresolved_alternative_unless_g2'] and P['pre_state_carries_post_grade']
          and cl['pre_state']['3']['v3_reading'] == 'ONE-HISTORY' and cl['pre_state']['3']['reading'] == 'TIED')
    qu = LK[LK.link == 'queue'].set_index('chain')
    check('queue chain 1: a False saved before the calculation -> not_evaluated, the masked-netting history kept (TIED)',
          qu.loc[1, 'alt_netting_state'] == 'not_evaluated' and 'masked_netting_not_evaluated' in str(qu.loc[1, 'alternatives'])
          and qu.loc[1, 'reading'] == 'TIED' and str(qu.loc[1, 'alt_level_states']) == '100:not_evaluated', qu.loc[1].to_dict())
    check('queue chain 2: an evaluated False (earliest record, zero sites) -> excluded, ONE-HISTORY (positive control: the instrument can exclude)',
          qu.loc[2, 'alt_netting_state'] == 'excluded' and qu.loc[2, 'reading'] == 'ONE-HISTORY', qu.loc[2].to_dict())
    check('queue chain 4: a gap level without an earliest record is not_evaluated beside possible levels -> possible (6 histories)',
          qu.loc[4, 'alt_netting_state'] == 'possible' and '303:not_evaluated' in str(qu.loc[4, 'alt_level_states']) and int(qu.loc[4, 'n_histories']) == 6,
          qu.loc[4].to_dict())
    check('claims: no excluded alt-netting on a not-evaluated row; exclusion only with an evaluated False on every level',
          P['no_false_alt_netting_on_not_evaluated_row'] and P['masked_netting_kept_unless_evaluated_false'])
    tab = CH.parse_table(md)
    check('summary table recomputed from the graded records (chain 3: grade G1_conflict, PRE-STATE TIED, TRANSITION PENDING)',
          P['summary_table_equals_graded_records'] and len(tab) == 6 and tab[3]['cells'][:3] == [G1, 'TIED', 'PENDING'], tab.get(3))
    check('claims agreement = recomputed from the per-link records; the AGREEMENT lines equal them',
          all(cl['agreement'][str(k)]['first_branch'] == CH3.agreement([x for x in res['links'] if x['chain'] == k])['first_branch'] for k in range(1, 7))
          and P['agreement_lines_equal_graded_records'] and cl['agreement']['3']['first_branch'] == 'pre_state')
    B = CH3.parse_md(md)
    vt = ['\n'.join(B[(k, 'VALUE')]['lines']) for k in range(1, 7)]
    check("value blocks: 'gross c/sh' on every gross markout, none bare after net $",
          all('gross c/sh' in x and not CH.BARE_RE.search(x) for x in vt) and P['gross_markouts_labelled'])
    check('draft records the six pending transition links (no_pending_link False) and returns the predicates it writes',
          not P['no_pending_link'] and len(cl['pending']) == 6 and res['claims']['predicates'] == P)


def s_refusals(world):
    rr, dr, _ = world
    out = rr / 'chains/chains_v4_draft'
    expect('overwrite: a second draft into the same directory is refused (write-once)', lambda: run_q(rr, dr, out), 'write-once')
    o2 = rr / 'chains/run_nott'
    ok, info = refuses(lambda: run_q(rr, dr, o2, mode='run'), 'touch_transition receipt absent')
    check("run without touch_transition/receipt.json refuses before any write, naming it and 'use run --draft'",
          ok and 'use run --draft' in info and not o2.exists(), info)
    with Mut('alt_false'):
        expect('a v4 mutant outside the selftest is refused', lambda: CH.run(mode='draft', rroot=rr, droot=dr, out=rr / 'chains/m0', require_freeze=False,
                                                                             log=lambda *a: None), 'planted mutant')
    with Mut('bogus'):
        expect('an unknown v4 mutant is refused', lambda: run_q(rr, dr, rr / 'chains/m1'), 'unknown MT_CH4_MUT')
    with Env('MT_CH3_MUT', 'tied_as_identified'):
        expect('a v3 mutant (MT_CH3_MUT) is refused by chains_v4', lambda: run_q(rr, dr, rr / 'chains/m2'), 'MT_CH3_MUT')
    check('no refused attempt left an output', not any((rr / 'chains' / n).exists() for n in ('m0', 'm1', 'm2', 'run_nott')))


def s_mutants(world):
    rr, dr, _ = world
    for m, needle in (('v3_pre_state_rule', 'never ONE-HISTORY unless the grade is G2'),
                      ('alt_false', 'a not-evaluated alt-netting state read as False'),
                      ('summary_ungraded', 'differs from the graded link records'),
                      ('v3_value_label', "without the 'gross c/sh' label")):
        o = rr / f'chains/mut_{m}'
        with Mut(m):
            ok, info = refuses(lambda: run_q(rr, dr, o), needle)
        check(f'mutant {m} caught by the verifier before any write',
              ok and not o.exists() and not any(p.name.startswith(f'.mut_{m}') for p in (rr / 'chains').iterdir()), info)


def s_run(world):
    rr, dr, F = world
    out = rr / 'chains/chains_v4'
    run_q(rr, dr, out, mode='run')
    cl = json.loads((out / 'claims.json').read_text())
    LK = S3.links_of(out)
    md = (out / 'chains.md').read_text()
    check('run with every sibling bound (touch_transition included): every predicate holds, no pending link, reconciliations and bindings ok',
          all(cl['predicates'].values()) and cl['reconciliation_all_hold'] and cl['bindings_all_ok'] and not cl['pending'],
          [k for k, v in cl['predicates'].items() if not v] + [k for k, v in cl['reconciliation'].items() if not v['holds']])
    tr = LK[LK.link == 'transition'].set_index('chain')
    B = CH3.parse_md(md)
    t4 = ' '.join(B[(4, 'TRANSITION')]['lines'])
    a4 = str(tr.loc[4, 'alternatives']).split('|')
    check('chain 4 (G2): its creation inside the post window matched to the census pair improve_bid and the response episode (restoring, 1 tick, '
          '500 ms) with candidate grade candidate_constrained (this order); TIED with joined / masked kept',
          tr.loc[4, 'reading'] == 'TIED' and a4 == ['at_creation:level_403:improve_bid:response_to_EP13B(restoring)', CH.JOINED, CH.MASKED]
          and 'candidate grade candidate_constrained (ok; candidate' in t4 and 'this_order' in t4 and 'by 1 tick(s)' in t4 and 'after 500 ms' in t4
          and tr.loc[4, 'evidence_grade'] == 'transition_context', (a4, t4))
    check('chain 4: creations before the window are context lines only (level 103 without a census pair keeps the status no_census_pair)',
          'level 103 created' in t4 and 'before the post window; census pair: no_census_pair' in t4 and not any('level_103' in x for x in a4), t4)
    check('chain 2 (G2, creation before the window, masked netting excluded): TRANSITION ONE-HISTORY joined_shown_cell (positive control)',
          tr.loc[2, 'reading'] == 'ONE-HISTORY' and str(tr.loc[2, 'alternatives']) == CH.JOINED, tr[['reading', 'alternatives']].to_dict())
    check('transition: G1 / G4 chains (1, 3, 5) keep unresolved_outside_window; chain 6 keeps a creation without a census pair as a status; a '
          'censored end at the creation frame is not a response (chain 1: two_side/widen, no response)',
          all(CH.UNRESOLVED in str(tr.loc[k, 'alternatives']).split('|') for k in (1, 3, 5))
          and 'at_creation:level_105:no_census_pair' in str(tr.loc[6, 'alternatives']).split('|')
          and 'at_creation:level_100:two_side/widen' in str(tr.loc[1, 'alternatives']).split('|'), tr['alternatives'].to_dict())
    t3 = ' '.join(B[(3, 'TRANSITION')]['lines'])
    check('chain 3: a response episode at a creation before the window is shown as context (EP12B, partial), never a retained history',
          'response episode EP12B' in t3 and '(partial)' in t3 and 'EP12B' not in str(tr.loc[3, 'alternatives']), t3)
    rec = json.loads((out / 'receipt.json').read_text())
    sb = rec['siblings']
    check('receipt binds the touch_transition files and posting_opportunity events.csv (chain markets read), with the resolved interface',
          sorted(sb['touch_transition']['files']) == ['census', 'episodes'] and sb['touch_transition']['files']['census']['rows_used'] == 6
          and sb['posting_opportunity']['files']['events']['rows_used'] == 4 and rec['interfaces']['touch_transition']['resolved']['census']['cls'] == 'cls',
          sb)


def s_falsifiers(w1, w2):
    rr, dr, _ = w1
    out = rr / 'chains/chains_v4_draft'
    run_q(rr, dr, out)
    tr = S3.links_of(out)
    tr = tr[tr.link == 'transition']
    check('a touch_transition file not matching its receipt: TRANSITION PENDING with the reason (draft)',
          (tr.status == 'pending').all() and tr.pending_reason.str.contains('differs from its receipt').all(), tr.pending_reason.iloc[0])
    o = rr / 'chains/chains_v4'
    ok, info = refuses(lambda: run_q(rr, dr, o, mode='run'), 'differs from its receipt')
    check('a run refuses it before any write', ok and not o.exists(), info)
    rr2, dr2, _ = w2
    out2 = rr2 / 'chains/chains_v4_draft'
    run_q(rr2, dr2, out2)
    tr2 = S3.links_of(out2)
    tr2 = tr2[tr2.link == 'transition']
    check('a touch_transition census without its class column: TRANSITION PENDING naming it (draft)',
          (tr2.status == 'pending').all() and tr2.pending_reason.str.contains('required columns missing').all()
          and tr2.pending_reason.str.contains('cls').all(), tr2.pending_reason.iloc[0])
    o2 = rr2 / 'chains/chains_v4'
    ok2, info2 = refuses(lambda: run_q(rr2, dr2, o2, mode='run'), 'required columns missing')
    check('a run refuses the schema mismatch before any write', ok2 and not o2.exists(), info2)


def main():
    tmp = Path(tempfile.mkdtemp(prefix='chains_v4_selftest_'))
    for v in ('MT_CH4_MUT', 'MT_CH3_MUT'):
        os.environ.pop(v, None)
    try:
        section('units', s_units)
        W = {}

        def worlds():
            W['A'] = v4_world(tmp / 'A')
            W['B'] = v4_world(tmp / 'B', tt=True)
            W['F1'] = falsify(tmp / 'B', tmp / 'F1', 'sha')
            W['F2'] = falsify(tmp / 'B', tmp / 'F2', 'schema')
        section('world', worlds)
        if 'A' in W:
            section('draft', s_draft, W['A'])
            section('refusals', s_refusals, W['A'])
            section('mutants', s_mutants, W['A'])
        if 'B' in W:
            section('run', s_run, W['B'])
        if 'F2' in W:
            section('falsifiers', s_falsifiers, W['F1'], W['F2'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_bad = sum(not ok for _, ok, _ in RES)
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  :: {info}'))
    print(json.dumps(dict(checks=len(RES), failed=n_bad, cpu_s=round(__import__('time').process_time(), 3), vmhwm_mib=round(CH3.hwm(), 1))))
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
