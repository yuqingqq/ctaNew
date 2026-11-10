"""Controls for the actual A4b wire schema, status clocks and full action cohort."""
import copy
import gzip
import json
import subprocess
import sys
import pytest
from predictfun.research import pf_wallet_calibration as c

BASE=1794096000000000000  # synthetic clocks on consumed Nov8, not market data
HASH='0x'+'11'*32
OTHER='0x'+'22'*32


def step(ms,inp):return dict(frame=dict(at=BASE+ms*1_000_000,input=inp),i=0,type='step',version=1)
def event(ms,ty,h=HASH,sid=None,stamp=None):
    data=dict(type=ty,orderId='7' if h==HASH else '8',orderHash=h,timestamp=BASE//1_000_000+(stamp if stamp is not None else ms-2),
              details=dict(marketId=123,outcome='YES',quoteType='BID',quantity='5',price='0.5',quantityFilled='0',strategyType='LIMIT'))
    if sid is not None:
        data.update(settlementId=sid,fill=dict(executedSizeWei='2000000000000000000',executedPriceWei='500000000000000000',executedValueWei='1000000000000000000'))
    return dict(recv_unix_ns=BASE+ms*1_000_000,data=data)


def fixture(dropped=0):
    kernel=[]
    for cid,h in [(1,HASH),(2,OTHER),(3,None)]:
        kernel.append(step(10+cid,dict(Control=dict(Place=dict(cid=cid,market=0,side='BuyUp',price=500000,size=5000000)))))
        if h:
            kernel.append(step(15+cid,dict(Exec=dict(Signed=dict(cid=cid,order_id=list(bytes.fromhex(h[2:])))))))
            kernel.append(step(20+cid,dict(Exec=dict(Accepted=dict(cid=cid)))))
    for ms,status in [(35,'Matched'),(105,'Confirmed')]:
        kernel.append(step(ms,dict(Exec=dict(Fill=dict(order_id=list(bytes.fromhex(HASH[2:])),trade_id='s1',status=status,size=2000000,price=500000,match_time=BASE)))))
    for ms,h in [(95,HASH),(115,OTHER)]:
        kernel.append(step(ms,dict(Exec=dict(UserCancel=dict(order_id=list(bytes.fromhex(h[2:])))))))
    kernel.sort(key=lambda x:x['frame']['at'])
    wallet=[dict(recv_unix_ns=BASE,session='connected'),dict(recv_unix_ns=BASE+1_000_000,session='live',heartbeats=1,pongs=0),
            event(22,'orderAccepted'),event(23,'orderAccepted',OTHER),event(30,'orderTransactionSubmitted',sid='s1',stamp=28),
            event(31,'orderTransactionSubmitted',sid='s1',stamp=28),event(90,'orderCancelled'),
            event(100,'orderTransactionSuccess',sid='s1',stamp=99),event(110,'orderCancelled',OTHER),
            dict(recv_unix_ns=BASE+200_000_000,session='ended',why='stopped',heartbeats=1,pongs=0)]
    wallet.append(dict(written=len(wallet),dropped=dropped))
    return kernel,wallet


def analyze(kernel,wallet):
    orders,fills=c.kernel_actions(kernel)
    e,s,f=c.wallet_events(wallet,BASE,BASE+300_000_000)
    return c.reconcile(orders,fills,e,s,f)


def test_positive_full_action_population_and_status_specific_clocks():
    r=analyze(*fixture())
    assert r['summary']['order_actions']==3 and r['summary']['deduplicated_fill_legs']==1
    assert [a['calibration_status'] for a in r['actions']]==['confirmed_filled','observed_terminal_without_fill','not_signed']
    f=r['fills'][0]
    assert f['quantity']=='2' and f['status_clocks']['Matched']['count']==2
    assert f['status_clocks']['Matched']['event_timestamps_ms']==[BASE//1_000_000+28]
    assert f['status_clocks']['Confirmed']['event_timestamps_ms']==[BASE//1_000_000+99]
    assert f['kernel_notice_count']==2 and not f['execution_time_verified']
    assert not r['summary']['execution_timing_calibrated'] and not r['summary']['queue_model_calibrated']


@pytest.mark.parametrize('mode',['dropped','unfinished','gap','late_connected','not_live'])
def test_missing_capture_cannot_be_a_verified_unfilled_order(mode):
    kernel,wallet=fixture(1 if mode=='dropped' else 0)
    if mode=='unfinished':wallet.pop()
    if mode=='gap':wallet[-2]['why']='silent'
    if mode=='late_connected':
        wallet[0]['recv_unix_ns']=BASE+15_000_000;wallet[1]['recv_unix_ns']=BASE+16_000_000
    if mode=='not_live':wallet[1]['session']='not_live'
    r=analyze(kernel,wallet)
    assert r['actions'][1]['calibration_status']=='capture_incomplete'


@pytest.mark.parametrize('mutation',['quantity','hash','footer','clock','unknown_type'])
def test_corrupted_identity_schema_and_clock_refused(mutation):
    kernel,wallet=fixture()
    if mutation=='quantity':wallet[4]['data']['fill']['executedSizeWei']='3000000000000000000'
    if mutation=='hash':wallet[4]['data']['orderHash']='broken'
    if mutation=='footer':wallet[-1]['written']+=1
    if mutation=='clock':wallet[4]['recv_unix_ns']=BASE
    if mutation=='unknown_type':wallet[4]['data']['type']='unexpected'
    with pytest.raises(ValueError):analyze(kernel,wallet)


def test_submitted_timestamp_outside_lifecycle_is_status_not_fitted_offset():
    kernel,wallet=fixture()
    for i in (4,5):wallet[i]['data']['timestamp']=BASE//1_000_000+500
    r=analyze(kernel,wallet)
    assert 'submitted_timestamp_outside_local_lifecycle' in r['fills'][0]['issues']
    assert r['actions'][0]['calibration_status']=='fill_identity_or_status_issue'


def test_strict_journal_export_header_and_old_kind_are_supported():
    kernel,wallet=fixture()
    modern=[dict(frame={'build':'synthetic'},i=0,type='header',version=1)]+kernel
    modern.append(dict(frame={'ignored':'outbox'},i=len(modern),type='outbox',version=1))
    assert analyze(modern,wallet)['summary']['order_actions']==3
    old=[dict(frame=r['frame'],kind='step') for r in kernel]
    assert analyze(old,wallet)['summary']['order_actions']==3


def test_kernel_fill_missing_wallet_leg_is_not_silently_unfilled():
    kernel,wallet=fixture();wallet=[row for row in wallet if row.get('data',{}).get('type') not in c.FILL_TYPES]
    wallet[-1]['written']=len(wallet)-1
    r=analyze(kernel,wallet)
    assert r['actions'][0]['calibration_status']=='kernel_fill_without_wallet_leg'
    assert len(r['unmatched_kernel_fills'])==2


def test_out_of_scope_payload_is_refused_before_decoding(tmp_path):
    p=tmp_path/'wallet.jsonl';p.write_bytes(b'{"recv_unix_ns":'+str(BASE+999_000_000).encode()+b',"data":INVALID}\n')
    with pytest.raises(ValueError,match='outside declared'):list(c.records(p,BASE,BASE+300_000_000,'wallet'))


def test_cli_artifacts_and_protected_window_refusal(tmp_path):
    kernel,wallet=fixture();scope=dict(start_ns=BASE,end_ns=BASE+300_000_000)
    for name,rows in [('kernel',kernel),('wallet',wallet)]:
        p=tmp_path/f'{name}.jsonl.gz'
        with gzip.open(p,'wt') as f:
            for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
        scope[name]=dict(path=str(p),sha256=c.digest(p))
    path=tmp_path/'scope.json';path.write_text(json.dumps(scope))
    cmd=[sys.executable,'-m','predictfun.research.pf_wallet_calibration','--scope',str(path),'--out',str(tmp_path/'result')]
    good=subprocess.run(cmd,capture_output=True,text=True)
    assert good.returncode==0,good.stderr
    report=json.loads((tmp_path/'result/receipt.json').read_text())
    assert report['summary']['order_actions']==3
    for name,h in report['artifacts'].items():assert c.digest(tmp_path/'result'/name)==h
    assert subprocess.run(cmd,capture_output=True).returncode!=0
    scope.update(start_ns=1794182400000000000,end_ns=1794182400001000000)
    path.write_text(json.dumps(scope))
    bad=subprocess.run(cmd[:-1]+[str(tmp_path/'protected')],capture_output=True,text=True)
    assert bad.returncode!=0 and 'protected Nov9' in bad.stderr and not (tmp_path/'protected').exists()
