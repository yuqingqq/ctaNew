"""Protocol controls and refusal cases; no strategy outcomes or credentials."""
import asyncio
import gzip
import json
from datetime import datetime, timezone

import pytest
from aiohttp import web

from data_collectors import predict_fun_recorder as pf
from data_collectors.predict_fun_inspect import inspect_run, book_diagnostics


def category(slug):
    start = int(slug.rsplit("-", 1)[1])
    minutes = int(slug.split("-")[-2][:-1])
    return {"slug": slug, "startsAt": datetime.fromtimestamp(start, timezone.utc).isoformat(),
            "endsAt": datetime.fromtimestamp(start + minutes * 60, timezone.utc).isoformat(),
            "marketVariant": "CRYPTO_UP_DOWN", "markets": [{"id": start,
            "outcomes": [{"name": "Up", "indexSet": 1, "onChainId": "123"},
                         {"name": "Down", "indexSet": 2, "onChainId": "456"}]}]}


def records(writer):
    return sorted((json.loads(line) for f in writer.path.glob("*/*.jsonl.gz")
                   for line in gzip.open(f, "rt")), key=lambda row: row["seq"])


def test_window_discovery_rolls_without_missing_next():
    slugs = pf.slugs_at(1792594501, ["btc", "eth"], [5, 15])
    assert len(slugs) == 12
    assert "btc-updown-5m-1792594800" in slugs
    end, ids = pf.category_markets(category(slugs[0]), slugs[0])
    assert end == ids[0] + 300


@pytest.mark.parametrize("bad", [None, "identity", "timestamp", "price", "size", "order", "duplicate", "shape"])
def test_book_depth_checker_positive_and_malformed_refusals(bad):
    data={"marketId":42,"updateTimestampMs":123,"bids":[[0.4,10],[0.3,20]],"asks":[[0.6,10],[0.7,20]]}
    if bad=="identity":data["marketId"]=43
    elif bad=="timestamp":data.pop("updateTimestampMs")
    elif bad=="price":data["bids"][0][0]=1.5
    elif bad=="size":data["bids"][0][1]=float('nan')
    elif bad=="order":data["bids"].reverse()
    elif bad=="duplicate":data["bids"][1][0]=0.4
    elif bad=="shape":data["asks"]=[0.5]
    if bad:
        with pytest.raises(ValueError):book_diagnostics(data,42)
    else:
        assert book_diagnostics(data,42)=={"two_sided":True,"empty":False,"locked_or_crossed":False}
        data["bids"]=[]
        assert book_diagnostics(data,42)["two_sided"] is False


@pytest.mark.parametrize("mutation", ["slug", "mapping", "tokens", "window", "variant", "empty"])
def test_discovery_refuses_ambiguous_identity(mutation):
    slug = "btc-updown-5m-1792594500"
    data = category(slug)
    if mutation == "slug":
        data["slug"] = "eth-updown-5m-1792594500"
    elif mutation == "mapping":
        data["markets"][0]["outcomes"][0]["indexSet"] = 2
    elif mutation == "tokens":
        data["markets"][0]["outcomes"][0]["onChainId"] = 123
    elif mutation == "window":
        data["endsAt"] = data["startsAt"]
    elif mutation == "variant":
        data["marketVariant"] = "DEFAULT"
    else:
        data["markets"] = []
    with pytest.raises(ValueError):
        pf.category_markets(data, slug)


@pytest.mark.parametrize("bad", ["not_object", "not_market", "null_outcomes", "null_name", "numeric_time", "missing_time", "naive_time", "nested_index", "duplicate_tokens", "duplicate_market"])
def test_malformed_nested_category_schema_is_a_recordable_refusal(bad):
    slug="btc-updown-5m-1792594500";data=category(slug)
    if bad=="not_object":data=None
    elif bad=="not_market":data['markets']=[None]
    elif bad=="null_outcomes":data['markets'][0]['outcomes']=None
    elif bad=='null_name':data['markets'][0]['outcomes'][0]['name']=None
    elif bad=='numeric_time':data['endsAt']=123
    elif bad=='missing_time':data.pop('endsAt')
    elif bad=='naive_time':data['endsAt']=data['endsAt'].replace('+00:00','')
    elif bad=='nested_index':data['markets'][0]['outcomes'][0]['indexSet']=[]
    elif bad=='duplicate_tokens':data['markets'][0]['outcomes'][1]['onChainId']='123'
    else:data['markets'].append(data['markets'][0])
    with pytest.raises(ValueError):pf.category_markets(data,slug)


@pytest.mark.parametrize('bad', ['missing_depth', 'wrong_market', 'missing_time', 'bad_level'])
def test_malformed_snapshot_stays_due_until_valid_depth(tmp_path, bad):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.resnapshot_due.add(42)
        book={'marketId':42,'updateTimestampMs':123,'bids':[],'asks':[]}
        if bad=='missing_depth':book.pop('asks')
        elif bad=='wrong_market':book['marketId']=43
        elif bad=='missing_time':book.pop('updateTimestampMs')
        else:book['bids']=[[float('nan'),1]]
        async def get(path,params=None):return {'success':True,'data':book}
        recorder.get=get
        try:
            await recorder.refresh_snapshot(42)
            assert 42 in recorder.resnapshot_due
            book.update(marketId=42,updateTimestampMs=123,bids=[],asks=[])
            await recorder.refresh_snapshot(42)
            assert 42 not in recorder.resnapshot_due
        finally:await writer.close('TEST')
        assert [r['kind'] for r in records(writer)]==['SNAPSHOT_FETCH_FAILED','RESNAPSHOT_FETCHED']
    asyncio.run(exercise())


def test_malformed_book_cannot_refresh_health(tmp_path):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.markets[42]=('btc',100)
        try:
            recorder.observe_book({'type':'M','topic':'predictOrderbook/42','data':{'marketId':42}},pf.clock())
            assert 42 not in recorder.book_receipts
        finally:await writer.close('TEST')
        assert records(writer)[0]['kind']=='BOOK_OBSERVATION_REFUSED'
    asyncio.run(exercise())


def test_missing_mainnet_key_refuses_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.delenv("PREDICT_FUN_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        pf.arguments(["--root", str(tmp_path / "never_created")])
    assert not (tmp_path / "never_created").exists()
    with pytest.raises(SystemExit):
        pf.arguments(["--testnet"])


@pytest.mark.parametrize("include_binance", [False, True])
def test_no_binance_uses_external_collectors_without_starting_auxiliary_feeds(monkeypatch, tmp_path, include_binance):
    async def exercise():
        monkeypatch.setenv("PREDICT_FUN_API_KEY", "synthetic-test-key")
        argv = ["--duration-s", "0.02"] + ([] if include_binance else ["--no-binance"])
        args = pf.arguments(argv)
        writer = pf.RawWriter(tmp_path, {"no_binance": args.no_binance})
        recorder = pf.Recorder(args, writer)
        calls = []
        async def wait():
            await recorder.stop.wait()
        async def binance(feed):
            calls.append(feed)
            await recorder.stop.wait()
        recorder.rest_loop = wait
        recorder.health_loop = wait
        recorder.predict_loop = wait
        recorder.binance_loop = binance
        await recorder.run()
        await writer.close("TEST")
        assert sorted(calls) == (["binance_futures", "binance_spot"] if include_binance else [])
        assert json.loads((writer.path / "manifest.json").read_text())["no_binance"] is not include_binance
    asyncio.run(exercise())


def test_raw_writer_rotation_and_backward_clock_preserve_sequence(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        for ns in (3600_000_000_000, 7200_000_000_000, 3601_000_000_000):
            writer.emit("events", "frame", stamp={"recv_ns": ns, "recv_mono_ns": ns}, raw='{"a":1}')
        await writer.close("TEST")
        rows = records(writer)
        assert [r["seq"] for r in rows] == [0, 1, 2]
        assert all(r["raw"] == '{"a":1}' for r in rows)
        assert len(list(writer.path.glob("events/*.gz"))) == 2
    asyncio.run(exercise())


def test_writer_disk_failure_is_not_success(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        (writer.path / "events").write_text("blocks directory creation")
        writer.emit("events", "positive_control")
        with pytest.raises(RuntimeError, match="persist"):
            await writer.close("TEST")
        assert json.loads((writer.path / "summary.json").read_text())["writer_error"]
    asyncio.run(exercise())


def test_get_allowlist_refuses_private_or_mutating_routes(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        args = pf.arguments(["--testnet", "--rest-only"])
        recorder = pf.Recorder(args, writer)
        for path in ("/v1/orders", "/v1/auth", "/v1/positions", "/v1/orders/remove"):
            with pytest.raises(ValueError, match="allowed"):
                await recorder.get(path)
        await writer.close("TEST")
    asyncio.run(exercise())


def test_match_pagination_keeps_raw_duplicates_and_refuses_cursor_loop(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        ts = "2026-10-21T10:00:00+00:00"
        recorder.match_cutoff[42] = datetime.fromisoformat(ts).timestamp() - 100
        calls = []
        async def get(path, params):
            calls.append(dict(params))
            return {"data": [{"market": {"id": 42}, "executedAt": ts}], "cursor": "same"}
        recorder.get = get
        await recorder.matches_once(42)
        await writer.close("TEST")
        assert len(calls) == 2 and calls[1]["after"] == "same"
        assert records(writer)[-1]["reason"] == "CURSOR_LOOP"
        assert recorder.match_cutoff[42] == datetime.fromisoformat(ts).timestamp() - 100
    asyncio.run(exercise())


def test_match_traversal_positive_and_wrong_market_refusal(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        recorder.match_cutoff[42] = 1
        responses = iter([
            {"data": [{"market": {"id": 42}, "executedAt": "2026-10-21T00:00:00Z"}], "cursor": "second"},
            {"data": [], "cursor": None}])
        async def get(path, params):
            return next(responses)
        recorder.get = get
        await recorder.matches_once(42)
        assert recorder.match_cutoff[42] > 1
        async def wrong_market(path, params):
            return {"data": [{"market": {"id": 43}, "executedAt": "2026-10-21T00:00:00Z"}]}
        recorder.get = wrong_market
        with pytest.raises(ValueError, match="identity"):
            await recorder.matches_once(42)
        await writer.close("TEST")
        assert records(writer)[0]["kind"] == "MATCH_RANGE_TRAVERSED"
        assert records(writer)[0]["completeness_certified"] is False
    asyncio.run(exercise())


def test_match_boundary_second_is_fully_traversed(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        recorder.match_cutoff[42] = 10.0
        calls = []
        def row(t):
            return {"market": {"id": 42}, "executedAt": datetime.fromtimestamp(t, timezone.utc).isoformat()}
        pages = iter([{"data": [row(12), row(10)], "cursor": "boundary"},
                      {"data": [row(10), row(9)], "cursor": None}])
        async def get(path, params):
            calls.append(dict(params))
            return next(pages)
        recorder.get = get
        await recorder.matches_once(42)
        await writer.close("TEST")
        assert len(calls) == 2, "stopping at equality skips remaining executions in that second"
        assert calls[1]["after"] == "boundary"
    asyncio.run(exercise())


def test_page_limit_resumes_backlog_instead_of_repeating_head(tmp_path):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only','--match-pages','1']),writer)
        recorder.match_cutoff[42]=1.0
        calls=[]
        def page(ts,cursor):
            return {'data':[{'market':{'id':42},'executedAt':datetime.fromtimestamp(ts,timezone.utc).isoformat()}],'cursor':cursor}
        pages={None:page(200,'tail'),'tail':page(150,'last'),'last':page(0,None)}
        async def get(path,params):
            calls.append(params.get('after'));return pages[params.get('after')]
        recorder.get=get
        try:
            await recorder.matches_once(42)
            assert recorder.match_cutoff[42]==1.0
            await recorder.matches_once(42)
            assert recorder.match_cutoff[42]==1.0
            await recorder.matches_once(42)
            assert calls==[None,'tail','last']
            assert recorder.match_cutoff[42]==140.0
            assert 42 not in recorder.match_backlogs
        finally:await writer.close('TEST')
    asyncio.run(exercise())


def test_empty_match_page_with_cursor_is_not_the_end_of_traversal(tmp_path):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.match_cutoff[42]=1.0
        calls=[]
        async def get(path,params):
            calls.append(params.get('after'))
            if params.get('after') is None:return {'data':[],'cursor':'nonempty-next'}
            return {'data':[{'market':{'id':42},'executedAt':'2026-10-21T00:00:00Z'}],'cursor':None}
        recorder.get=get
        try:
            await recorder.matches_once(42)
            assert calls==[None,'nonempty-next']
            assert recorder.match_cutoff[42]>1.0
        finally:await writer.close('TEST')
    asyncio.run(exercise())


@pytest.mark.parametrize('fault',['timeout','http','cursor_refused','schema','order','cursor_loop'])
def test_backlog_failure_preserves_cutoff_and_recovers(tmp_path,fault):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only','--match-pages','1']),writer)
        recorder.match_cutoff[42]=1.0
        calls=[];failed=False
        def row(ts):return {'market':{'id':42},'executedAt':datetime.fromtimestamp(ts,timezone.utc).isoformat()}
        async def get(path,params):
            nonlocal failed
            calls.append(params.get('after'))
            if params.get('after') is None:return {'data':[row(200)],'cursor':'tail'}
            if not failed:
                failed=True
                if fault=='timeout':raise asyncio.TimeoutError()
                if fault=='http':return None
                if fault=='cursor_refused':raise pf.MatchCursorRefused('synthetic refusal')
                if fault=='schema':return {'data':[row(10000),None],'cursor':None}
                if fault=='order':return {'data':[row(201)],'cursor':'next'}
                if fault=='cursor_loop':return {'data':[row(200)],'cursor':'tail'}
            return {'data':[row(0)],'cursor':None}
        recorder.get=get
        try:
            await recorder.matches_once(42)
            if fault in ('timeout','schema'):
                with pytest.raises((asyncio.TimeoutError,ValueError)):await recorder.matches_once(42)
            else:await recorder.matches_once(42)
            assert recorder.match_cutoff[42]==1.0
            if fault in ('timeout','http'):
                assert 42 in recorder.match_backlogs
            else:
                assert 42 not in recorder.match_backlogs
                await recorder.matches_once(42)  # Refused/changed cursors start from the head.
                assert calls[-1] is None
            await recorder.matches_once(42)
            assert recorder.match_cutoff[42]==140.0
            assert 42 not in recorder.match_backlogs
        finally:await writer.close('TEST')
    asyncio.run(exercise())


def test_match_batch_time_budget_yields_and_resumes(monkeypatch,tmp_path):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only','--match-batch-s','1']),writer)
        recorder.match_cutoff[42]=1.0
        elapsed=[0.0];calls=[]
        async def get(path,params):
            elapsed[0]+=.6;calls.append(params.get('after'))
            ts=200-len(calls)
            return {'data':[{'market':{'id':42},'executedAt':datetime.fromtimestamp(ts,timezone.utc).isoformat()}],
                    'cursor':str(len(calls))}
        recorder.get=get
        try:
            with monkeypatch.context() as scoped:
                scoped.setattr(pf.time,'monotonic',lambda:elapsed[0])
                await recorder.matches_once(42)
                assert calls==[None,'1']
                assert recorder.match_cutoff[42]==1.0
                await recorder.matches_once(42)
                assert calls[2]=='2'
        finally:await writer.close('TEST')
        assert all(r['reason']=='BATCH_TIME_LIMIT' and r['resume_queued'] for r in records(writer))
    asyncio.run(exercise())


def test_cold_start_rediscovers_all_retained_windows_first_live(monkeypatch,tmp_path):
    async def exercise():
        now=1792594500+100
        monkeypatch.setattr(pf.time,'time',lambda:now)
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        calls=[]
        async def get(path,params=None):
            if '/categories/' in path:
                slug=path.rsplit('/',1)[1];calls.append(slug)
                return {'data':category(slug)}
            mid=int(path.split('/')[-2])
            return {'data':{'marketId':mid,'updateTimestampMs':1,'bids':[],'asks':[]}}
        recorder.get=get
        try:await recorder.discover_once()
        finally:await writer.close('TEST')
        assert calls[:2]==['btc-updown-5m-1792594800','btc-updown-5m-1792594500']
        assert 'btc-updown-5m-1792593300' in calls
        assert all(int(slug.rsplit('-',1)[1])+300+recorder.args.retention_s>=now for slug in calls)
    asyncio.run(exercise())


@pytest.mark.parametrize('retention',[60,1800])
def test_retained_discovery_never_reintroduces_expired_windows(retention):
    now=1792594600
    slugs=pf.retained_slugs_at(now,['btc'],[5],retention)
    assert 'btc-updown-5m-1792594500' in slugs
    assert 'btc-updown-5m-1792594800' in slugs
    assert all(int(slug.rsplit('-',1)[1])+300+retention>=now for slug in slugs)
    if retention==60:assert 'btc-updown-5m-1792594200' not in slugs


def test_retirement_clears_backlog_with_explicit_incomplete_status(tmp_path,monkeypatch):
    async def exercise():
        monkeypatch.setattr(pf.time,'time',lambda:1792594600)
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.markets[42]=('expired',1)
        recorder.match_backlogs[42]={'pages':2}
        recorder.match_cutoff[42]=1
        async def get(path,params=None):return None
        recorder.get=get
        try:await recorder.discover_once()
        finally:await writer.close('TEST')
        assert not recorder.markets and not recorder.match_backlogs and not recorder.match_cutoff
        assert any(r.get('reason')=='RETIRED_WITH_BACKLOG' for r in records(writer))
    asyncio.run(exercise())


@pytest.mark.parametrize('status',[200,400])
def test_http_match_cursor_positive_and_refusal(tmp_path,status):
    async def exercise():
        async def endpoint(request):
            return web.json_response({'success':status==200,'data':[]},status=status)
        app=web.Application();app.router.add_get('/v1/orders/matches',endpoint)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.base=f'http://127.0.0.1:{port}'
        try:
            async with pf.aiohttp.ClientSession() as session:
                recorder.session=session
                if status==400:
                    with pytest.raises(pf.MatchCursorRefused):
                        await recorder.get('/v1/orders/matches',{'marketId':'42','after':'expired'})
                else:
                    assert (await recorder.get('/v1/orders/matches',{'marketId':'42','after':'valid'}))['data']==[]
        finally:
            await writer.close('TEST');await runner.cleanup()
        assert records(writer)[0]['status']==status
    asyncio.run(exercise())


@pytest.mark.parametrize('bad',['json','envelope','match_page','match_row'])
def test_inspector_preserves_malformed_rest_as_diagnostics(tmp_path,bad):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        raw={'json':'[','envelope':'null','match_page':'{"success":true,"data":null}',
             'match_row':'{"success":true,"data":[null]}'}[bad]
        writer.emit('rest','response',path='/v1/orders/matches',params={'marketId':'42'},status=200,raw=raw)
        await writer.close('TEST')
        report=inspect_run(writer.path)
        assert report['n_records']==1
        key='malformed_rest_envelopes' if bad in ('json','envelope') else 'malformed_match_pages' if bad=='match_page' else 'malformed_match_rows'
        assert report['protocol_diagnostics'][key]==1
        assert report['data_quality_certified'] is False
    asyncio.run(exercise())


@pytest.mark.parametrize('bad',[False,True])
def test_inspector_rest_snapshot_positive_and_refusal(tmp_path,bad):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        book={'marketId':42,'updateTimestampMs':1,'bids':[],'asks':[]}
        if bad:book.pop('asks')
        writer.emit('rest','response',path='/v1/markets/42/orderbook',params=None,status=200,
                    raw=json.dumps({'success':True,'data':book}))
        await writer.close('TEST')
        report=inspect_run(writer.path)
        assert report['protocol_diagnostics']['rest_books_checked']==1
        assert report['protocol_diagnostics']['malformed_rest_books']==int(bad)
    asyncio.run(exercise())


@pytest.mark.parametrize('bad',[None,'identity','time'])
def test_inspector_match_positive_identity_and_time_refusal(tmp_path,bad):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        row={'market':{'id':43 if bad=='identity' else 42},'executedAt':None if bad=='time' else '2026-10-21T00:00:00Z','transactionHash':'synthetic'}
        writer.emit('rest','response',path='/v1/orders/matches',params={'marketId':'42'},status=200,
                    raw=json.dumps({'success':True,'data':[row]}))
        await writer.close('TEST')
        report=inspect_run(writer.path)
        assert report['match_raw_rows_including_overlap']==1
        assert report['protocol_diagnostics']['malformed_match_rows']==int(bad=='identity')
        assert report['match_rows_with_unreadable_executedAt']==int(bad=='time')
        assert report['n_distinct_market_transaction_pairs']==int(bad!='identity')
    asyncio.run(exercise())


@pytest.mark.parametrize('bad', ['naive_time', 'nested_cursor', 'empty_cursor'])
def test_match_schema_refusal_does_not_advance_cutoff(tmp_path,bad):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.match_cutoff[42]=1.0
        payload={'data':[{'market':{'id':42},'executedAt':'2026-10-21T10:00:00Z'}],'cursor':None}
        if bad=='naive_time':payload['data'][0]['executedAt']='2026-10-21T10:00:00'
        elif bad=='nested_cursor':payload['cursor']=[]
        else:payload['cursor']=''
        async def get(path,params):return payload
        recorder.get=get
        try:
            with pytest.raises(ValueError):await recorder.matches_once(42)
            assert recorder.match_cutoff[42]==1.0
        finally:await writer.close('TEST')
    asyncio.run(exercise())


def test_rediscovery_refuses_reusing_market_id_for_another_window(tmp_path,monkeypatch):
    async def exercise():
        monkeypatch.setattr(pf.time,'time',lambda:1792594600)
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        async def get(path,params=None):
            if '/categories/' in path:
                data=category(path.rsplit('/',1)[1]);data['markets'][0]['id']=42
                return {'data':data}
            return {'data':{'marketId':42,'updateTimestampMs':1,'bids':[],'asks':[]}}
        recorder.get=get
        try:await recorder.discover_once()
        finally:await writer.close('TEST')
        assert len(recorder.markets)==1
        assert sum(r['kind']=='CATEGORY_REFUSED' for r in records(writer))==7
    asyncio.run(exercise())


def test_failed_initial_snapshot_remains_due_for_retry(tmp_path,monkeypatch):
    async def exercise():
        monkeypatch.setattr(pf.time,'time',lambda:1792594600)
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        async def get(path, params=None):
            if "/categories/" in path:
                return {"success": True, "data": category(path.rsplit("/", 1)[1])}
            return None
        recorder.get = get
        await recorder.discover_once()
        await writer.close("TEST")
        assert len(recorder.markets) == 8
        assert recorder.resnapshot_due == set(recorder.markets)
    asyncio.run(exercise())


def test_one_bad_category_does_not_block_other_windows(tmp_path,monkeypatch):
    async def exercise():
        monkeypatch.setattr(pf.time,'time',lambda:1792594600)
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        bad = sorted(pf.slugs_at(pf.time.time(), ["btc"], [5]))[0]
        async def get(path, params=None):
            if path.endswith(bad):
                raise ValueError("malformed category response")
            if "/categories/" in path:
                return {"success": True, "data": category(path.rsplit("/", 1)[1])}
            return {"success": True, "data": {}}
        recorder.get = get
        try:
            await recorder.discover_once()
        finally:
            await writer.close("TEST")
        assert len(recorder.markets) == 7
        assert any(r["kind"] == "CATEGORY_FETCH_FAILED" for r in records(writer))
    asyncio.run(exercise())


def test_snapshot_retry_clears_only_after_success_and_auth_failure_propagates(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        recorder.resnapshot_due.add(42)
        responses = iter([asyncio.TimeoutError(), {"success": True, "data": {"marketId":42,"updateTimestampMs":1,"bids":[],"asks":[]}}])
        async def get(path, params=None):
            response = next(responses)
            if isinstance(response, Exception):raise response
            return response
        recorder.get = get
        await recorder.refresh_snapshot(42)
        assert 42 in recorder.resnapshot_due
        await recorder.refresh_snapshot(42)
        assert 42 not in recorder.resnapshot_due
        async def auth_refusal(path, params=None):raise pf.FatalAccess("401")
        recorder.get = auth_refusal
        with pytest.raises(pf.FatalAccess):await recorder.refresh_snapshot(42)
        await writer.close("TEST")
        assert [r["kind"] for r in records(writer)] == ["SNAPSHOT_FETCH_FAILED", "RESNAPSHOT_FETCHED"]
    asyncio.run(exercise())


def test_bad_match_market_does_not_starve_other_markets(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        recorder.markets = {1: ("older", pf.time.time()+100), 2: ("newer", pf.time.time()+200)}
        calls = []
        async def discover():pass
        async def matches(mid):
            calls.append(mid)
            if mid==2:raise ValueError("poisoned match row")
            recorder.stop.set()
        recorder.discover_once = discover
        recorder.matches_once = matches
        try:
            await asyncio.wait_for(recorder.rest_loop(), timeout=0.5)
        finally:
            await writer.close("TEST")
        assert calls == [2,1]
        assert any(r["kind"]=="MATCH_POLL_INCOMPLETE" and r["market_id"]==2 for r in records(writer))
    asyncio.run(exercise())


def test_health_does_not_let_heartbeats_refresh_book_age(monkeypatch, tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        recorder.markets[42] = ("btc", 100)
        recorder.observe_book({"type":"M","topic":"predictOrderbook/42", "data":{"marketId":42,"bids":[],"asks":[],"updateTimestampMs":10}}, {"recv_ns":1,"recv_mono_ns":1_000_000_000})
        recorder.observe_book({"type":"M","topic":"heartbeat","data":99}, {"recv_ns":99,"recv_mono_ns":9_000_000_000})
        recorder.last_predict_frame={"recv_ns":99,"recv_mono_ns":9_000_000_000}
        with monkeypatch.context() as scoped:
            scoped.setattr(pf.time,"monotonic_ns",lambda:10_000_000_000)
            health=recorder.health_snapshot()
        assert health["last_predict_frame_age_s"]==1
        assert health["books"][0]["book_recv_age_s"]==9
        # Refuse mismatched identity without claiming it refreshed the book.
        recorder.observe_book({"type":"M","topic":"predictOrderbook/42", "data":{"marketId":43}}, {"recv_ns":101,"recv_mono_ns":10_000_000_000})
        assert recorder.book_receipts[42]["recv_ns"]==1
        await writer.close("TEST")
        assert records(writer)[0]["kind"]=="BOOK_OBSERVATION_REFUSED"
    asyncio.run(exercise())


@pytest.mark.parametrize('malformed', [None, '[]', '{"type":"M","topic":null,"data":{}}'])
def test_predict_reconnect_resubscribes_and_queues_resnapshot(monkeypatch, tmp_path, malformed):
    async def exercise():
        connections=[]
        async def websocket(request):
            ws=web.WebSocketResponse();await ws.prepare(request)
            index=len(connections);connections.append([])
            async for msg in ws:
                data=json.loads(msg.data)
                if data["method"]!="subscribe":continue
                connections[index].append(data["params"][0])
                await ws.send_json({"type":"R","requestId":data["requestId"],"success":True})
                if len(connections[index])==len(pf.TOPICS):
                    if index==0:
                        if malformed is not None:await ws.send_str(malformed)
                        await ws.close();break
                    recorder.stop.set()
            return ws
        app=web.Application();app.router.add_get('/ws',websocket)
        runner=web.AppRunner(app);await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(pf,'WS',f'http://127.0.0.1:{port}/ws')
        writer=pf.RawWriter(tmp_path,{})
        recorder=pf.Recorder(pf.arguments(['--testnet','--rest-only']),writer)
        recorder.key='synthetic';recorder.markets[42]=('btc',1e12)
        original_pause=recorder.pause
        async def fast_pause(seconds):await original_pause(min(seconds,0.001))
        recorder.pause=fast_pause
        task=None
        try:
            async with pf.aiohttp.ClientSession() as session:
                recorder.session=session
                task=asyncio.create_task(recorder.predict_loop())
                await asyncio.wait_for(recorder.stop.wait(),timeout=2)
                task.cancel();await asyncio.gather(task,return_exceptions=True)
        finally:
            if task and not task.done():task.cancel();await asyncio.gather(task,return_exceptions=True)
            await writer.close('TEST');await runner.cleanup()
        expected={f'{topic}/42' for topic in pf.TOPICS}
        assert len(connections)==2
        assert all(set(topics)==expected for topics in connections)
        assert 42 in recorder.resnapshot_due
        assert any(r['kind']=='WS_BOUNDARY' and r['status'].startswith('DISCONNECTED') for r in records(writer))
        if malformed is not None:
            assert any(r.get('raw')==malformed for r in records(writer))
            assert any(r['kind']=='WS_FAILURE' and r['error']=='ValueError' for r in records(writer))
    asyncio.run(exercise())


@pytest.mark.parametrize('raw', ['[', '[]', 'null', '{"type":"M","topic":null,"data":{}}', '{"type":"R","requestId":true,"success":true}', '{"type":"M","topic":"heartbeat","data":123}'])
def test_ws_envelope_positive_and_malformed_refusals(raw):
    if 'heartbeat' in raw:
        assert pf.parse_predict_frame(raw)['data']==123
    else:
        with pytest.raises(ValueError):pf.parse_predict_frame(raw)


def test_inspector_reports_retained_bad_ws_envelope(tmp_path):
    async def exercise():
        writer=pf.RawWriter(tmp_path,{})
        for raw in ('[]','[','{"type":"M","topic":"heartbeat","data":123}'):
            writer.emit('predict_ws','frame',connection_id='test',raw=raw)
        await writer.close('TEST')
        report=inspect_run(writer.path)
        assert report['n_records']==3
        assert report['protocol_diagnostics']['malformed_predict_frames']==2
        assert report['predict_topic_counts']=={'heartbeat':1}
    asyncio.run(exercise())


def test_empty_match_polls_do_not_move_cutoff_backward(tmp_path):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        cutoff = recorder.initial_match_cutoff
        async def get(path, params):
            return {"success": True, "data": [], "cursor": None}
        recorder.get = get
        await recorder.matches_once(42)
        await recorder.matches_once(42)
        assert recorder.match_cutoff[42] == cutoff
        await writer.close("TEST")
    asyncio.run(exercise())


@pytest.mark.parametrize("bad", [None, "source", "sequence", "population", "truncated"])
def test_inspector_positive_and_corruption_refusals(tmp_path, bad):
    async def exercise():
        writer = pf.RawWriter(tmp_path, {"network": "SYNTHETIC_CONTROL"})
        writer.emit("events", "MARKET_DISCOVERED", market_id=42)
        await writer.close("TEST")
        if bad == "source":
            (writer.path / "collector_source.py").write_text("mutated")
        elif bad in ("sequence", "population"):
            f = next(writer.path.glob("events/*.gz"))
            row = json.loads(gzip.open(f, "rt").read())
            with gzip.open(f, "wt") as handle:
                if bad == "sequence":
                    handle.write(json.dumps(row) + "\n")
                    handle.write(json.dumps(row) + "\n")
        elif bad == "truncated":
            f = next(writer.path.glob("events/*.gz"))
            f.write_bytes(f.read_bytes()[:-5])
        if bad:
            with pytest.raises((ValueError, EOFError)):
                inspect_run(writer.path)
        else:
            result = inspect_run(writer.path)
            assert result["n_discovered_markets"] == 1 and result["market_ids"] == [42]
            assert result["n_records"] == 1
            assert result["data_quality_certified"] is False
    asyncio.run(exercise())


@pytest.mark.parametrize("status", [101, 451])
def test_binance_public_endpoint_positive_and_refusal(monkeypatch, tmp_path, status):
    async def exercise():
        async def endpoint(request):
            assert "x-api-key" not in request.headers
            assert "btcusdt@bookTicker" in request.query["streams"]
            if status != 101:
                return web.Response(status=status)
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json({"stream": "btcusdt@bookTicker", "data": {"b": "100", "a": "101"}})
            async for msg in ws:
                pass
            return ws
        app = web.Application()
        app.router.add_get("/stream", endpoint)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(pf, "BINANCE_SPOT_WS", f"http://127.0.0.1:{port}/stream")
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(pf.arguments(["--testnet", "--rest-only"]), writer)
        task = None
        try:
            async with pf.aiohttp.ClientSession() as session:
                recorder.session = session
                task = asyncio.create_task(recorder.binance_loop("binance_spot"))
                for _ in range(100):
                    if writer.seq >= 2:
                        break
                    await asyncio.sleep(0.01)
                recorder.stop.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        finally:
            await writer.close("TEST")
            await runner.cleanup()
        rows = records(writer)
        if status == 101:
            assert any(r["source"] == "binance_spot" and "bookTicker" in r["raw"] for r in rows)
        else:
            assert any(r["kind"] == "WS_FAILURE" and r["status"] == 451 for r in rows)
            assert not any(r["source"] == "binance_spot" for r in rows)
    asyncio.run(exercise())


def test_http_access_refusal_is_fatal_and_never_logs_key(monkeypatch, tmp_path):
    async def exercise():
        async def refuse(request):
            return web.json_response({"success": False}, status=401)
        app = web.Application()
        app.router.add_get("/v1/categories/{slug}", refuse)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(pf, "REST", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("PREDICT_FUN_API_KEY", "secret-control-never-record")
        args = pf.arguments(["--probe"])
        writer = pf.RawWriter(tmp_path, {})
        recorder = pf.Recorder(args, writer)
        try:
            with pytest.raises(pf.FatalAccess, match="401"):
                await recorder.run()
        finally:
            await writer.close("TEST")
            await runner.cleanup()
        assert records(writer)[0]["status"] == 401
        assert "secret-control-never-record" not in json.dumps(records(writer))
    asyncio.run(exercise())


def test_local_protocol_records_frames_heartbeats_and_disconnects(monkeypatch, tmp_path):
    async def exercise():
        methods, echoed = [], []
        async def http(request):
            methods.append(request.method)
            assert request.headers["x-api-key"] == "synthetic-test-key"
            if "/categories/" in request.path:
                return web.json_response({"success": True, "data": category(request.match_info["slug"])})
            if request.path.endswith("orderbook"):
                return web.json_response({"success": True, "data": {"bids": [[0.4, 10]], "asks": [[0.6, 10]]}})
            return web.json_response({"success": True, "data": [], "cursor": None})
        async def websocket(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for msg in ws:
                data = json.loads(msg.data)
                if data["method"] == "heartbeat":
                    echoed.append(data["data"])
                    await ws.close()
                    break
                await ws.send_json({"type": "R", "requestId": data["requestId"], "success": True})
                await ws.send_json({"type": "M", "topic": data["params"][0], "data": {"marker": "raw-control"}})
                await ws.send_json({"type": "M", "topic": "heartbeat", "data": 123456789})
            return ws
        app = web.Application()
        app.router.add_get("/ws", websocket)
        app.router.add_get("/v1/categories/{slug}", http)
        app.router.add_get("/v1/markets/{mid}/orderbook", http)
        app.router.add_get("/v1/orders/matches", http)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setattr(pf, "REST", f"http://127.0.0.1:{port}")
        monkeypatch.setattr(pf, "WS", f"http://127.0.0.1:{port}/ws")
        monkeypatch.setenv("PREDICT_FUN_API_KEY", "synthetic-test-key")
        args = pf.arguments(["--duration-s", "0.5", "--no-binance"])
        writer = pf.RawWriter(tmp_path, {"api_key_env": args.api_key_env})
        recorder = pf.Recorder(args, writer)
        # Seed one mapped market to test the WS before REST throttling finishes.
        recorder.markets[42] = ("synthetic", 1e12)
        try:
            await recorder.run()
            await writer.close("TEST")
        finally:
            await runner.cleanup()
        rows = records(writer)
        assert methods and set(methods) == {"GET"}
        assert echoed == [123456789]
        frames = [r for r in rows if r["source"] == "predict_ws"]
        assert frames and any("raw-control" in r["raw"] for r in frames)
        assert all(r["recv_ns"] > 0 and r["recv_mono_ns"] > 0 for r in frames)
        assert any(r["kind"] == "WS_BOUNDARY" and r["status"].startswith("DISCONNECTED") for r in rows)
        assert "synthetic-test-key" not in json.dumps(rows)
        assert "synthetic-test-key" not in (writer.path / "manifest.json").read_text()
    asyncio.run(exercise())
