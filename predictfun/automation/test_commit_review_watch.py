"""Watcher control-flow tests use temporary Git repos and a fake queue transport."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
import commit_review_watch as W


class WatcherTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.repo=self.root/'repo';self.repo.mkdir();self.state=self.root/'state/state.json'
        self.run_git('init','-b','mm-research')
        self.base=self.commit('README.md','base','initial')
        W.initialize(self.state,self.repo,'mm-research','test-thread','/bin/true',self.base)
        self.sent=[]

    def tearDown(self):self.tmp.cleanup()

    def run_git(self,*a):
        return subprocess.check_output(['git','-C',str(self.repo),'-c','user.name=Watcher Test',
            '-c','user.email=watcher-test@example.invalid',*a],text=True,stderr=subprocess.DEVNULL).strip()

    def commit(self,name,body,message):
        p=self.repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
        self.run_git('add',name);self.run_git('commit','-m',message)
        return self.run_git('rev-parse','HEAD')

    def send(self,c,message):
        self.sent.append(message)
        return subprocess.CompletedProcess([],0,'Queued message test-id.\n','')

    def report(self,target):
        p=self.repo/'predictfun/reviews/automatic/report.md';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('Reviewed '+target+'\nNext priority: repair the matched comparison.\n');return p

    def test_idle_and_ignore_review_and_state_only(self):
        self.assertEqual(W.check(self.state,self.send)['status'],'no_new_relevant_commits')
        self.commit('predictfun/HANDOFF.md','updated','state only')
        self.commit('predictfun/research/repro.py','check','review\n\n'+W.REVIEW_TRAILER)
        r=W.check(self.state,self.send)
        self.assertEqual(r['status'],'no_new_relevant_commits');self.assertEqual(self.sent,[])

    def test_idle_scan_refreshes_timestamp_without_moving_checkpoint(self):
        for at in ('2026-11-18T01:45:00+00:00', '2026-11-18T01:50:00+00:00'):
            with mock.patch.object(W, 'stamp', return_value=at):
                result=W.check(self.state,self.send)
            state=json.loads(self.state.read_text())
            self.assertEqual(result['status'],'no_new_relevant_commits')
            self.assertEqual(state['last_scan_at'],at)
            self.assertEqual(state['cursor'],self.base)
            self.assertIsNone(state['last_reviewed'])
            self.assertIsNone(state['pending'])
        self.assertEqual(self.sent,[])

    def test_dispatch_deduplicate_ack_then_new_commit(self):
        first=self.commit('predictfun/research/model.py','fixed','fix matching')
        r=W.check(self.state,self.send);self.assertEqual(r['status'],'queued')
        self.assertIn('last_scan_at',json.loads(self.state.read_text()))
        self.assertIn(first,self.sent[0]);self.assertIn('Do not implement fixes',self.sent[0])
        second=self.commit('predictfun/research/model.py','fixed again','another fix')
        self.assertEqual(W.check(self.state,self.send)['status'],'pending');self.assertEqual(len(self.sent),1)
        with self.assertRaises(ValueError):W.acknowledge(self.state,'wrong',self.report(first))
        with self.assertRaises(ValueError):W.acknowledge(self.state,r['request_id'],self.repo/'README.md')
        p=self.report('not the target')
        with self.assertRaises(ValueError):W.acknowledge(self.state,r['request_id'],p)
        W.acknowledge(self.state,r['request_id'],self.report(first))
        self.assertEqual(W.acknowledge(self.state,r['request_id'],self.report(first))['status'],'already_acknowledged')
        r2=W.check(self.state,self.send);self.assertEqual(r2['target'],second);self.assertEqual(len(self.sent),2)

    def test_transport_timeout_does_not_duplicate(self):
        self.commit('predictfun/research/model.py','fixed','fix')
        def timeout(c,m):raise subprocess.TimeoutExpired('queue',45)
        self.assertEqual(W.check(self.state,timeout)['status'],'uncertain')
        self.assertEqual(W.check(self.state,self.send)['status'],'pending');self.assertEqual(self.sent,[])

    def test_warning_is_not_confirmed_delivery(self):
        self.commit('predictfun/research/model.py','fixed','fix')
        def warning(c,m):return subprocess.CompletedProcess([],0,'Queued','resume warning')
        self.assertEqual(W.check(self.state,warning)['status'],'needs_attention')

    def test_branch_rewrite_refused(self):
        other=self.commit('other.txt','other','other work')
        W.check(self.state,self.send)
        self.run_git('reset','--hard',self.base)
        with self.assertRaises(ValueError):W.check(self.state,self.send)
        self.assertEqual(self.sent,[])


if __name__=='__main__':unittest.main()
