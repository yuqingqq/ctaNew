"""Run one small result summary at the active worker boundary, then resume."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from model_wait_post_driver import busy,resume_size,write,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume-record',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir();env=dict(os.environ,PF_SLOT='codex',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def status(state,**kw):write(a.out/'runner.json',dict(status=state,asof=time.time(),**kw))
    try:
        status('waiting_for_current_worker')
        while busy():time.sleep(3)
        manifest=json.loads(a.manifest.read_text())
        for name,h in manifest['sources'].items():
            if digest(HERE/name)!=h:raise ValueError('summary source changed')
        for name,h in manifest['inputs'].items():
            if digest(name)!=h:raise ValueError('replay input changed')
        status('running')
        with (a.out/'run.log').open('x') as f:
            rc=subprocess.call([str(HERE/'run_light.sh'),'entry_repair_sizes','--',sys.executable,str(HERE/'pf_entry_repair_sizes.py'),
                '--base',str(a.base),'--out',str(a.out)],env=env,stdout=f,stderr=subprocess.STDOUT)
        if rc:raise RuntimeError('size summary failed; attempt retained')
        status('complete')
    except BaseException as e:status('failed',error=str(e));raise
    finally:resume_size(a.resume_record)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    main()
