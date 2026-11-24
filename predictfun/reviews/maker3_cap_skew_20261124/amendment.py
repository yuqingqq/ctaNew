import json,subprocess,sys
import experiment as E
NAMES=['observe_v2.py','check_v2.py','selftest_v2.py','amendment.md','amendment.py']
def bound():
 m=json.loads((E.H/'amendment.json').read_text())
 for n,h in m.items():
  p=E.H/n;assert E.T.sha(p)==h
  assert subprocess.check_output(['git','show','HEAD:'+str(p.relative_to(E.R))],cwd=E.R)==p.read_bytes()
def prepare():
 assert not (E.H/'amendment.json').exists();E.T.dump(E.H/'amendment.json',{n:E.T.sha(E.H/n) for n in NAMES})
if __name__=='__main__':prepare()
