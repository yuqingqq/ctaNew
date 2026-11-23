"""Stage only this review and its own state blocks; preserve development edits."""
import hashlib
import json
import subprocess as sp
from pathlib import Path

ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).resolve().parent
PREFIX = str(HERE.relative_to(ROOT))
def git(*args, data=None):
    return sp.check_output(['git', '-C', str(ROOT), *args], input=data)
def digest(b):
    return hashlib.sha256(b).hexdigest()
assert not git('diff', '--cached', '--name-only').strip(), 'Other staged work: refuse'
head = git('rev-parse', 'HEAD').decode().strip()
blocks = json.loads((HERE / 'state_blocks.json').read_text())
blocks = {k: v.replace('\n+', '\n') for k, v in blocks.items()}
(HERE / 'state_blocks.json').write_text(json.dumps(blocks, indent=2)+'\n')
dirty = git('diff', '--name-only', '-z', 'HEAD').decode().split('\0')
preserve = {}
for name in dirty:
    if not name or name in blocks:
        continue
    p = ROOT / name
    preserve[name] = {'exists': p.exists(), 'sha256': digest(p.read_bytes()) if p.exists() else None}
logpath = ROOT / 'predictfun/research/run_light.log'
logbytes = logpath.read_bytes()
scope = {'parent': head, 'preserved_dirty': preserve, 'owned': {},
         'run_log_prefix_bytes': len(logbytes), 'run_log_prefix_sha256': digest(logbytes)}
def insert(name, b, block):
    marker = b'MATCHED_GAP_REVIEW_20261123_CODEX' if name.endswith('.md') else b'matched_gap_review_20261123_codex:'
    assert marker not in b
    if name.endswith('HANDOFF.md'):
        anchor = b'<!-- /STANDING_BATCH_RULE_20261120_USER -->\n'
        assert b.count(anchor) == 1
        return b.replace(anchor, anchor + block, 1)
    if name.endswith('STATUS.yml'):
        first, rest = b.split(b'\n', 1)
        return first + b'\n' + block + rest
    return block + b
for name, block in blocks.items():
    p = ROOT / name
    before = p.read_bytes()
    base = git('show', 'HEAD:' + name)
    work = insert(name, before, block.encode())
    staged = insert(name, base, block.encode())
    assert p.read_bytes() == before
    p.write_bytes(work)
    blob = git('hash-object', '-w', '--stdin', data=staged).decode().strip()
    git('update-index', '--add', '--cacheinfo', f'100644,{blob},{name}')
    scope['owned'][name] = {'before_work_sha256': digest(before), 'after_work_sha256': digest(work),
                           'expected_index_sha256': digest(staged)}
assert git('rev-parse', 'HEAD').decode().strip() == head
(HERE / 'commit_scope.json').write_text(json.dumps(scope, indent=2)+'\n')
files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name not in ('SHA256SUMS', 'stage.log', 'verify.log'))
(HERE / 'SHA256SUMS').write_text(''.join(f'{digest(p.read_bytes())}  {p.name}\n' for p in files))
git('add', '--', *[str(p.relative_to(ROOT)) for p in files], PREFIX + '/SHA256SUMS')
staged_names = git('diff', '--cached', '--name-only').decode().splitlines()
assert all(n in blocks or n.startswith(PREFIX + '/') for n in staged_names), staged_names
for name, r in scope['owned'].items():
    assert digest(git('show', ':' + name)) == r['expected_index_sha256']
print(json.dumps({'parent': head, 'files_staged': staged_names, 'other_dirty_paths_preserved': len(preserve)}, indent=2))
