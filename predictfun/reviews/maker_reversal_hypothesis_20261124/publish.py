"""Stage this packet and additive owned state, preserving other-session work."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).resolve().parent

def git(*args, data=None):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], input=data)

def sha(data):
    return hashlib.sha256(data).hexdigest()

assert not git('diff', '--cached', '--name-only').strip(), 'staged work exists'
blocks = json.loads((HERE / 'state_blocks.json').read_text())
scope = {'parent': git('rev-parse', 'HEAD').decode().strip(), 'preserved_dirty': {}, 'owned_state': {}}
for name in git('diff', '--name-only', '-z', 'HEAD').decode().split('\0'):
    if not name or name in blocks:
        continue
    path = ROOT / name
    scope['preserved_dirty'][name] = sha(path.read_bytes()) if path.exists() else None
for name, block in blocks.items():
    path = ROOT / name
    work = path.read_bytes()
    base = git('show', 'HEAD:' + name)
    own = block.encode()
    assert b'REVERSAL_HYPOTHESIS_20261124_CODEX' not in work
    assert b'reversal_hypothesis_20261124_codex:' not in work

    def add(old):
        if name.endswith('HANDOFF.md'):
            anchor = b'<!-- /STANDING_BATCH_RULE_20261120_USER -->\n'
            assert old.count(anchor) == 1
            return old.replace(anchor, anchor + own, 1)
        if name.endswith('STATUS.yml'):
            first, rest = old.split(b'\n', 1)
            return first + b'\n' + own + rest
        return own + old

    path.write_bytes(add(work))
    staged = add(base)
    blob = git('hash-object', '-w', '--stdin', data=staged).decode().strip()
    git('update-index', '--add', '--cacheinfo', '100644,' + blob + ',' + name)
    scope['owned_state'][name] = {'preexisting_work_sha': sha(work), 'staged_sha': sha(staged), 'post_work_sha': sha(path.read_bytes())}
    assert path.read_bytes().replace(own, b'', 1) == work
    assert staged.replace(own, b'', 1) == base
for name, expected in scope['preserved_dirty'].items():
    path = ROOT / name
    assert (sha(path.read_bytes()) if path.exists() else None) == expected, name
assert git('rev-parse', 'HEAD').decode().strip() == scope['parent'], 'HEAD advanced'
(HERE / 'commit_scope.json').write_text(json.dumps(scope, indent=2) + '\n')
files = []
for path in sorted(HERE.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.name != 'SHA256SUMS':
        assert path.stat().st_size < 1024 * 1024, path
        files.append(path)
(HERE / 'SHA256SUMS').write_text(''.join(sha(path.read_bytes()) + '  ' + str(path.relative_to(HERE)) + '\n' for path in files))
files.append(HERE / 'SHA256SUMS')
git('add', '--', *[str(path.relative_to(ROOT)) for path in files])
print(git('diff', '--cached', '--stat').decode())
