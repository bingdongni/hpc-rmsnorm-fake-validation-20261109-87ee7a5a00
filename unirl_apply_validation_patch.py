import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(os.environ['UNIRL_PROBE_ROOT'])
manifest = json.loads((Path(__file__).parent / 'unirl_patch_manifest.json').read_text(encoding='utf-8'))
head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
assert head == manifest['base']
patch = base64.b64decode(os.environ.pop('UNIRL_PATCH_B64'), validate=True)
assert hashlib.sha256(patch).hexdigest() == manifest['patch_sha256']
subprocess.run(['git', '-C', str(root), 'config', 'core.autocrlf', 'false'], check=True)
for name in manifest['files']:
    path = root / name
    data = path.read_bytes()
    if b'\r\n' in data:
        path.write_bytes(data.replace(b'\r\n', b'\n'))

handle, temporary = tempfile.mkstemp(prefix='unirl-validation-', suffix='.patch', dir=os.environ['RUNNER_TEMP'])
try:
    with os.fdopen(handle, 'wb') as out:
        out.write(patch)
    for args in (['apply', '--check'], ['apply']):
        result = subprocess.run(['git', '-C', str(root), *args, temporary], capture_output=True)
        if result.returncode:
            raise RuntimeError('Patch application failed')
finally:
    Path(temporary).unlink(missing_ok=True)

for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print('Verified base revision and all candidate file hashes')
