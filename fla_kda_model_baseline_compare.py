import os
from pathlib import Path
import re
import subprocess
import sys

node = 'tests/models/test_modeling_kda.py::test_modeling[L4-B4-T1024-H4-D64-l2True-bsNone-torch.bfloat16-varlen]'
results = []
for label, variable in (('baseline', 'FLA_PROBE_BASE'), ('candidate', 'FLA_PROBE_CANDIDATE')):
    root = Path(os.environ[variable])
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.environ['FLA_PROBE_EXTRA_PATH'] + os.pathsep + str(root)
    environment['FLA_CI_ENV'] = '0'
    environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    environment.pop('FLA_DISABLE_BACKEND_DISPATCH', None)
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', node, '-q', '-o', 'log_cli=false', '--disable-warnings', '--tb=short'],
        cwd=root, env=environment, capture_output=True, text=True, timeout=300,
    )
    output = result.stdout + result.stderr
    match = re.search(r'output diff:\s*([0-9.]+) ratio:\s*([0-9.]+)', output)
    if result.returncode != 1 or not match:
        print(label, 'unexpected test outcome', 'returncode', result.returncode, flush=True)
        print(output[-5000:], flush=True)
        raise SystemExit(1)
    values = tuple(float(item) for item in match.groups())
    print(label, 'expected training test failure:', 'output_diff', values[0], 'ratio', values[1], flush=True)
    results.append(values)
if results[0] != results[1]:
    raise SystemExit('Baseline and candidate numerical failures differed')
print('Baseline and candidate numerical failures match.', flush=True)
