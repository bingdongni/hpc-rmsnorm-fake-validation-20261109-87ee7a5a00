import os, sys, subprocess
from pathlib import Path
k=sys.argv[1]
r=Path(k).resolve()
env={**os.environ,"PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","OMP_NUM_THREADS":"1","PYTHONPATH":str(r/"src" if k=="psd" else r)}
tests={"conv":"test/test_fold.py","rot":"tests/test_rotary.py","psd":"tests/test_whitening.py"}
subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=",tests[k]],cwd=r,env=env,check=True)
