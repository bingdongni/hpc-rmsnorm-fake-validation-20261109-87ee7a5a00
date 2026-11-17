from pathlib import Path
import os, sys, subprocess
root=Path(sys.argv[1]).resolve()
env={**os.environ,"OMP_NUM_THREADS":"1","PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","PYTHONPATH":str(root)}
subprocess.run([sys.executable,"-m","pytest","-q","-p","pytest_mock","-o","addopts=","tests/loss/test_vicreg_loss.py"],env=env,cwd=root,check=True)
