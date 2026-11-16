import sys
from pathlib import Path
import subprocess
import os
root=Path("tomo/packages/primitives/torch-fourier-filter").resolve()
workspace=Path("tomo/packages").resolve()
paths=[str(d) for d in workspace.glob("*/*/src") if d.is_dir()]
env={**os.environ,"PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1","OMP_NUM_THREADS":"1","PYTHONPATH":os.pathsep.join(paths)}
subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=","tests"],cwd=root,env=env,check=True)
