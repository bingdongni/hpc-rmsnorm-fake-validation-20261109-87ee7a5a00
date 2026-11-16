import os
from pathlib import Path
import subprocess
import sys

name = sys.argv[1]
root = Path(name).resolve()
env = {**os.environ, "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "SCIPY_ARRAY_API":"1"}
def run(args, **extra):
    subprocess.run([sys.executable,"-m","pytest","-q","--disable-warnings",*args],cwd=root,env={**env,**extra},check=True)
if name == "geom":
    for backend in ("numpy", "autograd", "pytorch"):
        run(["tests/tests_geomstats/test_learning/test_geometric_median.py"],GEOMSTATS_BACKEND=backend)
elif name == "pyro":
    run(["tests/distributions/test_hmm.py","-k","discrete_hmm"])
elif name == "rotary":
    run(["tests/test_rotary.py","-k","not flash"])
