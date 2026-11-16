import os
from pathlib import Path
import subprocess
import sys

key = sys.argv[1]
repo = Path(key).resolve()
tests = ["tests/test_ssim_map.py"] if key == "ms" else ["tests"]
if key == "mppi" and os.name == "nt":
    # The three CPU Inductor cases need MSVC. The Linux job runs all three.
    tests.extend(["-k", "not test_compile"])
env = dict(os.environ, PYTHONPATH=str(repo if key == "ms" else repo / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
subprocess.run([sys.executable, "-m", "pytest", "-q", "-o", "addopts=", *tests], cwd=repo, env=env, check=True)
