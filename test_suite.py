import os
from pathlib import Path
import subprocess
import sys

name = sys.argv[1]
cases = {
    "pot": ["test/test_bregman.py", "-k", "empirical_sinkhorn or lazy_euclidean_cost"],
    "diffeq": ["tests/api_tests.py", "-k", "dense_output"],
    "tslearn": ["tests/test_metrics_soft_dtw_loss_pytorch.py"],
}
root = Path(name).resolve()
env = dict(os.environ, PYTHONPATH=str(root))
subprocess.run([sys.executable, "-m", "pytest", "-q", "--disable-warnings", *cases[name]], cwd=root, env=env, check=True)
