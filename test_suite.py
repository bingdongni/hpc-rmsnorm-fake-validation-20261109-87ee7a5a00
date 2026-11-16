import os, subprocess, sys
from pathlib import Path

k = sys.argv[1]
root = Path(k).resolve()
paths = {"wave":["tests/test_packets_3d.py","tests/test_packets.py::test_partial_reconstruction","tests/test_packets.py::test_partial_expansion_2d"], "vt":["tests/test_residual_fsq.py"], "torchode":["tests"]}
paths["wave"] = [p for p in paths["wave"] if (root / p.split("::")[0]).exists()]
env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD":"1", "OMP_NUM_THREADS":"1", "PYTHONPATH":str(root / "src" if k == "wave" else root)}
subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=",*paths[k]],env=env,cwd=root,check=True)
