from pathlib import Path
import os
import subprocess
import sys

name=sys.argv[1]
tests={
    "nys":["tests"],
    "lhotse":["test/dataset/sampling/test_sampling.py","-k","weighted or partitions_are_equal or multiple_ranks"],
    "rupt":["tests/test_costs.py","tests/test_detection.py"]
}[name]
root=Path(name).resolve()
env={**os.environ,"PYTHONPATH":str(root/"src" if name=="rupt" else root)}
subprocess.run([sys.executable,"-m","pytest","-q","-o","addopts=",*tests],cwd=root,env=env,check=True)
