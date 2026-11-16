import os
from pathlib import Path
import subprocess
import sys
name=sys.argv[1]
targets={"aisuite":["tests/agents/test_state_store.py","tests/agents/test_continuation.py","tests/agents/test_async_runner.py"],
         "local":["tests/test_local_attention.py"],
         "sigma":["tests/test_validators_metadata.py","tests/test_validation.py","tests/test_collection.py","tests/test_validators.py"],
         "e3nn":["tests/o3/wigner_test.py","tests/o3/irreps_test.py","tests/o3/rotation_test.py"]}
subprocess.run([sys.executable,"-m","pytest","-q","--disable-warnings",*targets[name]],cwd=Path(name),
               env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1"},check=True)
