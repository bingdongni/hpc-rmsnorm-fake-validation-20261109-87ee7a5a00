import importlib.util
from pathlib import Path
import sys
import tempfile
import types


checkout, tests, expectation = sys.argv[1:4]
sys.path.insert(0, str(Path(checkout).resolve()))
if sys.platform == "win32":
    # ColossalAI imports POSIX resource to report CPU memory, unused on this CUDA path.
    sys.modules["resource"] = types.ModuleType("resource")
    sys.path.append("D:/Python/Lib/site-packages")

import torch
from colossalai.checkpoint_io import GeneralCheckpointIO


assert Path(sys.modules[GeneralCheckpointIO.__module__].__file__).resolve().is_relative_to(Path(checkout).resolve())
print("GPU", torch.cuda.get_device_name(), torch.__version__, flush=True)
spec = importlib.util.spec_from_file_location("test_resume", Path(tests) / "tests/test_checkpoint_io/test_adamw_step_resume.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for mode in ("capturable", "fused"):
    with tempfile.TemporaryDirectory() as folder:
        try:
            module.test_sharded_adamw_step_resume(Path(folder), mode)
        except AssertionError as err:
            if expectation != "baseline" or "device" not in str(err) or "cpu" not in str(err):
                raise
            print("Baseline", mode, str(err), flush=True)
        else:
            assert expectation == "candidate", f"upstream passed: {mode}"
            print("Verified", mode, "and default group plus inactive parameter; next AdamW update and moments match", flush=True)
