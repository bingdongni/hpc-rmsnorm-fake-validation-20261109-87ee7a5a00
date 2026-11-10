import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

baseline = Path(os.environ["TORCHFT_BASE"])
candidate = Path(os.environ["TORCHFT_CANDIDATE"])
harness = Path(os.environ["TORCHFT_HARNESS"])
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, flush=True)
shutil.copyfile(candidate / "torchft/quantization_test.py", baseline / "torchft/quantization_test.py")
for label, root, args, expected_code, expected_text in (
    ("baseline regression", baseline, ["-q", "-k", "test_large_index_metadata"], 1,
     "value cannot be converted to type int32 without overflow"),
    ("candidate quantization suite", candidate, ["-q"], 0, "57 passed"),
):
    env = os.environ.copy()
    env["TORCHFT_PROBE_ROOT"] = str(root)
    result = subprocess.run(
        [sys.executable, str(harness / "run_torchft_quant_tests.py"), *args],
        cwd=root, env=env, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=600,
    )
    output = result.stdout + result.stderr
    summary = [line for line in output.splitlines() if "passed" in line and " in " in line]
    print(label, "exit", result.returncode, "summary", summary[-1:] if summary else "missing", flush=True)
    if result.returncode != expected_code or expected_text not in output:
        print(output[-6000:], flush=True)
        raise SystemExit(f"{label} failed")

env = os.environ.copy()
env["TORCHFT_PROBE_ROOT"] = str(candidate)
result = subprocess.run(
    [sys.executable, str(harness / "torchft_quant_large_address_probe.py")],
    cwd=candidate, env=env, text=True, encoding="utf-8", errors="replace",
    capture_output=True, timeout=240,
)
print(result.stdout, flush=True)
if result.returncode != 0 or "input stride above 2^31 elements" not in result.stdout:
    print(result.stderr[-6000:], flush=True)
    raise SystemExit("large-address GPU probe failed")
