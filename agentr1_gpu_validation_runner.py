import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, flush=True)
baseline = Path(os.environ["AGENTR1_BASE"])
candidate = Path(os.environ["AGENTR1_CANDIDATE"])
test = Path("tests/test_agent_flow_rewards.py")
(baseline / "tests").mkdir(exist_ok=True)
shutil.copyfile(candidate / test, baseline / test)
for label, root, expected_exit, expected_text in (
    ("baseline", baseline, 1, "8 failed, 4 passed"),
    ("candidate", candidate, 0, "12 passed"),
):
    env = os.environ.copy()
    env["AGENTR1_PROBE_ROOT"] = str(root)
    result = subprocess.run([sys.executable, os.environ["AGENTR1_HARNESS"]], cwd=root, env=env,
                            text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=180)
    output = result.stdout + result.stderr
    summary = [line for line in output.splitlines() if "passed" in line and ("failed" in line or " in " in line)]
    print(label, "exit", result.returncode, "summary", summary[-1:] if summary else "missing", flush=True)
    if result.returncode != expected_exit or expected_text not in output:
        print(output[-8000:], flush=True)
        raise SystemExit(f"{label}: unexpected test result")
