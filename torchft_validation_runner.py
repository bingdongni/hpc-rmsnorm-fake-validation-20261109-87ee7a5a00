import os
from pathlib import Path
import re
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
    ("candidate regression tests", candidate,
     ["-q", "-k", "large_index_metadata or large_rank_offsets or non_contiguous_inputs"], 0, "3 passed"),
):
    env = os.environ.copy()
    env["TORCHFT_PROBE_ROOT"] = str(root)
    env["TORCHFT_TEST_SEED"] = "0"
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

outcomes = []
for label, root in (("baseline", baseline), ("candidate", candidate)):
    env = os.environ.copy()
    env["TORCHFT_PROBE_ROOT"] = str(root)
    env["TORCHFT_TEST_SEED"] = "0"
    result = subprocess.run(
        [sys.executable, str(harness / "run_torchft_quant_tests.py"),
         "-q", "-k", "test_end_to_end", "--tb=line"],
        cwd=root, env=env, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=600,
    )
    output = result.stdout + result.stderr
    failures = re.findall(r"^FAILED .*::(test_end_to_end_\d+)", output, re.MULTILINE)
    errors = re.findall(r"^E\s+AssertionError: (.*)$", output, re.MULTILINE)
    summary = [line for line in output.splitlines() if "passed" in line and " in " in line]
    print(label, "existing end-to-end suite exit", result.returncode,
          "summary", summary[-1:] if summary else "missing", "failures", failures, "errors", errors, flush=True)
    if result.returncode not in (0, 1) or not summary:
        print(output[-6000:], flush=True)
        raise SystemExit(f"{label} existing test suite could not run")
    outcomes.append((result.returncode, failures, errors))
assert outcomes[0] == outcomes[1], "candidate differs from baseline in existing end-to-end tests"
print("Seed-0 end-to-end failures and error values match baseline and candidate", flush=True)

seed_results = []
for root in (baseline, candidate):
    env = os.environ.copy()
    env["TORCHFT_PROBE_ROOT"] = str(root)
    env["TORCHFT_SEED_COUNT"] = "4"
    result = subprocess.run(
        [sys.executable, str(harness / "torchft_quant_seed_probe.py")],
        cwd=root, env=env, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=180,
    )
    if result.returncode != 0:
        print(result.stderr[-6000:], flush=True)
        raise SystemExit("seeded baseline comparison failed")
    seed_results.append(result.stdout)
assert seed_results[0] == seed_results[1] and "seed 3 FAIL" in seed_results[0]
print("Seeded pre-existing tolerance failure matches baseline and candidate:",
      seed_results[0].splitlines()[-1], flush=True)

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
