"""Run upstream quantization tests while omitting the unrelated Rust extension."""
from pathlib import Path
import os
import sys
import types

import pytest
import torch

root = Path(os.environ.get("TORCHFT_PROBE_ROOT", r"C:\PR\torchft-scout-20261110"))
test_file = Path(os.environ.get("TORCHFT_TEST_FILE", str(root / "torchft/quantization_test.py")))
package = types.ModuleType("torchft")
package.__path__ = [str(root / "torchft")]
sys.modules["torchft"] = package
if "TORCHFT_TEST_SEED" in os.environ:
    torch.manual_seed(int(os.environ["TORCHFT_TEST_SEED"]))
raise SystemExit(pytest.main([str(test_file), *sys.argv[1:]]))
