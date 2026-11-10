"""Check an existing random quantization tolerance for reproducibility."""
import importlib
import os
from pathlib import Path
import sys
import types

import torch

root = Path(os.environ["TORCHFT_PROBE_ROOT"])
package = types.ModuleType("torchft")
package.__path__ = [str(root / "torchft")]
sys.modules["torchft"] = package
test_module = importlib.import_module("torchft.quantization_test")
test = test_module.QuantizationTest()
tensor_size, multiplier, reduce_op, dtype = test.END_TO_END_CONFIGS[21]
print("case", tensor_size, multiplier, reduce_op, dtype, flush=True)
for seed in range(int(os.environ.get("TORCHFT_SEED_COUNT", "20"))):
    torch.manual_seed(seed)
    try:
        test.run_test(world_size=2, tensors_num=3, tensor_size=tensor_size,
                      multiplier=multiplier, tolerance=0.05, reduce_op=reduce_op, type=dtype)
    except AssertionError as exc:
        print("seed", seed, "FAIL", str(exc).splitlines()[0], flush=True)
        if os.environ.get("TORCHFT_STOP_ON_FAIL", "1") == "1":
            break
    else:
        print("seed", seed, "PASS", flush=True)
