"""Run normalization tests with the native schema and a CUDA reference implementation.

This checks the Python registration on devices that cannot load hpc-ops' native module;
it does not test the native CUDA kernel.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

root = Path(os.environ.get("HPC_VALIDATION_ROOT", r"C:\PR\tencent-hpc-ops-scout-20261109"))
normalization = Path(os.environ.get("HPC_NORMALIZATION_SOURCE", str(root / "hpc" / "normalization.py")))
(root / "build" / "lib.fake").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "tests"))

library = torch.library.Library("hpc", "DEF")
library.define(
    "fused_rmsnorm_with_scale(Tensor input, Tensor weight, Tensor scale, "
    "float eps, bool is_moe) -> (Tensor, Tensor, Tensor)"
)


def cuda_reference(x, weight, scale, eps, is_moe):
    normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    output = (normalized / scale[0]).to(torch.float8_e4m3fn)
    second = (normalized / scale[1]).to(torch.float8_e4m3fn) if is_moe else torch.empty_like(output)
    fp32 = normalized if is_moe else torch.empty_like(normalized)
    return output, fp32, second


library.impl("fused_rmsnorm_with_scale", cuda_reference, "CUDA")
package = types.ModuleType("hpc")
package.__path__ = [str(root / "hpc")]
sys.modules["hpc"] = package
spec = importlib.util.spec_from_file_location("hpc.normalization", normalization)
module = importlib.util.module_from_spec(spec)
sys.modules["hpc.normalization"] = module
spec.loader.exec_module(module)
package.normalization = module

tests = [
    "-q",
    "--tb=no",
    str(root / "tests" / "test_normalization.py") + "::test_fused_rmsnorm_with_scale_fake",
    str(root / "tests" / "test_normalization.py") + "::test_fused_rmsnorm_with_scale_compiled",
]
if os.environ.get("HPC_INCLUDE_NUMERIC") == "1":
    tests.append(str(root / "tests" / "test_normalization.py") + "::test_fused_rmsnorm_with_scale")
sys.exit(pytest.main(tests))
