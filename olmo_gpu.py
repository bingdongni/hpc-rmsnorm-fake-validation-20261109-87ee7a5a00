import argparse
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types

parser = argparse.ArgumentParser()
parser.add_argument("checkout", type=Path)
parser.add_argument("--legacy", type=Path)
parser.add_argument("--baseline", action="store_true")
args = parser.parse_args()
if args.legacy:
    sys.path.insert(0, str(args.legacy.resolve()))
src = args.checkout.resolve() / "src"
sys.path.insert(0, str(src))
if sys.platform == "win32":
    # The top-level nn importer also imports the POSIX-only data loader. Load
    # the unmodified LM head and its real dependencies without importing that loader.
    import olmo_core
    nn = types.ModuleType("olmo_core.nn")
    nn.__path__ = [str(src / "olmo_core/nn")]
    nn.__spec__ = importlib.machinery.ModuleSpec(nn.__name__, loader=None, is_package=True)
    sys.modules[nn.__name__] = nn
    kernels = Path("C:/PR/fla-research-deps-20261110/triton")
    os.environ["CC"] = str(kernels / "runtime/tcc/tcc.exe")
    os.environ["CUDA_PATH"] = str(kernels / "backends/nvidia")

import torch
from olmo_core.nn.lm_head import LMHead, LMLossImplementation
from olmo_core.nn.functional import fused_linear_cross_entropy_loss

assert Path(sys.modules[LMHead.__module__].__file__).resolve().is_relative_to(src)
assert Path(sys.modules[fused_linear_cross_entropy_loss.__module__].__file__).resolve().is_relative_to(src)
assert torch.cuda.is_available()
version = importlib.metadata.version("liger-kernel")
print("GPU", torch.cuda.get_device_name(), torch.__version__, "liger", version, flush=True)
torch.manual_seed(42)
errors = []
for dtype in [torch.float32, torch.bfloat16]:
    eager = LMHead(d_model=128, vocab_size=1024, bias=False, dtype=dtype, init_device="cuda")
    fused = LMHead(d_model=128, vocab_size=1024, bias=False, dtype=dtype, init_device="cuda",
                   loss_implementation=LMLossImplementation.fused_linear)
    fused.load_state_dict(eager.state_dict())
    for z in [None, 1e-4]:
        for reduction in ["mean", "sum"]:
            x = torch.randn(2, 37, 128, device="cuda", dtype=dtype, requires_grad=True)
            y = x.detach().clone().requires_grad_()
            labels = torch.randint(0, 1024, (2, 37), device="cuda")
            labels[:, -3:] = -100
            eager.zero_grad(set_to_none=True)
            fused.zero_grad(set_to_none=True)
            ref = eager(x, labels=labels, loss_reduction=reduction, z_loss_multiplier=z)
            try:
                got = fused(y, labels=labels, loss_reduction=reduction, z_loss_multiplier=z)
            except ValueError as error:
                if args.baseline and "too many values to unpack" in str(error):
                    print("Upstream fused LMHead:", str(error))
                    sys.exit(0)
                raise
            assert not args.baseline and got.logits is None
            ref.loss.backward()
            got.loss.backward()
            atol = 2e-6 if dtype == torch.float32 else 1e-2
            rtol = 2e-4 if dtype == torch.float32 else 1e-2
            for actual, expected in [(got.loss, ref.loss), (got.ce_loss, ref.ce_loss),
                                     (y.grad, x.grad), (fused.w_out.weight.grad, eager.w_out.weight.grad)]:
                torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
            if z:
                torch.testing.assert_close(got.z_loss.float(), ref.z_loss.float(), rtol=rtol, atol=atol)
            errors.append({"dtype": str(dtype), "z": z, "reduction": reduction, "loss": got.loss.item()})
print(json.dumps({"liger": version, "cases": errors}))
