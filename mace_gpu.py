"""Compare mixed-precision CUDA products with independent dense matrices."""
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from scipy.sparse.linalg import eigsh

checkout = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(checkout))
from mace.calculators import MACECalculator, vibrations
from tests.golden.anchors import load_anchor
from tests.golden.harness import FIXTURES_DIR

assert Path(vibrations.__file__).resolve().is_relative_to(checkout)
assert torch.cuda.is_available()
print("Device:", torch.cuda.get_device_name())
atoms = read(FIXTURES_DIR / "dimer_short.xyz")
field = [0.1, -0.2, 0.3]
vector = np.linspace(-0.4, 0.7, 6)
for dtypes in [(torch.float32, torch.float64), (torch.float64, torch.float32)]:
    models = [load_anchor(n, dtype=d) for n,d in zip(("tiny_mace","tiny_scaleshift"), dtypes)]
    dense = []
    for model in models:
        calc = MACECalculator(models=model, device="cuda", external_field=field)
        dense.append(calc.get_hessian(atoms).reshape(6,6).astype(np.float64))
    ref = np.mean(dense, axis=0)
    calc = MACECalculator(models=models, device="cuda", external_field=field)
    flags = []
    def watch(model, args, out):
        pos = args[0]["positions"]
        assert pos.dtype == next(model.parameters()).dtype
        torch.testing.assert_close(args[0]["external_field"], pos.new_tensor(field))
        def on_grad(g):
            flags.append((torch.is_inference(g), torch.is_inference_mode_enabled()))
        out["forces"].register_hook(on_grad)
    with models[1].register_forward_hook(watch), torch.inference_mode():
        op = vibrations.hessian_operator(calc, atoms)
        y = op @ vector
        ev = eigsh(op, k=1, which="SA", v0=vector, return_eigenvectors=False)[0]
    dy = np.max(np.abs(y-ref @ vector))
    de = np.abs(ev-np.linalg.eigvalsh(ref)[0])
    print("Mixed CUDA:", dtypes, "max product error", dy, "lowest eigenvalue error", de,
          "backward calls", len(flags), "inference flags", set(flags))
    np.testing.assert_allclose(y, ref @ vector, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(ev, np.linalg.eigvalsh(ref)[0], atol=3e-5, rtol=3e-5)
    assert flags and not any(a or b for a,b in flags)
