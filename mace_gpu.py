"""Compare MACE's matrix-free products with its dense analytical Hessian on CUDA."""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from scipy.sparse.linalg import eigsh

checkout = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(checkout))
from mace.calculators import MACECalculator
from mace.calculators import vibrations
from tests.golden.anchors import load_anchor
from tests.golden.harness import FIXTURES_DIR

assert Path(vibrations.__file__).resolve().is_relative_to(checkout)
torch.set_num_threads(2)
calc = MACECalculator(models=load_anchor("tiny_scaleshift"), device="cuda", default_dtype="float64")


def measure(fn):
    times, peaks = [], []
    for _ in range(3):
        gc.collect()
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
        peaks.append((torch.cuda.max_memory_allocated() - used) / 2**20)
    return value, round(statistics.median(times), 2), round(max(peaks), 2)


print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
small = read(FIXTURES_DIR / "triclinic_bulk.xyz")
calc.get_hessian(small)
vibrations.hessian_operator(calc, small) @ np.ones(3 * len(small))
for repeat in (1, 2):
    atoms = small.repeat((repeat,) * 3)
    size = 3 * len(atoms)
    q = np.sin(np.arange(size) * 0.31) + 0.25

    def dense():
        return calc.get_hessian(atoms).reshape(size, size)

    def product():
        return vibrations.hessian_operator(calc, atoms) @ q

    h, dense_ms, dense_mib = measure(dense)
    out, cold_ms, cold_mib = measure(product)
    expected = h @ q
    np.testing.assert_allclose(out, expected, atol=1e-8, rtol=1e-8)
    operator = vibrations.hessian_operator(calc, atoms)
    out, hot_ms, hot_mib = measure(lambda: operator @ q)
    np.testing.assert_allclose(out, expected, atol=1e-8, rtol=1e-8)
    eigen_error = None
    if repeat == 1:
        eig = eigsh(operator, k=1, which="SA", v0=q, return_eigenvectors=False)[0]
        eigen_error = float(abs(eig - np.linalg.eigvalsh(h)[0]))
        assert eigen_error < 1e-8
    print(json.dumps({"mace_atoms": len(atoms), "dense_ms": dense_ms, "build_and_product_ms": cold_ms,
                      "repeat_product_ms": hot_ms, "dense_peak_mib": dense_mib, "build_product_peak_mib": cold_mib,
                      "repeat_additional_mib": hot_mib, "max_product_error": float(np.abs(out - expected).max()),
                      "smallest_eigenvalue_error": eigen_error}), flush=True)
    del operator
