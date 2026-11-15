import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
import time

from ase.build import molecule
import numpy as np
import torch


parser = argparse.ArgumentParser()
for name in ("candidate", "baseline", "weights"):
    parser.add_argument(name, type=Path)
args = parser.parse_args()
with args.weights.open("rb") as fp:
    assert hashlib.file_digest(fp, "sha256").hexdigest() == "877af6bf84cc69d6d5cbd377a61cfc19c45fb87a00583f7c867e2f9d244cc8fd"
sys.path.insert(0, str(args.candidate.resolve()))
from torchmdnet.calculators import TMDNETCalculator

spec = importlib.util.spec_from_file_location("torchmdnet._baseline_calculator", args.baseline.resolve())
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
assert torch.cuda.is_available()
checkpoint = str(args.weights.resolve())
calculators = [base.TMDNETCalculator(checkpoint, device="cuda"), TMDNETCalculator(checkpoint, device="cuda", eager_forces=False)]
print(json.dumps({"gpu": torch.cuda.get_device_name(), "model": "Acellera/AceFF-2.0", "torch": torch.__version__}), flush=True)

initial = molecule("trans-butane")
initial.info["charge"] = 0
geometries = []
for angle in range(0, 360, 15):
    atoms = initial.copy()
    # Rotate the second ethyl group, including the H atoms attached to its central carbon.
    atoms.set_dihedral(0, 1, 2, 3, angle, indices=[2, 3, 5, 8, 9, 12, 13])
    np.testing.assert_allclose((atoms.get_dihedral(0, 1, 2, 3) - angle + 180) % 360 - 180, 0., atol=1e-8)
    geometries.append(atoms)


def scan(calc):
    result = []
    for atoms in geometries:
        atoms.calc = calc
        result.append(atoms.get_potential_energy())
    return np.array(result)


for calc in calculators:
    scan(calc)
times, peaks = [[], []], [[], []]
profiles = [None, None]
for run in range(6):
    for i in ((0, 1) if run % 2 == 0 else (1, 0)):
        calc = calculators[i]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        start = time.perf_counter()
        profiles[i] = scan(calc)
        torch.cuda.synchronize()
        times[i].append((time.perf_counter() - start) * 1000)
        peaks[i].append((torch.cuda.max_memory_allocated() - allocated) / 2**20)
np.testing.assert_allclose(profiles[0], profiles[1], atol=2e-5, rtol=0.)
assert np.ptp(profiles[0]) > .05
assert "forces" not in calculators[1].results
forces = []
for calc in calculators:
    atoms = geometries[np.argmin(profiles[0])]
    atoms.calc = calc
    atoms.get_potential_energy()
    forces.append(atoms.get_forces())
np.testing.assert_allclose(forces[0], forces[1], atol=2e-5, rtol=0.)
print(json.dumps(dict(aceff="trans-butane rigid torsion scan", geometries=len(geometries),
                      baseline_ms=round(median(times[0]), 2), candidate_ms=round(median(times[1]), 2),
                      baseline_MiB=round(median(peaks[0]), 3), candidate_MiB=round(median(peaks[1]), 3),
                      energy_span_eV=float(np.ptp(profiles[0])), max_abs_energy_eV=float(np.max(np.abs(profiles[0] - profiles[1]))),
                      max_abs_force_eV_A=float(np.max(np.abs(forces[0] - forces[1]))))), flush=True)
