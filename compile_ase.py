from pathlib import Path
import sys

from ase.io import read
import numpy as np

repo = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(repo))
from torchmdnet.calculators import TMDNETCalculator

checkpoint = str(repo / "tests/example_tensornet.ckpt")
atoms = read(repo / "tests/caffeine.pdb")
atoms.info["charge"] = 0
atoms.calc = TMDNETCalculator(checkpoint)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
atoms.calc = TMDNETCalculator(checkpoint, compile=True, eager_forces=False)
np.testing.assert_allclose(atoms.get_potential_energy(), energy, atol=1e-4, rtol=1e-5)
assert "forces" not in atoms.calc.results
np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-4, rtol=1e-5)
assert atoms.calc.compiled and atoms.calc.evals == 2
print("Compiled ASE energy-only and subsequent force request matched eager mode.")
