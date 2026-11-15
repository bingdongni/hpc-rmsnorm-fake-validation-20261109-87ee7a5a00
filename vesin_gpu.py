import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import vesin_torch

checkout = Path(sys.argv[1]).resolve()
source = checkout / "python/vesin_torch"
spec = importlib.util.spec_from_file_location("vesin_torch._candidate_neighbors", source / "vesin_torch/_neighbors.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
vesin_torch.NeighborList = module.NeighborList
assert torch.cuda.is_available()
print("GPU", torch.cuda.get_device_name(), "torch", torch.__version__, flush=True)
if "--baseline" in sys.argv:
    positions = torch.tensor([[0.1, 0., 0.], [2.8, 0., 0.]], dtype=torch.float64, requires_grad=True)
    try:
        vesin_torch.NeighborList(0.8, full_list=True, padded_to=3).compute(positions, 3 * torch.eye(3, dtype=positions.dtype), True, "Dd")
    except (IndexError, RuntimeError) as error:
        assert "index" in str(error).lower(), str(error)
        print("Upstream padded CPU force evaluation:", type(error).__name__, str(error))
    else:
        raise AssertionError("expected the base checkout to fail")
else:
    raise SystemExit(pytest.main(["-q", str(source / "tests/test_neighbors.py"), "-k", "padded_neighbors_autograd or pad_neighbors or padded_outputs"]))
