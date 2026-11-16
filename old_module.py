import importlib.util
from pathlib import Path
import sys
import torch
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("local_vicreg_loss", root / "lightly/loss/vicreg_loss.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.manual_seed(27)
for shape in [(16, 89), (4, 3, 11)]:
    x = torch.randn(shape, dtype=torch.double, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    e = m.VICRegLoss()(x, y)
    a = m.VICRegLoss(covariance_block_size=7)(x, y)
    torch.testing.assert_close(a, e)
    for i, j in zip(torch.autograd.grad(a, (x, y)), torch.autograd.grad(e, (x,y))):
        torch.testing.assert_close(i, j)
t = torch.randn(4, 5, dtype=torch.double, requires_grad=True)
assert torch.autograd.gradgradcheck(lambda v: m.covariance_loss(v, 2), (t,))
print("VICRegLoss source module passed on", torch.__version__, "loss, both gradients and gradgradcheck")

