import sys,importlib.util
from pathlib import Path
import torch
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root))
p=Path(sys.argv[2]) if len(sys.argv)>2 else root/'lightly/loss/vicreg_loss.py'
spec=importlib.util.spec_from_file_location('vicreg_check',p)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def loss(x):return mod.covariance_loss(x,2)
x=torch.randn(6,7,requires_grad=True)
opt=torch.compile(loss,backend='aot_eager',fullgraph=True)
try:
    a=opt(x);g=torch.autograd.grad(a,x)[0]
    expected=loss(x);torch.testing.assert_close(a,expected);torch.testing.assert_close(g,torch.autograd.grad(expected,x)[0])
    print("Compiled forward and backward matched eager")
except Exception as exc:
    print(type(exc).__name__,str(exc)[:640]);raise SystemExit(1)

