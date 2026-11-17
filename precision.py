import importlib.util
import sys
from pathlib import Path

import torch

root=Path(sys.argv[1]).resolve()
dev, revision=sys.argv[2:]
sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location("local_vicreg",root/"lightly/loss/vicreg_loss.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(31)
if dev=='cuda':print("Device",torch.cuda.get_device_name(),torch.__version__,flush=True)
for dtype,width in ((torch.float16,4096),(torch.bfloat16,512)):
    row=torch.full((width,),0.5/width**0.25,dtype=dtype,device=dev)
    z=torch.stack([row,-row])
    try:reference=m.covariance_loss(z)
    except RuntimeError as exc:
        if dev=="cpu" and "not implemented for" in str(exc):
            print("Skipped unsupported dense CPU reference",torch.__version__,dtype,flush=True);continue
        raise
    actual=m.covariance_loss(z,1)
    print("Compared",torch.__version__,revision,dev,dtype,width,float(reference),float(actual),flush=True)
    if revision=='candidate':torch.testing.assert_close(actual,reference)
    else:assert abs(float(actual)/float(reference)-0.5)<0.02

if hasattr(torch,'func'):
    z=torch.randn(2,5,3,7,dtype=torch.double,device=dev)
    def blocked(v):return m.covariance_loss(v,2)
    dense=m.covariance_loss
    if revision=='base':
        try:torch.func.grad(blocked)(z[0])
        except RuntimeError as exc:assert 'setup_context' in str(exc)
        else:raise AssertionError("expected the original transform error")
    else:
        for transform in (torch.func.grad,torch.func.jacrev,torch.func.jacfwd):
            torch.testing.assert_close(transform(blocked)(z[0]),transform(dense)(z[0]))
        torch.testing.assert_close(torch.vmap(blocked)(z),torch.vmap(dense)(z))
        torch.testing.assert_close(torch.vmap(torch.func.grad(blocked))(z),torch.vmap(torch.func.grad(dense))(z))
        dz=torch.randn_like(z[0]);torch.testing.assert_close(torch.func.jvp(blocked,(z[0],),(dz,)),torch.func.jvp(dense,(z[0],),(dz,)))
        print("Compared",torch.__version__,"grad/vmap/forward and reverse Jacobians and JVP to dense",flush=True)

