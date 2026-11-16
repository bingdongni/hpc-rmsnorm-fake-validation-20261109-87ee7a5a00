import json, sys
from pathlib import Path
import torch

key, tree, device, revision = sys.argv[1:]
root = Path(tree).resolve()
sys.path.insert(0, str(root/"src" if key=="wave" else root))
torch.manual_seed(19)
out = dict(project=key,revision=revision,device=torch.cuda.get_device_name() if device=="cuda" else device)
if key=="wave":
    import ptwt, pywt
    x = torch.randn(11,12,9,device=device,dtype=torch.float64,requires_grad=True)
    if revision=="base":
        assert not hasattr(ptwt, "WaveletPacket3D")
        assert ptwt.wavedec3(x,"db2",level=1)[0].ndim==3
    else:
        wp = ptwt.WaveletPacket3D(x,"db2",mode="reflect",maxlevel=2)
        ref = pywt.WaveletPacketND(x.detach().cpu().numpy(),"db2",mode="reflect",maxlevel=2)
        for node in ref.get_level(2):
            torch.testing.assert_close(wp[node.path].detach().cpu(),torch.from_numpy(node.data))
        y = wp.reconstruct()[""]
        torch.testing.assert_close(y,x)
        torch.testing.assert_close(torch.autograd.grad(y.sum(),x)[0],torch.ones_like(x))
        out.update(subbands=64,reconstruction_max_error=(y-x).abs().max().item())
elif key=="vt":
    from vector_quantize_pytorch import FSQ, ResidualFSQ
    opt = dict(materialize_codebook=False) if revision=="candidate" else {}
    quant = ResidualFSQ(levels=[8]*6,num_quantizers=3,**opt).to(device)
    x = torch.randn(2,8,6,device=device)
    y,ids = quant(x)
    actual = quant.get_output_from_indices(ids)
    torch.testing.assert_close(actual,y,atol=2e-6,rtol=1e-5)
    out["registered_buffer_bytes"]=sum(t.numel()*t.element_size() for t in quant.buffers())
    if revision=="candidate":
        assert out["registered_buffer_bytes"] < 1024
        obj=FSQ([16]*8,materialize_codebook=False).to(device)
        vals=torch.tensor([[0,2**24+1,2**31,2**32-1]],device=device)
        torch.testing.assert_close(obj.codes_to_indices(obj.indices_to_codes(vals)), vals)
        out["addressable_codes"]=obj.codebook_size
    else:
        assert out["registered_buffer_bytes"] >= 3*8**6*6*4
else:
    from torchode import solve_ivp, FixedStepController
    from torchode.interface import METHODS
    init=torch.tensor([[1.],[2.]],device=device,dtype=torch.float64,requires_grad=True)
    ts=init.new_tensor([[0.,.37,1.],[0.,-.43,-1.]])
    if revision=="base":
        assert "bosh3" not in METHODS
        method="heun"
    else: method="bosh3"
    sol=solve_ivp(lambda t,y:y,init,ts,method=method,controller=FixedStepController(),dt0=ts[:,-1]/32)
    err=(sol.ys-init[:,None]*ts.exp().unsqueeze(-1)).abs().max().item()
    assert err < (6e-4 if revision=="base" else 1e-5)
    out["max_abs_error"]=err
print("Compared",json.dumps(out))
