import json
import math
import os
from pathlib import Path
import sys
import torch

project, tree, device, revision = sys.argv[1:]
root = Path(tree).resolve()
if os.name == "nt":
    root = Path("//?/" + root.as_posix())
sys.path[:0] = [str(d) for d in (root/"packages").glob("*/*/src") if d.is_dir()]
torch.manual_seed(24)
torch.set_num_threads(1)
if device == "cuda":
    assert torch.cuda.is_available() and "4060" in torch.cuda.get_device_name()
    print("Device", torch.cuda.get_device_name(), torch.__version__)
candidate = revision == "candidate"

def fold():
    from unfoldNd import FoldNd
    params = dict(output_size=(5000, 5000), kernel_size=(2, 2), stride=1000)
    v = torch.randn(1, 4, 25, device=device, requires_grad=True)
    try:
        out = FoldNd(**params)(v)
    except RuntimeError as err:
        assert not candidate and "Output size elements" in str(err)
        return {"large_image_supported": False}
    assert candidate
    assert torch.equal(out, torch.nn.Fold(**params)(v))
    grad = torch.autograd.grad(out.square().sum(), v)[0]
    torch.testing.assert_close(grad, 2*v)
    # Independently scatter overlapping blocks into a small 3D volume.
    from itertools import product
    a = torch.randn(2, 3*8, 4*3*5, device=device)
    actual = FoldNd((5, 4, 6), (2, 2, 2))(a)
    ref = torch.zeros_like(actual)
    for i, (x,y,z) in enumerate(product(range(2),repeat=3)):
        ref[:,:,x:x+4,y:y+3,z:z+5] += a.reshape(2,3,8,4,3,5)[:,:,i]
    torch.testing.assert_close(ref, actual)
    return {"elements": out.numel(), "reference_2d_exact":True,"gradient_max_error":(grad-2*v).abs().max().item(),"overlap_3d_error":(ref-actual).abs().max().item()}

def rotate():
    from rotary_embedding_torch import RotaryEmbedding
    res={}
    m=RotaryEmbedding(16, interpolate_factor=2.).to(device)
    ids=torch.tensor([[0,1,2,0,1],[0,0,0,1,2],[51,52,0,1,2]],device=device)
    for dtype in (torch.float32,torch.bfloat16,torch.float16):
        t=torch.randn(3,2,5,24,device=device,dtype=dtype,requires_grad=True)
        try: a=m.rotate_queries_or_keys(t, position_ids=ids)
        except TypeError as err:
            assert not candidate and "position_ids" in str(err)
            return {"per_example_ids":False}
        assert candidate
        paired=t[...,:16].float().reshape(3,2,5,8,2)
        phases=(ids.float()[:,None,:,None] / 2)* (10000.**(-torch.arange(0,16,2,device=device)/16))
        c,s=phases.cos(),phases.sin()
        real,imag=paired.unbind(-1)
        expected=torch.cat([torch.stack([real*c-imag*s,imag*c+real*s],dim=-1).flatten(-2).to(dtype),t[...,16:]],dim=-1)
        g=torch.autograd.grad(a.sum(),t)[0]
        gr=torch.autograd.grad(expected.sum(),t)[0]
        torch.testing.assert_close(a,expected)
        torch.testing.assert_close(g,gr,rtol=.009,atol=.009)
        res[str(dtype)]={"value_error":(a-expected).abs().max().item(),"grad_error":(g-gr).abs().max().item()}
    return res

def spectrum():
    from torch_fourier_filter.whitening import power_spectral_density, whitening_filter
    shape=(16,21)
    image=torch.randn(shape,device=device)
    ft=torch.fft.rfftn(image)
    size=24
    f=torch.fft.fftfreq(shape[0],device=device)[:,None]
    r=(f.square()+torch.fft.rfftfreq(shape[1],device=device)[None,:].square()).sqrt()
    bins=torch.linspace(0,math.sqrt(2)/2,size,device=device)
    weights=(1-(r[...,None]-bins).abs()/(bins[1]-bins[0])).clamp(min=0)
    count=weights.sum((0,1))
    reference=((weights*ft.abs().square()[...,None]).sum((0,1))/count.clamp_min(1e-12)).sqrt()
    try: actual=power_spectral_density(ft,num_freq_bins=size,real_space_last_dim=shape[-1])
    except TypeError as err:
        assert not candidate and "real_space_last_dim" in str(err)
        legacy=power_spectral_density(ft.contiguous(),num_freq_bins=size)
        return {"max_radial_spectrum_error":(reference-legacy).abs().max().item(),"odd_width_supported":False}
    assert candidate
    torch.testing.assert_close(reference,actual,atol=1e-4,rtol=1e-5)
    w=whitening_filter(ft,num_freq_bins=size,real_space_last_dim=shape[-1],output_rfft=False)
    assert w.shape==image.shape and w.device==image.device
    return {"max_radial_spectrum_error":(reference-actual).abs().max().item(),"full_output_shape":list(w.shape)}

result={"conv":fold,"rot":rotate,"tomo":spectrum}[project]()
print("Compared",json.dumps(dict(project=project,revision=revision,device=device,**result)))
