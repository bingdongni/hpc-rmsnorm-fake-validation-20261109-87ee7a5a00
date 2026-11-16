"""Public-entry comparisons for three independent upstream changes."""
import itertools
import json
import math
import os
from pathlib import Path
import sys

os.environ["SCIPY_ARRAY_API"] = "1"
import torch

name, checkout, device, revision = sys.argv[1:]
sys.path.insert(0, str(Path(checkout).resolve()))
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
if device == "cuda":
    assert torch.cuda.is_available() and "4060" in torch.cuda.get_device_name()
    print("Device", torch.cuda.get_device_name(), torch.__version__)
torch.set_num_threads(1)
torch.manual_seed(847)
fixed = revision == "candidate"

def geometry():
    from geomstats.geometry.euclidean import Euclidean
    from geomstats.geometry.spd_matrices import SPDMatrices
    from geomstats.learning.geometric_median import GeometricMedian

    euclidean = GeometricMedian(Euclidean(1))
    x = torch.tensor([[0.], [0.], [1.]])
    y = euclidean.fit(x).estimate_
    eye = torch.eye(2)
    matrices = torch.stack([eye * 4, eye, eye])
    z = GeometricMedian(SPDMatrices(2)).fit(matrices).estimate_
    weights = torch.tensor([3., 2., 1.])
    points = torch.tensor([[0., 0.], [2., 0.], [0., 1.]])
    w = GeometricMedian(Euclidean(2), init_point=points[0]).fit(points, weights=weights).estimate_
    assert y.device.type == z.device.type == w.device.type == device
    if fixed:
        assert y.abs().max() < 5e-6 and (z - eye).abs().max() < 1e-7
        assert w.abs().max() < 1e-7
    else:
        assert y.item() == 1.0
    return dict(median=y.item(), objective=x.sub(y).abs().sum().item(),
                spd_error=(z-eye).abs().max().item(), weighted_error=w.abs().max().item())

def paths():
    import pyro.distributions as dist
    a = torch.randn(2, 3, dtype=torch.float64)
    t = torch.randn(5, 3, 3, dtype=torch.float64)
    means = torch.randn(5, 3, 2, dtype=torch.float64)
    obs = dist.Normal(means, 1.).to_event(1)
    hmm = dist.DiscreteHMM(a, t, obs)
    y = torch.randn(7, 1, 5, 2, dtype=torch.float64)
    assert hmm.log_prob(y).shape == (7, 2)
    assert hasattr(hmm, "map_states") == fixed
    if not fixed:
        return dict(filter_shape=list(hmm.filter(y).logits.shape))
    actual = hmm.map_states(y)
    states = list(itertools.product(range(3), repeat=5))
    likelihood = obs.log_prob(y.unsqueeze(-2))
    scores = []
    for s in states:
        score = (a.softmax(-1) * t[0].softmax(-1)[:,s[0]]).sum(-1).log()
        for j, state in enumerate(s):
            score = score + likelihood[..., j, state]
            if j:
                score = score + t[j].log_softmax(-1)[s[j-1], state]
        scores.append(score)
    reference = torch.tensor(states)[torch.stack(scores,-1).argmax(-1)]
    assert actual.device.type == device and torch.equal(actual,reference)
    return dict(exhaustive_paths=len(states), decoded_sequences=14, length=5)

def attention():
    import rotary_embedding_torch as rotary
    import torch.nn.functional as F
    assert hasattr(rotary,"get_yarn_freqs") == fixed
    if not fixed:
        return dict(yarn_available=False)
    dim, factor, window = 64, 4., 2048
    positions = torch.arange(8176,8209)
    angles, scale = rotary.get_yarn_freqs(positions,dim,original_max_seq_len=window,factor=factor)
    # Scalar double-precision band reference, independent of the tensor ramp.
    lo = math.floor(dim/2*math.log(window/(2*math.pi*32),10000))
    hi = math.ceil(dim/2*math.log(window/(2*math.pi),10000))
    freq = []
    for j in range(dim//2):
        amp = min(1., max(0., (j-lo)/(hi-lo)))
        freq.append(10000**(-2*j/dim)*((1-amp)+amp/factor))
    phases = positions.double()[:,None]*torch.tensor(freq,dtype=torch.float64)
    max_angle = (angles[:,::2]-phases).abs().max().item()
    assert max_angle < 0.001
    errors = {}
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        q,k,v = torch.randn(3,2,4,33,80).to(dtype).unbind()
        query = rotary.apply_rotary_emb(angles[-1:],q[...,-1:,:],scale=scale)
        key = rotary.apply_rotary_emb(angles,k,scale=scale)
        def oracle(x,phase):
            pairs = torch.view_as_complex(x[...,:dim].double().reshape(*x.shape[:-1],dim//2,2))
            rot = pairs * torch.polar(torch.ones_like(phase), phase) * (1+0.1*math.log(factor))
            return torch.cat([torch.view_as_real(rot).flatten(-2),x[...,dim:].double()],-1).to(dtype)
        qref = oracle(q[...,-1:,:],phases[-1:])
        kref = oracle(k,phases)
        result = F.scaled_dot_product_attention(query,key,v)
        reference = F.scaled_dot_product_attention(qref,kref,v)
        err = (result.float()-reference.float()).abs().max().item()
        assert result.device.type == device and err < {torch.float32:.001,torch.float16:.006,torch.bfloat16:.035}[dtype], err
        assert torch.equal(key[...,dim:], k[...,dim:])
        errors[str(dtype)] = err
    return dict(positions=[8176,8208],rotary_dim=64,head_dim=80,max_angle_error=max_angle,attention_max_error=errors)

with torch.device(device):
    result = {"geom": geometry, "pyro": paths, "rotary": attention}[name]()
print("Compared",json.dumps(dict(project=name,revision=revision,device=device,**result)))
