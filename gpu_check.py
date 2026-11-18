import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torchist.metrics import wasserstein_distance_1d
from hqq.core.quantize import Quantizer
assert torch.cuda.is_available()
print('GPU', torch.cuda.get_device_name(0), 'PyTorch', torch.__version__, 'CUDA', torch.version.cuda)
rng=np.random.default_rng(20261118)
for n in (2, 19, 1024):
    coords = np.cumsum(rng.random(n))
    p, q = rng.random((2,n))
    p[::4] = 0
    q[1::4] = 0
    p, q = p/p.sum(), q/q.sum()
    expected = wasserstein_distance(coords, coords, p, q)
    grid, a, b = [torch.from_numpy(x).cuda() for x in (coords,p,q)]
    for x,y in ((a,b),(a.to_sparse(),b.to_sparse()),(a.to_sparse(),b)):
        assert abs(wasserstein_distance_1d(x,y,grid).item()-expected)<1e-10
print('Compared 9 CUDA histograms with SciPy')
from torch.nn.functional import linear
w = torch.randn(13, 24, device='cuda')
x = torch.randn(2, 24, device='cuda',dtype=torch.float16)
for axis in (0,1):
    for gs in (None, 12):
        for view in (False,True):
            args=dict(nbits=3,group_size=gs,axis=axis,device='cuda',optimize=False)
            qw,meta=Quantizer.quantize(w,bitpack=False,**args)
            expected=linear(x,Quantizer.dequantize(qw,meta).half())
            pw,pm=Quantizer.quantize(w,view_as_float=view,compute_dtype=torch.float16,**args)
            torch.testing.assert_close(linear(x,Quantizer.dequantize(pw,pm).half()),expected,atol=0,rtol=0)
print('Compared 8 packed CUDA linear outputs with unpacked 3-bit weights')

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
torch.manual_seed(281)
w = torch.randn(13, 24, device='cuda')
x = torch.randn(3, 24, device='cuda', dtype=torch.float16)
cfg = BaseQuantizeConfig(nbits=3, group_size=8, axis=1)
cfg['weight_quant_params'].update(channel_wise=False, optimize=False)
layer = HQQLinear.from_weights(w, None, cfg, device='cuda', compute_dtype=torch.float16)
qw, meta = Quantizer.quantize(w, bitpack=False, **cfg['weight_quant_params'])
# HQQLinear stores its scale and zero in compute_dtype.
meta['scale'], meta['zero'] = meta['scale'].half(), meta['zero'].half()
ref = Quantizer.dequantize(qw, meta)
torch.testing.assert_close(layer(x), linear(x,ref),rtol=0,atol=0)
print('Compared HQQLinear.forward with an unpacked 3-bit global quantizer')

