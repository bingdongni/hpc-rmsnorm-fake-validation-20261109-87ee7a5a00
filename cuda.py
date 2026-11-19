from pathlib import Path
import numpy as np
import torch
import ot

root=Path.cwd()
assert root in Path(ot.__file__).resolve().parents, ot.__file__
device=torch.device("cuda")
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
rng=np.random.default_rng(325)
for dtype,eps in [(torch.float32,1e-5),(torch.float64,1e-12)]:
    for sizes in [(1,32),(321,512),(1024,1536,2048)]:
        xs=[rng.normal(size=n) for n in sizes]
        ws=[rng.uniform(.5,1.5,size=n) for n in sizes]
        # Independent inverse-CDF reference on ordered empirical atoms.
        xref=[]; cdfs=[]
        for x,a in zip(xs,ws):
            order=np.argsort(x)
            cdf=a[order].cumsum()
            cdfs.append(cdf/cdf[-1]);xref.append(x[order])
        grid=np.unique(np.concatenate(cdfs))
        expected=np.mean([x[np.searchsorted(c,grid)] for x,c in zip(xref,cdfs)],axis=0)
        z,p=ot.lp.free_support_barycenter_1d([torch.tensor(x,device=device,dtype=dtype) for x in xs],[torch.tensor(a,device=device,dtype=dtype) for a in ws])
        assert z.device==p.device==torch.device('cuda:0')
        # Round-off can coalesce very close breakpoints. Compare quantiles at
        # interior probabilities and the first two moments of the distribution.
        mass=np.diff(grid,prepend=0)
        if dtype==torch.float64:
            np.testing.assert_allclose(z.cpu(),expected,rtol=eps,atol=eps)
            np.testing.assert_allclose(p.cpu(),mass,rtol=eps,atol=eps)
        else:
            probes=np.linspace(.005,.995,101)
            got=z.cpu().numpy()[np.searchsorted(p.cpu().numpy().cumsum(),probes)]
            np.testing.assert_allclose(got,expected[np.searchsorted(grid,probes)],atol=eps)
        for degree in (1,2):
            np.testing.assert_allclose((p*z**degree).sum().item(), mass@(expected**degree),rtol=eps,atol=eps)
        print(dtype,sizes,'verified',len(p),'atoms')

