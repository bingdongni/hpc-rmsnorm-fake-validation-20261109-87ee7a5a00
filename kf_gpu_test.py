import os, sys, warnings
from pathlib import Path
import torch
import pytest
root=Path('torchkf').resolve()
sys.path.insert(0,str(root/'src'))
from torch_kf import GaussianState,KalmanFilter
assert Path(sys.modules['torch_kf'].__file__).resolve().is_relative_to(root)
assert torch.cuda.is_available()
print("Device",torch.cuda.get_device_name(0), torch.__version__)
torch.manual_seed(7)
D,B,T=3,2,4
eye=torch.eye(D,dtype=torch.float64,device='cuda')
state=GaussianState(torch.randn(T,B,D,1,dtype=eye.dtype,device='cuda'), eye.repeat(T,B,1,1))
f=eye*0.7; noises=torch.stack([eye*0.25,eye*1.5])
model=KalmanFilter(eye,eye,eye,eye)
with warnings.catch_warnings(record=True) as events:
    warnings.simplefilter('always')
    out=model.rts_smooth(state,process_matrix=f,process_noise=noises)
    assert len(events)==1 and 'process_noise' in str(events[0].message)
error=0.
for i,noise in enumerate(noises):
    ref=KalmanFilter(f,eye,noise,eye).rts_smooth(state[:,i])
    for a,b in [(out.mean[:,i],ref.mean),(out.covariance[:,i],ref.covariance)]:
        torch.testing.assert_close(a,b)
        error=max(error,(a-b).abs().max().item())
for t in (0,1):
    short=state[:t]
    got=model.rts_smooth(short,process_matrix=f)
    torch.testing.assert_close(got.mean,short.mean)
    torch.testing.assert_close(got.covariance,short.covariance)
print("Compared CUDA shared and batched overrides against per-filter defaults, including a 3x3 matrix with three transitions. Max absolute error",error)
os.chdir(root);sys.exit(pytest.main(['-q','tests']))

