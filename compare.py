import json
from pathlib import Path
import sys
import torch

project, tree, device, revision = sys.argv[1:5]
root=Path(tree).resolve()
sys.path.insert(0, str(root if project=="ms" else root/"src"))
candidate=revision=="candidate"
torch.manual_seed(103)
torch.set_num_threads(1)
if device=="cuda":
    assert torch.cuda.is_available() and "4060" in torch.cuda.get_device_name()
    print("Device",torch.cuda.get_device_name(),torch.__version__)


def similarity():
    from pytorch_msssim import SSIM, ssim
    import torch.nn.functional as F
    sizes=[(2,2,17,19),(2,2,9,13,15)]
    results={}
    for shape in sizes:
        x=torch.rand(shape,device=device,dtype=torch.float64,requires_grad=True)
        y=torch.rand_like(x)
        metric=SSIM(data_range=1,win_size=5,spatial_dims=len(shape)-2,channel=2)
        try:
            actual=metric(x,y,return_map=True)
        except TypeError as exc:
            assert not candidate and "return_map" in str(exc)
            assert metric(x,y).ndim==0
            return {"local_map":False}
        assert candidate
        # Materialize a full multidimensional Gaussian kernel as the reference.
        coords=torch.arange(-2,3,device=device,dtype=torch.float64)
        v=torch.exp(-coords.square()/(2*1.5**2))
        v=v/v.sum()
        kernel=v
        for _ in range(len(shape)-3):
            kernel=kernel.unsqueeze(-1)*v
        weight=kernel.expand(2,1,*kernel.shape).contiguous()
        conv=F.conv2d if len(shape)==4 else F.conv3d
        filt=lambda z:conv(z,weight,groups=2)
        mx,my=filt(x),filt(y)
        vx,vy,covxy=filt(x*x)-mx*mx,filt(y*y)-my*my,filt(x*y)-mx*my
        ref=(2*mx*my+0.01**2)/(mx*mx+my*my+0.01**2)*(2*covxy+0.03**2)/(vx+vy+0.03**2)
        assert actual.shape==ref.shape
        err=(actual-ref).abs().max().item()
        xgrad=torch.autograd.grad(actual.square().mean(),x)[0]
        rgrad=torch.autograd.grad(ref.square().mean(),x)[0]
        err_grad=(xgrad-rgrad).abs().max().item()
        assert err<0.00001 and err_grad<1e-6,(err,err_grad)
        torch.testing.assert_close(actual.mean(),ssim(x,y,win_size=5,data_range=1))
        results[str(len(shape)-2)+"d"]={"shape":list(actual.shape),"map_error":err,"gradient_error":err_grad}
    return results


def smoothing():
    from torch_kf import GaussianState, KalmanFilter
    kw={"device":device,"dtype":torch.float64}
    T,D,B=6,2,3
    I=torch.eye(D,**kw)
    h=torch.tensor([[1.,0.]],**kw)
    r=torch.tensor([[0.4]],**kw)
    f=I.repeat(T-1,1,1)
    dt=torch.tensor([0.25,2,1.25,0.5,1.5],**kw)
    f[:,0,1]=dt
    q=(0.12*dt[:,None,None])*I
    model=KalmanFilter(I,h,I,r)
    state=GaussianState(torch.zeros(B,D,1,**kw),I.expand(B,D,D))
    y=torch.randn(T,B,1,1,**kw)
    history=[]
    for t in range(T):
        if t: state=model.predict(state,process_matrix=f[t-1],process_noise=q[t-1])
        state=model.update(state,y[t]); history.append(state)
    trajectory=GaussianState(torch.stack([s.mean for s in history]),torch.stack([s.covariance for s in history]))
    try: smoothed=model.rts_smooth(trajectory,process_matrix=f,process_noise=q)
    except TypeError as exc:
        assert not candidate and "process_matrix" in str(exc)
        return {"time_varying_smoother":False}
    assert candidate
    # A single block Gaussian solve conditions every state on the whole observation vector.
    a=torch.eye(T*D,**kw)
    for i in range(T-1): a[(i+1)*D:(i+2)*D,i*D:(i+1)*D]=-f[i]
    prior=a.T @ torch.linalg.solve(torch.block_diag(I,*q),a)
    H=torch.block_diag(*([h]*T))
    precision=prior+H.T@H / r.item()
    cov=torch.linalg.inv(precision)
    means=torch.linalg.solve(precision,H.T@y.reshape(T,B)/r.item()).reshape(T,D,B).transpose(1,2)
    diag=torch.stack([cov[t*D:(t+1)*D,t*D:(t+1)*D] for t in range(T)])
    em=(smoothed.mean[...,0]-means).abs().max().item()
    ec=(smoothed.covariance-diag[:,None]).abs().max().item()
    assert em<1e-12 and ec<1e-12,(em,ec)
    return {"sequences":B,"timesteps":T,"max_mean_error":em,"max_covariance_error":ec}


def rollout():
    from pytorch_mppi import MPPI
    from pytorch_mppi.mppi import SpecificActionSampler
    kw={"device":device,"dtype":torch.float64}
    result={}
    for M in (1,4):
        observations=[]
        class Recorder(SpecificActionSampler):
            def sample_trajectories(self,state,info):
                return state.new_zeros(1,5,2)
            def specific_dynamics(self,next_state,state,action,t):
                observations.append((state.clone(),next_state.clone(),action.clone()))
                return next_state+0.3
        advance=lambda x,u:0.8*x+u+0.75
        c=MPPI(advance,lambda x,u:x.square().sum(-1),2,torch.eye(2,**kw),num_samples=16,horizon=5,
               device=device,rollout_samples=M,specific_action_sampler=Recorder(),
               terminal_state_cost=lambda xs,us:xs[...,-1,:].sum(-1))
        initial=torch.tensor([0.5,-1],**kw)
        c.command(initial)
        state=initial.expand(M,16,2)
        errors=[]
        for old,nxt,u in observations:
            errors.append((old-state).abs().max().item())
            expected=advance(state,u)
            torch.testing.assert_close(nxt,expected)
            state=expected+0.3
        assert len(errors)==5
        torch.testing.assert_close(c.states[:,:,-1],state)
        err=max(errors)
        assert err<1e-12 if candidate else err>0.1
        result[str(M)]={"steps":5,"max_previous_state_error":err}
    return result


out={"ms":similarity,"torchkf":smoothing,"mppi":rollout}[project]()
print("Compared",json.dumps({"project":project,"revision":revision,"device":device,**out}))
