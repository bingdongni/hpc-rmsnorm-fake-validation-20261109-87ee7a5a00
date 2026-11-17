from pathlib import Path
import json, os, sys
name,tree,dev,rev=sys.argv[1:]
root=Path(tree).resolve();sys.path.insert(0,str(root))
import torch
row=dict(project=name,revision=rev,device=dev)
if dev=='cuda':
 assert torch.cuda.is_available()
 row['gpu']=torch.cuda.get_device_name()
torch.manual_seed(91)
torch.set_num_threads(1)
if name=='lin':
 from linear_operator.operators import ToeplitzLinearOperator as T
 import linear_operator
 assert Path(linear_operator.__file__).resolve().is_relative_to(root)
 n=512
 i=torch.arange(n,device=dev,dtype=torch.float64)
 a=torch.exp(-i/5).requires_grad_();b=torch.exp(-i/3).requires_grad_()
 rhs=torch.randn(n,3,device=dev,dtype=a.dtype)
 if dev=='cuda': torch.cuda.reset_peak_memory_stats()
 op=T(a)*T(b)
 y=op@rhs
 if dev=='cuda': row['peak_bytes']=torch.cuda.max_memory_allocated()
 assert isinstance(op,T)==(rev=='candidate')
 distance=(i[:,None]-i).abs().long()
 truth=(a[distance]*b[distance])@rhs
 torch.testing.assert_close(y,truth,rtol=1e-7,atol=1e-7)
 if rev=='candidate':
  grad=torch.randn_like(y)
  for u,v in zip(torch.autograd.grad(y,(a,b),grad),torch.autograd.grad(truth,(a,b),grad)):torch.testing.assert_close(u,v)
  a2=torch.exp(-torch.arange(65536,device=dev)/5)
  out=(T(a2)*T(a2))@torch.ones(65536,1,device=dev)
  torch.testing.assert_close(out[32768,0],torch.tensor((1+torch.exp(torch.tensor(-.4)))/(1-torch.exp(torch.tensor(-.4))),device=dev),rtol=1e-5,atol=1e-5)
  row['large_n']=len(out)
if name=='xt':
 from x_transformers.x_transformers import Attention, Attend
 import x_transformers
 assert Path(x_transformers.__file__).resolve().is_relative_to(root)
 # Compare at the projected attention entry point, as well as via Attention.forward.
 kwargs=dict(device=dev,dtype=torch.float16 if dev=='cuda' else torch.float32)
 Q=torch.randn(1,16,1,64,**kwargs);K=torch.randn(1,2,8192,64,**kwargs);V=torch.randn_like(K)
 f=Attend(causal=True)
 if dev=='cuda': torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
 out,_=f(Q,K,V)
 if dev=='cuda':row['peak_delta_bytes']=torch.cuda.max_memory_allocated()-before
 ref,_=f(Q,K.repeat(1,8,1,1),V.repeat(1,8,1,1))
 torch.testing.assert_close(out,ref,rtol=.007,atol=.0003)
 att=Attention(dim=32,heads=4,kv_heads=2,dim_head=8).to(**kwargs)
 x=torch.randn(2,3,32,**kwargs,requires_grad=True)
 context=torch.randn(2,17,32,**kwargs,requires_grad=True)
 y=att(x,context=context)
 if isinstance(y,tuple): y=y[0]
 for g in torch.autograd.grad(y.sum(),(x,context)): assert torch.isfinite(g).all()
 row['context_tokens']=8192
if name=='quad':
 import torchquad
 assert Path(torchquad.__file__).resolve().is_relative_to(root)
 if rev=='base':assert not hasattr(torchquad,'GaussLobattoLegendre')
 else:
  torch.set_default_dtype(torch.float64);torch.set_default_device(dev)
  a=torch.tensor([[-1.,2.],[0.,1.]],device=dev,requires_grad=True)
  def fun(x):return x[:,0]**3*x[:,1]**5
  out=torchquad.GaussLobattoLegendre().integrate(fun,dim=2,N=(3,4),integration_domain=a)
  l,r=a[:,0],a[:,1];ref=(r[0]**4-l[0]**4)*(r[1]**6-l[1]**6)/24
  torch.testing.assert_close(out,ref);torch.testing.assert_close(torch.autograd.grad(out,a)[0],torch.autograd.grad(ref,a)[0])
  row['integral']=out.item()
print('Compared '+json.dumps(row),flush=True)

