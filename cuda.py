import torch
from hqq.core.quantize import Quantizer, HQQLinear, BaseQuantizeConfig
from torch.nn.functional import linear
assert torch.cuda.is_available()
print('GPU',torch.cuda.get_device_name(0),'PyTorch',torch.__version__,'CUDA',torch.version.cuda)
torch.manual_seed(213)
torch.set_num_threads(2)
cases=0
for dev in ('cpu','cuda'):
    for shape in [(13,24),(10,24),(3,4,8)]:
        w=torch.randn(shape,device=dev)
        for axis in (0,1):
            for gs in (None,8,24,w.numel()):
                for channel in (False,True):
                    if len(shape)>2 and (not channel or gs is None):continue
                    for view in (False,True):
                        args=dict(nbits=3,axis=axis,group_size=gs,channel_wise=channel,optimize=False,device=dev)
                        qw,qm=Quantizer.quantize(w,bitpack=False,**args)
                        pw,pm=Quantizer.quantize(w,compute_dtype=torch.float16,view_as_float=view,**args)
                        torch.testing.assert_close(Quantizer.dequantize(pw,pm),Quantizer.dequantize(qw,qm),rtol=0,atol=0)
                        cases+=1
print('Compared',cases,'packed/unpacked quantizer outputs across CPU and CUDA')
w=torch.randn(13,24,device='cuda');x=torch.randn(3,24,device='cuda',dtype=torch.float16)
for axis in (0,1):
    for channel in (False,True):
        for view in (False,True):
            cfg=BaseQuantizeConfig(nbits=3,group_size=8,axis=axis)
            cfg['weight_quant_params'].update(channel_wise=channel,optimize=False,view_as_float=view)
            layer=HQQLinear.from_weights(w,None,cfg,device='cuda',compute_dtype=torch.float16)
            q,m=Quantizer.quantize(w,bitpack=False,**cfg['weight_quant_params'])
            m['scale'],m['zero']=m['scale'].half(),m['zero'].half()
            expected=linear(x,Quantizer.dequantize(q,m))
            torch.testing.assert_close(layer(x),expected,rtol=0,atol=0)
print('Compared 8 CUDA HQQLinear.forward results with the unpacked reference')

