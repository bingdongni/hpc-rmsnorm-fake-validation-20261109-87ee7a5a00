import json
import os
import sys
from pathlib import Path
from itertools import chain

project, tree, device, revision = sys.argv[1:5]
root = Path(tree).resolve()
sys.path.insert(0, str(root/"src" if project == "rupt" else root))
candidate = revision == "candidate"


def attention():
    import torch
    from nystrom_attention import NystromAttention

    if device == "cuda":
        assert torch.cuda.is_available()
        print("Device", torch.cuda.get_device_name(), torch.__version__)
        assert "4060" in torch.cuda.get_device_name()
    torch.manual_seed(23)
    torch.set_num_threads(1)
    model = NystromAttention(dim=32, dim_head=8, heads=4, num_landmarks=8,
                             pinv_iterations=4, residual=False).to(device).eval()
    x = torch.randn(1,32,32,device=device,requires_grad=True)
    other = 16*torch.randn_like(x)
    out=model(x)
    batched=model(torch.cat([x,other]))[:1]
    difference=(batched-out).abs().max().item()
    grad=(torch.autograd.grad(batched.square().sum(),x)[0]-
          torch.autograd.grad(out.square().sum(),x)[0]).abs().max().item()
    if candidate:
        assert difference<2e-5 and grad<1e-4,(difference,grad)
    else:
        assert difference>1e-4,(difference,grad)
    stats={"max_batch_error":difference,"max_gradient_error":grad}
    if device == "cuda":
        import copy
        for dtype in (torch.float16,torch.bfloat16):
            try:
                half_model=copy.deepcopy(model).to(dtype=dtype)
                low=half_model(x.to(dtype))
            except RuntimeError as exc:
                assert not candidate and ("type" in str(exc).lower() or "dtype" in str(exc).lower()),exc
                stats[str(dtype)]="dtype mismatch"
            else:
                assert candidate and low.dtype==dtype
                error=(low.float()-out).abs().max().item()
                assert error < (0.008 if dtype==torch.float16 else 0.06),error
                stats[str(dtype)]=error
    return stats


def sampling():
    import numpy as np
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest
    from lhotse.dataset.sampling import WeightedSimpleCutSampler

    cuts=DummyManifest(CutSet,begin_id=0,end_id=100)
    def draw(rank,world,ambient,epoch):
        np.random.seed(ambient)
        s=WeightedSimpleCutSampler(cuts,list(range(1,101)),num_samples=48,max_cuts=4,
                                   world_size=world,rank=rank,seed=37)
        s.set_epoch(epoch)
        return [list(batch.ids) for batch in s]
    full=draw(0,1,101,2)
    ranks=[draw(i,3,20+i,2) for i in range(3)]
    actual=[b for step in zip(*ranks) for b in step]
    flat=list(chain.from_iterable(actual))
    assert (actual==full)==candidate
    assert draw(0,1,101,2)!=draw(0,1,202,3)
    if candidate: assert len(set(flat))==48
    result = {"distinct":len(set(flat)),"drawn":len(flat),"ranks":3,"recomposes_single_rank":actual==full}
    if sys.platform == "linux":
        from tempfile import TemporaryDirectory
        import torch.multiprocessing as mp
        with TemporaryDirectory() as temp:
            mp.spawn(rank_worker,args=(temp,),nprocs=3,join=True)
            gathered=json.loads((Path(temp)/"cuts.json").read_text())
            assert gathered==actual
        result["backend"]="gloo, three processes"
    return result


def rank_worker(rank, temp):
    import numpy as np
    import torch.distributed as dist
    from lhotse import CutSet
    from lhotse.testing.dummies import DummyManifest
    from lhotse.dataset.sampling import WeightedSimpleCutSampler
    dist.init_process_group("gloo",init_method=(Path(temp)/"rendezvous").as_uri(),rank=rank,world_size=3)
    try:
        cuts=DummyManifest(CutSet,begin_id=0,end_id=100)
        np.random.seed(20+rank)
        s=WeightedSimpleCutSampler(cuts,list(range(1,101)),num_samples=48,max_cuts=4,seed=37)
        assert s.rank==rank and s.world_size==3
        s.set_epoch(2)
        ids=[list(b.ids) for b in s]
        batches=[None]*3
        dist.all_gather_object(batches,ids)
        if rank==0:
            (Path(temp)/"cuts.json").write_text(json.dumps([b for group in zip(*batches) for b in group]))
    finally:
        dist.destroy_process_group()


def change_points():
    import numpy as np
    import ruptures as rpt
    rng=np.random.default_rng(13)
    times=np.cumsum(rng.integers(1,9,40))
    # The continuous signal changes slope at row 16, which ends the first segment.
    knot=17
    y=np.column_stack([times+2*np.maximum(0,times-times[knot-1]),
                       4-0.25*times+1.5*np.maximum(0,times-times[knot-1])])
    try:
        alg=rpt.Dynp(model="clinear",params={"timestamps":times+2**60},min_size=3,jump=1)
    except TypeError as exc:
        assert not candidate and "timestamps" in str(exc)
        return {"supports_timestamps": False}
    assert candidate
    result=alg.fit_predict(y,n_bkps=1)
    assert result==[knot,len(y)],result
    pelt=rpt.Pelt(model="clinear",params={"timestamps":times},min_size=3,jump=1)
    assert pelt.fit_predict(y,pen=2)==result
    row_index_cost=rpt.costs.CostCLinear().fit(y).sum_of_costs(result)
    cost=alg.cost.sum_of_costs(result)
    assert row_index_cost>1 and cost<1e-10
    return {"supports_timestamps":True,"dynp_and_pelt":result,"elapsed_cost":float(cost),"row_index_cost":float(row_index_cost)}


if __name__=="__main__":
    result={"rupt":change_points,"nys":attention,"lhotse":sampling}[project]()
    print("Compared",json.dumps({"project":project,"revision":revision,"device":device,**result}))
