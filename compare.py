import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

key, tree, device, revision = sys.argv[1:]
root = Path(tree).resolve()
sys.path.insert(0, str(root / "src" if key == "marvin" else root))
row = dict(project=key, revision=revision, device=device)

def lightly_case():
    import torch
    import torch.distributed as dist
    from lightly.loss import VICRegLoss
    from lightly.loss import vicreg_loss
    assert Path(vicreg_loss.__file__).resolve().is_relative_to(root)
    multi = "RANK" in os.environ
    if multi: dist.init_process_group("gloo")
    rank=dist.get_rank() if multi else 0
    torch.manual_seed(27 + rank)
    cuda=device=="cuda"
    if cuda: row["gpu"]=torch.cuda.get_device_name()
    size=4096 if cuda else 89
    x=torch.randn(16,size,device=device,requires_grad=True)
    y=torch.randn_like(x,requires_grad=True)
    settings=dict(gather_distributed=multi)
    if cuda: torch.cuda.reset_peak_memory_stats()
    baseline=VICRegLoss(**settings)(x,y)
    grads=torch.autograd.grad(baseline,(x,y))
    if cuda: row["dense_peak_bytes"]=torch.cuda.max_memory_allocated()
    if revision=="candidate":
        if cuda: torch.cuda.reset_peak_memory_stats()
        loss=VICRegLoss(covariance_block_size=128 if cuda else 13,**settings)(x,y)
        actual=torch.autograd.grad(loss,(x,y))
        torch.testing.assert_close(loss,baseline,rtol=4e-5,atol=1e-5)
        for a,e in zip(actual,grads): torch.testing.assert_close(a,e,rtol=5e-5,atol=1e-5)
        if cuda:
            row["block_peak_bytes"]=torch.cuda.max_memory_allocated()
            assert row["block_peak_bytes"] < row["dense_peak_bytes"] * .5
        row["torch_version"]=torch.__version__
        row["rank_count"]=dist.get_world_size() if multi else 1
    else:
        import inspect
        assert "covariance_block_size" not in inspect.signature(VICRegLoss).parameters
    if multi: dist.destroy_process_group()

if __name__=="__main__":
    lightly_case()
    print("Compared", json.dumps(row))
