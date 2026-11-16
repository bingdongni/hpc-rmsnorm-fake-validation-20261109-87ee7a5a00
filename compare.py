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

def marvin_case():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MARVIN_DATABASE_URL"] = "sqlite+aiosqlite:///" + (Path(tmp) / "case.db").as_posix()
        os.environ["MARVIN_HOME"] = tmp
        import marvin
        from marvin.thread import Thread
        from marvin.engine.llm import UserMessage
        from marvin.database import get_async_engine
        assert Path(marvin.__file__).resolve().is_relative_to(root)
        async def check():
            parent = Thread()
            await parent.add_messages_async([UserMessage(str(i)) for i in range(280)])
            await parent.add_system_message_async("policy one")
            await parent.add_system_message_async("policy two")
            if revision == "base":
                assert not hasattr(parent, "iter_messages_async")
            else:
                history=[m async for m in parent.iter_messages_async(batch_size=13)]
                assert len(history)==280
                assert set(m.message.parts[0].content for m in history)==set(map(str,range(280)))
                assert await parent.get_messages_async(limit=3)==history[-3:]
                reverse=[m async for m in parent.iter_messages_async(newest_first=True,batch_size=17)]
                assert reverse==history[::-1]
                assert len([m async for m in parent.iter_messages_async(include_system_messages=True)])==282
                row["visible_messages"]=len(history)
            await get_async_engine().dispose()
        asyncio.run(check())

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
        row["rank_count"]=dist.get_world_size() if multi else 1
    else:
        import inspect
        assert "covariance_block_size" not in inspect.signature(VICRegLoss).parameters
    if multi: dist.destroy_process_group()

def nfs_case():
    import nfstream
    from nfstream import NFStreamer, plugins
    assert Path(nfstream.__file__).resolve().is_relative_to(root)
    fixture=root/"tests/pcaps/google_ssl.pcap"
    args=dict(source=str(fixture),splt_analysis=255,accounting_mode=3,n_dissections=0,n_meters=1)
    if revision=="base": assert not hasattr(plugins,"BurstSequence")
    else: args["udps"]=plugins.BurstSequence(sequence_length=8)
    flows=list(NFStreamer(**args))
    assert flows
    runs=0
    for f in flows:
        assert f.bidirectional_packets<=255
        sizes=[]; counts=[]
        for direction,size in zip(f.splt_direction,f.splt_ps):
            if size<=0:continue
            signed=-size if direction else size
            if sizes and (sizes[-1]<0)==(signed<0):
                sizes[-1]+=signed; counts[-1]+=1
            else:sizes.append(signed); counts.append(1)
        runs+=len(sizes)
        if revision=="candidate":
            assert f.udps.burst_bytes==(sizes+[0]*8)[:8]
            assert f.udps.burst_packets==(counts+[0]*8)[:8]
    assert runs>=2
    row.update(real_pcap_flows=len(flows),direction_runs=runs)

if __name__=="__main__":
    {"marvin":marvin_case,"lightly":lightly_case,"nfs":nfs_case}[key]()
    print("Compared",json.dumps(row))

