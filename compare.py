import asyncio
from pathlib import Path
import json
import sys
import os
import time

name, tree, revision = sys.argv[1:]
tree=Path(tree).resolve()
candidate=revision=="candidate"
source={"llama":tree/"packages/llama-index-workflows/src", "kits":tree, "web":tree/"src"}[name]
sys.path.insert(0,str(source))

async def agentflow():
    from workflows import Workflow,step
    from workflows.events import StartEvent, StopEvent
    from workflows.retry_policy import retry_policy, stop_before_delay, stop_after_attempt, wait_none
    class Busy(Exception):
        retry_after_seconds=45.
    options=dict(stop=stop_before_delay(10)|stop_after_attempt(3),wait=wait_none())
    try:
        policy=retry_policy(**options,retry_after=lambda exc:exc.retry_after_seconds)
        assert candidate
    except TypeError as err:
        assert not candidate and "retry_after" in str(err)
        policy=retry_policy(**options)
    class Run(Workflow):
        calls=0
        @step(retry_policy=policy)
        async def generate(self, ev:StartEvent)->StopEvent:
            self.calls+=1
            if self.calls==1:raise Busy("retry later")
            return StopEvent(result="second request")
    w=Run(timeout=5)
    try:
        assert await w.run()=="second request" and not candidate
    except Busy:
        assert candidate
    assert w.calls==(1 if candidate else 2)
    return {"provider_calls":w.calls,"server_hint_seconds":45,"retry_budget_seconds":10}

def detector():
    from KitNET.KitNET import KitNET
    import numpy as np
    rng=np.random.RandomState(36)
    train=rng.rand(181,16)
    x=rng.rand(2048,16)
    model=KitNET(16,4,FM_grace_period=30,AD_grace_period=150)
    for row in train: model.process(row)
    start=time.perf_counter()
    expected=np.array([model.process(row) for row in x])
    single=time.perf_counter()-start
    start=time.perf_counter()
    try: scores=model.execute(x)
    except (IndexError,ValueError) as err:
        assert not candidate
        return {"batch_supported":False,"single_score_count":len(expected),"old_batch_exception":type(err).__name__}
    if not candidate:
        assert np.shape(scores)!=expected.shape or not np.allclose(scores,expected)
        return {"batch_supported":False,"old_output_shape":list(np.shape(scores))}
    batch=time.perf_counter()-start
    assert scores.shape==(len(x),) and model.n_executed==2*len(x) and model.n_trained==len(train)
    np.testing.assert_allclose(scores,expected,rtol=1e-12,atol=1e-12)
    return {"samples":len(x),"max_score_error":float(np.max(np.abs(scores-expected))),"row_seconds":single,"batch_seconds":batch}

def shards():
    from webdataset import ResampledShards, WebDataset, TarWriter, split_by_node
    import torch.distributed as dist
    import tempfile
    from collections import Counter
    from itertools import islice
    dist.init_process_group("gloo")
    rank=dist.get_rank()
    assert dist.get_world_size()==2
    with tempfile.TemporaryDirectory() as tmp:
        paths=[str(Path(tmp)/f"{i}.tar") for i in range(3)]
        for i,path in enumerate(paths):
            with TarWriter(path) as w:w.write({"__key__":str(i),"txt":str(i)})
        try: s=ResampledShards(paths,nshards=2048,seed=28,deterministic=True,weights=[0,1,3])
        except TypeError as err:
            assert not candidate and "weights" in str(err)
            s=ResampledShards(paths,nshards=2048,seed=28,deterministic=True)
        keys=[paths.index(r["url"]) for r in s]
        summary=Counter(keys)
        if candidate:
            assert not summary[0] and .70<summary[2]/len(keys)<.80
            ds=WebDataset(paths,resampled=True,shardshuffle=False,shard_weights=[0,1,0],nodesplitter=split_by_node)
            assert [r["__key__"] for r in islice(ds,8)]==["1"]*8
        else:
            assert summary[0]>0
        by_rank=[None,None]
        dist.all_gather_object(by_rank,(keys[:16],dict(summary)))
        assert by_rank[0][0]!=by_rank[1][0]
    dist.destroy_process_group()
    return {"rank_counts":[r[1] for r in by_rank],"rows_per_rank":2048}

result=asyncio.run(agentflow()) if name=="llama" else detector() if name=="kits" else shards()
if int(os.environ.get("RANK","0"))==0:
    print("Compared",json.dumps(dict(project=name,revision=revision,**result)))
