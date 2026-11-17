import sys
from pathlib import Path
import torch
import torch.distributed as dist
from litdata import StreamingDataset, StreamingDataLoader
from litdata.streaming import Cache
from litdata.streaming.item_loader import TokensLoader

path = sys.argv[1]
if len(sys.argv) > 2:
    Path(path).mkdir(parents=True, exist_ok=True)
    cache = Cache(path, chunk_size=16, item_loader=TokensLoader(5))
    for i in range(8):
        cache[i] = torch.arange(16*i, 16*(i+1), dtype=torch.int32)
    cache.done()
    cache.merge()
    raise SystemExit

def loader(stride):
    dataset = StreamingDataset(path, item_loader=TokensLoader(5, stride=stride), shuffle=False, drop_last=True)
    return StreamingDataLoader(dataset, batch_size=1, num_workers=0)

def start(batch):
    ids=batch[0].tolist()
    assert ids == list(range(ids[0], ids[0]+5))
    return ids[0]

dist.init_process_group("gloo")
try:
    original = loader(4)
    stream = iter(original)
    first = start(next(stream))
    state = original.state_dict()
    suffix = [start(batch) for batch in stream]
    resumed = loader(4)
    resumed.load_state_dict(state)
    assert [start(batch) for batch in resumed] == suffix
    incompatible = loader(5)
    try:
        incompatible.load_state_dict(state)
        next(iter(incompatible))
    except (RuntimeError, ValueError) as exc:
        assert "item_loader" in str(exc), str(exc)
    else:
        raise AssertionError("changed stride wasn't detected")
    ranks = [None]*dist.get_world_size()
    dist.all_gather_object(ranks,[first]+suffix)
    assert len(ranks) == 2 and len(ranks[0]) == len(ranks[1]) == 12
    assert sorted(ranks[0]+ranks[1]) == [16*i+j for i in range(8) for j in (0,4,8)]
    if dist.get_rank()==0:
        print("Two Gloo ranks returned 24 disjoint five-token windows. Both resumed at the saved position and rejected a changed stride.")
finally:
    dist.destroy_process_group()

