import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from wids import DistributedChunkedSampler

dist.init_process_group('gloo')
rank = dist.get_rank()
try:
    for total in [1, 9, 12]:
        sampler = DistributedChunkedSampler(range(12), num_samples=total, drop_last=True, shuffle=True, seed=17, chunksize=2)
        loader = DataLoader(range(12), sampler=sampler, batch_size=2)
        local = [i for batch in loader for i in batch.tolist()]
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        assert len(sampler) == total // len(gathered)
        assert all(len(row) == len(sampler) for row in gathered), gathered
        assert sorted(i for row in gathered for i in row) == list(range(total - total % len(gathered)))
        if rank == 0: print('Gloo partitions', total, gathered, flush=True)
finally:
    dist.destroy_process_group()

