from datetime import timedelta
import statistics
import sys
import time
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.modules['ray'] = types.ModuleType('ray')
from unirl.algorithms.grpo import GRPO
from unirl.train.stack.loss_scale import resolve_loss_scales
from unirl.types.sample import Part


class Backend:
    def __init__(self, group):
        self.group = group

    def all_reduce_loss_sums(self, values):
        value = torch.tensor(values, dtype=torch.float64)
        self.group.allreduce([value]).wait()
        return value.tolist()

    def gradient_average_world_size(self):
        return 2


def worker(rank, port):
    store = dist.TCPStore('127.0.0.1', port, None, False, timeout=timedelta(seconds=30), use_libuv=False)
    options = dist.ProcessGroupGloo._Options()
    options._devices = [dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')]
    options._timeout = timedelta(seconds=30)
    group = dist.ProcessGroupGloo(store, rank, 2, options)
    backend = Backend(group)
    algorithm = GRPO(stage=object(), loss_agg_mode='seq-mean-token-mean', sampling_temperature=1.0)
    ids = [str(i) for i in range(32)]
    unpadded = Part(sample_ids=ids)
    mask = torch.zeros(32, dtype=torch.bool)
    if rank == 1:
        mask[-1] = True
    padded = Part(sample_ids=ids, is_padding=mask)
    micros = [(0, 8), (8, 16), (16, 24), (24, 32)]
    for name, part in (('unpadded', unpadded), ('padded', padded)):
        group.barrier().wait()
        for _ in range(10):
            resolve_loss_scales(algorithm, part, micros=micros, order=None, backend=backend,
                                rank_info=None, owner='benchmark')
        samples = []
        for _ in range(100):
            start = time.perf_counter()
            resolve_loss_scales(algorithm, part, micros=micros, order=None, backend=backend,
                                rank_info=None, owner='benchmark')
            samples.append((time.perf_counter() - start) * 1000)
        group.barrier().wait()
        print('rank', rank, name, 'median_ms', round(statistics.median(samples), 4), flush=True)
    del group


if __name__ == '__main__':
    store = dist.TCPStore('127.0.0.1', 0, None, True, timeout=timedelta(seconds=30), use_libuv=False)
    mp.spawn(worker, args=(store.port,), nprocs=2, join=True)
