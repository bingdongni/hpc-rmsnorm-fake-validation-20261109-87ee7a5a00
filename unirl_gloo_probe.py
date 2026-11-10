import ast
from datetime import timedelta
import os
from pathlib import Path
import sys
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.modules['ray'] = types.ModuleType('ray')
from unirl.algorithms.grpo import GRPO
from unirl.algorithms.gspo import GSPO
from unirl.algorithms.cppo import CPPO
from unirl.algorithms.dppo import DPPO
from unirl.algorithms.drpo import DRPO
from unirl.algorithms.sft import SFT
from unirl.train.stack.loss_scale import resolve_loss_scales
from unirl.types.sample import Part, _part_with_field
from unirl.types.segments.text import TextSegment


def pad_method():
    path = Path(os.environ['UNIRL_PROBE_ROOT']) / 'unirl/trainer/agentic.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AgenticTrainer')
    method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == '_pad_to_dp_multiple')
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {'torch': torch, 'Part': Part, '_part_with_field': _part_with_field}
    exec(compile(module, str(path), 'exec'), scope)
    return scope['_pad_to_dp_multiple']


class Backend:
    def __init__(self, group):
        self.group = group

    def all_reduce_loss_sums(self, values):
        tensor = torch.tensor(values, dtype=torch.float64)
        self.group.allreduce([tensor]).wait()
        return tensor.tolist()

    def gradient_average_world_size(self):
        return 2


def build(count):
    rows = [torch.tensor([1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6]), torch.tensor([7, 8])][:count]
    masks = [torch.tensor([1., 0.]), torch.tensor([0., 1., 1.]), torch.ones(1), torch.ones(2)][:count]
    segment = TextSegment.pack(tokens=rows, log_probs=[torch.zeros(len(x)) for x in rows], loss_mask=masks)
    return Part(sample_ids=[f's{i}' for i in range(count)], segment=segment,
                advantages=torch.tensor([1., -1., 0.5, 0.25][:count]))


def make_algorithm(theta, mode):
    class Stage:
        def replay(self, conditions, *, segment, temperature):
            return theta * segment.tokens.float()

    if mode == 'gspo':
        return GSPO(stage=Stage(), sampling_temperature=1.0)
    kind, reduction = mode.split('/')
    if kind == 'drpo':
        return DRPO(pipeline=types.SimpleNamespace(ar=Stage()), loss_agg_mode=reduction,
                    sampling_temperature=1.0, horizon=1)
    cls = {'grpo': GRPO, 'cppo': CPPO, 'dppo': DPPO}[kind]
    return cls(stage=Stage(), loss_agg_mode=reduction, sampling_temperature=1.0, horizon=1)


def backward(algo, part, scale):
    algo.compute_loss_and_backward(conditions={}, segment=part.segment, advantages=part.advantages,
                                   training_progress=0.0, loss_scale=scale)


def move(part, device):
    part.segment = part.segment.to_device(device)
    part.advantages = part.advantages.to(device)
    return part


def worker(rank, port):
    store = dist.TCPStore('127.0.0.1', port, None, False, timeout=timedelta(seconds=30), use_libuv=False)
    options = dist.ProcessGroupGloo._Options()
    options._devices = [dist.ProcessGroupGloo.create_device(hostname='127.0.0.1')]
    options._timeout = timedelta(seconds=30)
    group = dist.ProcessGroupGloo(store, rank, 2, options)
    device = os.environ.get('UNIRL_PROBE_DEVICE', 'cpu')
    if device == 'cuda':
        torch.cuda.set_device(0)
    try:
        pad = pad_method()
        for count in (1, 3, 4):
            part = build(count)
            padded = pad(types.SimpleNamespace(stack=types.SimpleNamespace(dp_size=2)), part)
            shard_size = padded.batch_size // 2
            shard = move(padded.slice(rank * shard_size, (rank + 1) * shard_size), device)
            part = move(part, device)
            micros = [(i, i + 1) for i in range(shard_size)]
            reductions = ('token-mean', 'seq-mean-token-mean', 'seq-mean-token-sum-norm')
            modes = [f'{kind}/{reduction}' for kind in ('grpo', 'cppo', 'dppo', 'drpo')
                     for reduction in reductions] + ['gspo']
            for mode in modes:
                reference_theta = torch.tensor(0.0, device=device, requires_grad=True)
                backward(make_algorithm(reference_theta, mode), part, 1.0)
                theta = torch.tensor(0.0, device=device, requires_grad=True)
                algorithm = make_algorithm(theta, mode)
                scales, _ = resolve_loss_scales(algorithm, shard, micros=micros, order=None,
                                                backend=Backend(group), rank_info=None, owner='probe')
                for (start, end), scale in zip(micros, scales):
                    backward(algorithm, shard.slice(start, end), scale)
                reduced = theta.grad.detach().cpu()
                group.allreduce([reduced]).wait()
                reduced /= 2
                torch.testing.assert_close(reduced, reference_theta.grad.cpu(), rtol=1e-6, atol=1e-6)
                if rank == 0:
                    print('count', count, 'mode', mode, 'gradient', float(reduced), 'pass', flush=True)
        padded = pad(types.SimpleNamespace(stack=types.SimpleNamespace(dp_size=2)), build(3))
        shard = padded.slice(rank * 2, (rank + 1) * 2)
        try:
            resolve_loss_scales(SFT(stage=object()), shard, micros=[(0, 2)], order=None,
                                backend=Backend(group), rank_info=None, owner='probe')
        except ValueError as exc:
            assert 'zero loss for zero advantage' in str(exc)
        else:
            raise AssertionError('SFT padding was not rejected')
        if rank == 0:
            print('padded SFT rejected on both ranks before any collective', flush=True)
    finally:
        del group


if __name__ == '__main__':
    store = dist.TCPStore('127.0.0.1', 0, None, True, timeout=timedelta(seconds=30), use_libuv=False)
    mp.spawn(worker, args=(store.port,), nprocs=2, join=True)
