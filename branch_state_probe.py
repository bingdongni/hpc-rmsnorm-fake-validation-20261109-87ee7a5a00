"""Independent GDN state-factor feasibility probe; see branch_state_probe.md."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import torch


def fixture(nodes: int, heads: int, dim: int, device: str, dtype: torch.dtype, shape: str):
    generator = torch.Generator(device=device).manual_seed(20261111)
    parents = [-1]
    for i in range(1, nodes):
        parents.append(i - 1 if shape == 'chain' else 0 if shape == 'star' else (i - 1) // 2)
    ancestry = torch.zeros((nodes, nodes), dtype=torch.bool)
    for i, parent in enumerate(parents):
        ancestry[i, i] = True
        if parent >= 0:
            ancestry[i] |= ancestry[parent]
    tensors = [torch.randn((nodes, heads, dim), device=device, dtype=dtype, generator=generator)
               for _ in range(3)]
    q, k, v = tensors
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    alpha = 0.97 + 0.025 * torch.rand((nodes, heads), device=device, dtype=dtype, generator=generator)
    beta = torch.rand((nodes, heads), device=device, dtype=dtype, generator=generator)
    prefix = torch.randn((heads, dim, dim), device=device, dtype=dtype, generator=generator) / dim**0.5
    return parents, ancestry.to(device), q, k, v, alpha, beta, prefix


def dense(data):
    parents, ancestry, q, k, v, alpha, beta, prefix = data
    states = torch.empty((len(parents), *prefix.shape), device=q.device, dtype=q.dtype)
    outputs = torch.empty_like(v)
    for i, parent in enumerate(parents):
        previous = prefix if parent < 0 else states[parent]
        decayed = previous * alpha[i, :, None, None]
        residual = (v[i] - torch.einsum('hd,hdv->hv', k[i], decayed)) * beta[i, :, None]
        states[i] = decayed + k[i, :, :, None] * residual[:, None, :]
        outputs[i] = torch.einsum('hd,hdv->hv', q[i], states[i])
    return outputs, states


def compact(data, accepted: int):
    parents, ancestry, q, k, v, alpha, beta, prefix = data
    # All dimensions here are bounded fixture sizes; no full state per node.
    cumulative_log = ancestry.to(q.dtype) @ torch.log(alpha)
    cumulative = torch.exp(cumulative_log)
    ratio = torch.exp(cumulative_log.T[:, :, None] - cumulative_log.T[:, None, :])
    strict = ancestry & ~torch.eye(len(parents), device=q.device, dtype=torch.bool)
    gram = torch.einsum('ihd,jhd->hij', k, k)
    matrix = gram * ratio * strict
    matrix = matrix * beta.T[:, :, None]
    matrix.diagonal(dim1=-2, dim2=-1).fill_(1)
    base_k = torch.einsum('nhd,hdv->nhv', k, prefix)
    rhs = beta[..., None] * (v - cumulative[..., None] * base_k)
    factors = torch.linalg.solve_triangular(matrix, rhs.permute(1, 0, 2), upper=False, unitriangular=True)
    qk = torch.einsum('ihd,jhd->hij', q, k)
    weights = qk * ratio * ancestry
    outputs = cumulative[..., None] * torch.einsum('nhd,hdv->nhv', q, prefix)
    outputs += torch.bmm(weights, factors).permute(1, 0, 2)
    accepted_weights = ratio[:, accepted, :] * ancestry[accepted]
    state = cumulative[accepted, :, None, None] * prefix
    state += torch.einsum('nhd,hn,hnv->hdv', k, accepted_weights, factors)
    return outputs, state, factors


def check(device: str, dtype: torch.dtype, nodes: int, heads: int, dim: int, shape: str):
    data = fixture(nodes, heads, dim, device, dtype, shape)
    expected, states = dense(data)
    actual, committed, factors = compact(data, accepted=nodes - 1)
    output_error = (expected - actual).abs().max().item()
    state_error = (states[-1] - committed).abs().max().item()
    tolerance = 1e-10 if dtype == torch.float64 else 3e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(committed, states[-1], atol=tolerance, rtol=tolerance)
    return {'tree': shape, 'nodes': nodes, 'heads': heads, 'dimension': dim,
            'output_max_abs_error': output_error, 'accepted_state_max_abs_error': state_error}


def peak_allocation(function):
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del result
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('branch-state-results.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(2)
    records = [check('cpu', torch.float64, 16, 2, 16, shape) for shape in ('chain', 'star', 'balanced')]
    result = {'cpu_float64': records, 'torch': torch.__version__, 'gpu_validated': False,
              'scope': 'Synthetic GDN recurrence, not a new method or end-to-end model benchmark.'}
    if args.gpu:
        assert torch.cuda.is_available(), 'A real CUDA device is required'
        torch.backends.cuda.matmul.allow_tf32 = False
        result['gpu'] = torch.cuda.get_device_name(0)
        result['sm'] = list(torch.cuda.get_device_capability(0))
        result['cuda_float32'] = [check('cuda', torch.float32, 64, 8, 128, shape)
                                   for shape in ('chain', 'star', 'balanced')]
        data = fixture(64, 8, 128, 'cuda', torch.float32, 'balanced')
        # Warm CUDA library workspaces before allocation measurements.
        warm = compact(data, 63)
        del warm
        dense_peak = peak_allocation(lambda: dense(data))
        compact_peak = peak_allocation(lambda: compact(data, 63))
        result['allocation_probe'] = {
            'dense_peak_additional_bytes': dense_peak, 'compact_peak_additional_bytes': compact_peak,
            'nodes': 64, 'heads': 8, 'dimension': 128, 'dtype': 'float32',
            'limitation': 'PyTorch allocated tensor memory, excluding inputs, cached/reserved memory and driver/library allocations; no throughput claim.',
        }
        result['gpu_validated'] = True
    result['duration_seconds'] = round(time.perf_counter() - started, 3)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
