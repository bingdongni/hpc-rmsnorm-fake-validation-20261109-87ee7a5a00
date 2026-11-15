"""Compare candidate SSM reads, branch selection and committing the accepted token."""
import json
import statistics

import torch
from mamba_ssm.ops.triton.selective_state_update import selective_state_update as update


def measure(fn):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / (2**20)
    del result
    timings = []
    for _ in range(40):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return dict(peak_allocated_MiB=peak, median_cuda_ms=statistics.median(timings))


def main():
    torch.manual_seed(519)
    parents, fanout, heads, dim, dstate = 16, 4, 8, 64, 128
    size = parents * fanout
    dev = dict(device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(parents, heads, dim, dstate, **dev)
    unchanged = cache.clone()
    slots = torch.arange(parents, device="cuda").repeat_interleave(fanout)
    slot_table = torch.stack((slots.flip(0), slots), dim=1).int()
    candidates = dict(x=torch.randn(size, heads, dim, **dev),
                      dt=torch.randn(size, heads, 1, **dev).expand(-1, -1, dim),
                      B=torch.randn(size, 2, dstate, **dev), C=torch.randn(size, 2, dstate, **dev),
                      z=torch.randn(size, heads, dim, **dev))
    fixed = dict(A=-torch.rand(heads, 1, 1, device="cuda").expand(-1, dim, dstate),
                 D=torch.randn(heads, 1, **dev).expand(-1, dim),
                 dt_bias=torch.rand(heads, 1, **dev).expand(-1, dim), dt_softplus=True)

    def private():
        return update(cache.index_select(0, slots), **candidates, **fixed)

    def shared():
        return update(cache, **candidates, **fixed, state_batch_indices=slot_table[:, 1],
                      disable_state_update=True)

    private_states = cache.index_select(0, slots)
    expected = update(private_states, **candidates, **fixed)
    actual = shared()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(cache, unchanged)
    private_stats, shared_stats = measure(private), measure(shared)
    assert private_stats["peak_allocated_MiB"] - shared_stats["peak_allocated_MiB"] >= 7.9

    # Commit one candidate per parent, then decode the next step on both state pools.
    chosen = actual.float().sum((-1, -2)).reshape(parents, fanout).argmax(-1)
    chosen += torch.arange(parents, device="cuda") * fanout
    accepted = {k: v[chosen] for k, v in candidates.items()}
    expected_cache = private_states[chosen].clone()
    accepted_out = update(cache, **accepted, **fixed)
    torch.testing.assert_close(accepted_out, expected[chosen], rtol=0, atol=0)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
    out = update(cache, **accepted, **fixed)
    torch.testing.assert_close(out, update(expected_cache, **accepted, **fixed), rtol=0, atol=0)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, parents=parents,
                          fanout=fanout, heads=heads, dim=dim, dstate=dstate, dtype="bfloat16",
                          private_copy=private_stats, shared_read=shared_stats, exact=True)))


if __name__ == "__main__":
    main()
