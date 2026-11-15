"""Compare two ways to train with an already resident dense teacher log-probability cache."""

import json
from statistics import median

import torch
import triton

from liger_kernel.transformers.functional import liger_fused_linear_kl_div


def measure():
    tokens, hidden, vocab = 2048, 256, 32768
    student = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    head = torch.randn(vocab, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cache = torch.randn(tokens, vocab, device="cuda", dtype=torch.float32).log_softmax(-1)

    def step(direct):
        target = cache if direct else cache.exp()
        loss = liger_fused_linear_kl_div(student, head, target, log_target=direct)
        return loss.detach(), *torch.autograd.grad(loss, (student, head))

    ref = step(False)
    actual = step(True)
    for a, b in zip(actual, ref):
        torch.testing.assert_close(a, b, rtol=0.01, atol=2e-5)
    del actual, ref

    samples = {False: [], True: []}
    for repetition in range(12):
        for direct in ((False, True) if repetition % 2 == 0 else (True, False)):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start.record()
            outputs = step(direct)
            end.record()
            end.synchronize()
            peak = torch.cuda.max_memory_allocated() - resident
            if repetition >= 2:
                samples[direct].append((peak / (1024**2), start.elapsed_time(end)))
            del outputs
    results = {}
    for direct, values in samples.items():
        results["direct_log_target" if direct else "materialized_probabilities"] = {
            "peak_additional_MiB": median(p for p, _ in values),
            "median_ms": round(median(t for _, t in values), 3),
        }
    return {"tokens": tokens, "hidden": hidden, "vocab": vocab, "timed_steps_each": 10, **results}


def train():
    x = torch.randn(64, 16, device="cuda")
    teacher = torch.randn(257, 16, device="cuda") / 4
    cached = (x @ teacher.T).log_softmax(-1)
    w_log = torch.zeros_like(teacher, requires_grad=True)
    w_ref = torch.zeros_like(teacher, requires_grad=True)
    start = None
    for _ in range(30):
        loss = liger_fused_linear_kl_div(x, w_log, cached, log_target=True)
        expected = torch.nn.functional.kl_div((x @ w_ref.T).log_softmax(-1), cached, log_target=True, reduction="batchmean")
        torch.testing.assert_close(loss, expected, atol=2e-6, rtol=1e-5)
        if start is None:
            start = float(loss.detach())
        with torch.no_grad():
            w_log.sub_(torch.autograd.grad(loss, w_log)[0], alpha=2)
            w_ref.sub_(torch.autograd.grad(expected, w_ref)[0], alpha=2)
    torch.testing.assert_close(w_log, w_ref, atol=1e-6, rtol=1e-5)
    end = float(loss.detach())
    assert end < start
    return {"steps": 30, "first_loss": start, "last_loss": end, "max_weight_diff": float((w_log - w_ref).detach().abs().max())}


if __name__ == "__main__":
    torch.manual_seed(221)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "triton": triton.__version__,
                      "comparison": measure(), "small_distillation_run": train()}, indent=2))
