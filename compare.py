"""Run the public entry point against one exact source checkout."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import warnings

import torch

name, root, device, revision = sys.argv[1:]
root = Path(root).resolve()
sys.path.insert(0, str(root))
baseline = revision == "base"
assert device != "cuda" or torch.cuda.is_available()
torch.manual_seed(311)

if name in ("pot", "autocast"):
    # The solver is pure Python. Linux builds the adjacent native modules from
    # this checkout. Windows reuses only these three extensions from the wheel.
    source = Path(importlib.metadata.distribution("POT").locate_file("ot"))
    for subdir, module in [("bsp", "bsp_wrap"), ("lp", "emd_wrap"), ("partial", "partial_cython")]:
        target = root / "ot" / subdir
        if not any(target.glob(module + "*.pyd")) and not any(target.glob(module + "*.so")):
            candidates = []
            for native in [source / subdir, Path("pot/ot") / subdir]:
                candidates += list(native.glob(module + "*.pyd")) + list(native.glob(module + "*.so"))
            assert candidates, module
            shutil.copy2(candidates[0], target)
    import ot
    assert Path(ot.__file__).resolve().is_relative_to(root)
    if name == "autocast":
        xs = 200 + torch.arange(14, dtype=torch.float32, device=device).reshape(7, 2) / 8
        xt = xs[:5] - 0.25
        opts = dict(reg=2, isLazy=True, batchSize=3, stopThr=1e-5, numIterMax=100, warn=False)
        expected = ot.bregman.empirical_sinkhorn2(xs, xt, **opts)
        with torch.autocast(device):
            actual = ot.bregman.empirical_sinkhorn2(xs, xt, **opts)
        finite = bool(torch.isfinite(actual))
        if revision == "first":
            assert not torch.allclose(actual, expected)
        else:
            torch.testing.assert_close(actual, expected)
        print("Compared " + json.dumps(dict(project=name, revision=revision, device=device,
              finite=finite, expected=expected.item(), actual=actual.item() if finite else None)))
        sys.exit(0)
    x = torch.randn(257, 16, dtype=torch.float64).to(device)
    y = torch.randn(129, 16, dtype=torch.float64).to(device)
    nx = ot.backend.get_backend(x)
    from_numpy, to_numpy = nx.from_numpy, nx.to_numpy
    copied = {"coordinate_arrays": 0, "cost_blocks": 0, "cost_bytes": 0}

    def send(*arrays, **kwargs):
        for a in arrays:
            if a.ndim == 2:
                copied["cost_blocks"] += 1
                copied["cost_bytes"] += a.nbytes
        return from_numpy(*arrays, **kwargs)

    def receive(*arrays, **kwargs):
        copied["coordinate_arrays"] += sum(a.ndim == 2 for a in arrays)
        return to_numpy(*arrays, **kwargs)

    expected = ot.bregman.empirical_sinkhorn2(x.cpu().numpy(), y.cpu().numpy(), 10)
    nx.from_numpy, nx.to_numpy = send, receive
    result = ot.solve_sample(x, y, reg=10, lazy=True, batch_size=32)
    torch.testing.assert_close(result.value_linear, x.new_tensor(expected), rtol=1e-6, atol=1e-8)
    if baseline:
        assert copied["coordinate_arrays"] > 0 and copied["cost_blocks"] > 0
    else:
        assert copied == {"coordinate_arrays": 0, "cost_blocks": 0, "cost_bytes": 0}
    details = dict(value=result.value_linear.item(), **copied)
elif name == "tslearn":
    import tslearn
    from tslearn.metrics import SoftDTWLossPyTorch
    assert Path(tslearn.__file__).resolve().is_relative_to(root)
    x = torch.randn(3, 5, 2, dtype=torch.float64, device=device)
    y = torch.randn(3, 6, 2, dtype=torch.float64, device=device)
    lengths = [(2, 4), (5, 3), (2, 4)]
    for i, (m, n) in enumerate(lengths):
        x[i, m:] = torch.nan
        y[i, n:] = torch.nan
    x.requires_grad_()
    y.requires_grad_()
    loss = SoftDTWLossPyTorch(normalize=True)
    ref = torch.cat([loss(x[i:i+1, :m], y[i:i+1, :n]) for i, (m, n) in enumerate(lengths)])
    try:
        actual = loss(x, y, x_lengths=[m for m, n in lengths], y_lengths=[n for m, n in lengths])
    except TypeError as exc:
        assert baseline and "x_lengths" in str(exc)
        details = {"unsupported_keyword": "x_lengths"}
    else:
        assert not baseline
        torch.testing.assert_close(actual, ref)
        grads = torch.autograd.grad(actual.sum(), (x, y))
        expected = torch.autograd.grad(ref.sum(), (x, y))
        for g, want in zip(grads, expected):
            torch.testing.assert_close(g, want)
        details = {"max_value_error": (actual-ref).abs().max().item(),
                   "max_gradient_error": max((g-w).abs().max().item() for g,w in zip(grads, expected))}
elif name == "diffeq":
    import torchdiffeq
    assert Path(torchdiffeq.__file__).resolve().is_relative_to(root)
    y = torch.tensor([1., 2.], dtype=torch.float64, device=device, requires_grad=True)
    integral = y.new_zeros((), requires_grad=True)
    rate = y.new_tensor(0.3, requires_grad=True)
    times = y.new_tensor([0., 0.23, 0.74, 1.])
    def rhs(t, state):
        return rate * state[0], state[0].sum()
    simple = torchdiffeq.odeint_dense(lambda t, v: rate*v, y, times[0], times[-1])
    torch.testing.assert_close(simple(times[1].cpu()), y*torch.exp(rate*times[1]))
    try:
        dense = torchdiffeq.odeint_dense(rhs, (y, integral), times[0], times[-1])
    except AssertionError:
        assert baseline
        details = {"tuple_rejected": True}
    else:
        assert not baseline
        ref = torchdiffeq.odeint(rhs, (y, integral), times)
        out = dense(times[2].cpu())
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b[2])
        params = (y, integral, rate)
        gradients = torch.autograd.grad(out[0].sum() + out[1], params)
        expected = torch.autograd.grad(ref[0][2].sum() + ref[1][2], params)
        for a, b in zip(gradients, expected):
            torch.testing.assert_close(a, b)
        details = {"max_gradient_error": max((a-b).abs().max().item() for a,b in zip(gradients, expected))}
else:
    raise ValueError(name)
if device == "cuda":
    torch.cuda.synchronize()
print("Compared " + json.dumps(dict(project=name, revision=revision, device=device,
      hardware=torch.cuda.get_device_name() if device == 'cuda' else 'CPU', **details)))
