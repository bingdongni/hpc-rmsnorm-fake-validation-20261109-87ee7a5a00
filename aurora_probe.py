"""Check the grid transform against SciPy on Microsoft's published CAMS static data."""
import hashlib
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import torch

from aurora import Batch, Metadata
from aurora.batch import interpolate_numpy
from tests.v1p5._helpers import _make_batch, _make_small_v1p5, _OUTPUT_ONLY_SURF, _SURF_VARS


def elapsed(fn):
    times = []
    for _ in range(4):
        torch.cuda.synchronize()
        before = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - before) * 1000)
        del result
    return statistics.median(times[1:])


def main(path):
    assert torch.cuda.is_available()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "b1c1d28a887509f688e644a32bd937ae44748f2854b49a3949d60a545b79f379"
    )
    with netCDF4.Dataset(path) as data:
        data.set_auto_mask(False)
        arrays = {name: np.asarray(field[:]) for name, field in data.variables.items()}
    lat, lon = (arrays.pop(key).astype(np.float64) for key in ("latitude", "longitude"))
    fields = {name: torch.from_numpy(x).to("cuda") for name, x in arrays.items()
              if x.shape == (lat.size, lon.size)}
    assert (lat.size, lon.size, len(fields)) == (451, 900, 11), list(arrays)
    # Use two static fields as shaped views to exercise Batch's surface and level layouts.
    batch = Batch(surf_vars={"lsm": fields["lsm"][None, None]},
                  static_vars={name: field for name, field in fields.items() if name not in ("lsm", "z")},
                  atmos_vars={"z": fields["z"][None, None, None]},
                  metadata=Metadata(lat=torch.from_numpy(lat).cuda(), lon=torch.from_numpy(lon).cuda(),
                                    time=(datetime(2023, 1, 1),), atmos_levels=(1000,)))
    lat_out = np.linspace(90, -90, 721)
    lon_out = np.linspace(0, 360, 1440, endpoint=False)

    def roundtrip():
        return {name: torch.from_numpy(interpolate_numpy(
                    field.cpu().double().numpy(), lat, lon, lat_out, lon_out)).float().cuda()
                for name, field in fields.items()}

    def native():
        result = batch.regrid(0.25)
        return {**result.static_vars, "lsm": result.surf_vars["lsm"][0, 0],
                "z": result.atmos_vars["z"][0, 0, 0]}

    expected = roundtrip()
    output = native()
    max_error = {}
    for name in fields:
        a, b = output[name], expected[name]
        assert a.is_cuda and a.dtype == torch.float32 and a.shape == (721, 1440)
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)
        max_error[name] = (a - b).abs().max().item()
    del expected, output
    cpu_roundtrip_ms, native_ms = elapsed(roundtrip), elapsed(native)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                          points_per_field=721 * 1440, fields=len(fields), max_abs=max_error,
                          cpu_roundtrip_ms=cpu_roundtrip_ms, native_cuda_ms=native_ms)), flush=True)

    torch.manual_seed(73)
    model = _make_small_v1p5().eval().cuda()
    example = _make_batch(surf_vars=tuple(v for v in _SURF_VARS if v not in _OUTPUT_ONLY_SURF)).to("cuda")
    source = example.surf_vars["2t"].requires_grad_()
    prediction = model(example, lead_times=torch.tensor([6.0], device="cuda"))
    obs = prediction.regrid(22.5).surf_vars["2t"]
    grad, = torch.autograd.grad(obs.mean(), source)
    assert obs.is_cuda and grad.is_cuda and torch.isfinite(grad).all() and grad.abs().max() > 0
    print(json.dumps(dict(model=type(model).__name__, initialized="random", output=list(obs.shape),
                          input_gradient_max=grad.abs().max().item())), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
