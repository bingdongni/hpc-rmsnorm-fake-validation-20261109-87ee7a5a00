"""Measure FP8 block conversion on one released Qwen3-4B-FP8 projection."""
import argparse
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer, convert_checkpoint
from compressed_tensors.utils.impl_backend import ImplBackend
from safetensors.torch import load_file, save_file


MODEL = "Qwen/Qwen3-4B-FP8"
REV = "96b30dc13593a244a5e59e84687309f53c375cfa"
SHARD = "model-00001-of-00002.safetensors"
PROJ = "model.layers.0.mlp.up_proj"


def read_range(start, stop):
    url = f"https://huggingface.co/{MODEL}/resolve/{REV}/{SHARD}?range={start}-{stop}"
    req = Request(url, headers={"Range": f"bytes={start}-{stop-1}"})
    with urlopen(req, timeout=60) as response:
        assert response.status == 206
        assert response.headers["Content-Range"].startswith(f"bytes {start}-{stop-1}/")
        data = response.read(stop - start + 1)
    assert len(data) == stop - start
    return data


def load_projection(cache):
    tensors = {}
    header = None
    for name, dtype, shape, digest in [
        ("weight", torch.float8_e4m3fn, (9728, 2560), "be8156e40e25236e16463742b9a3d32f719c8552cbddec3efd802b34f32156a8"),
        ("scale", torch.bfloat16, (76, 20), "6076e9c4205359d9c26800eac4193206cb759f8bf9f58b2dcdddb9df77e326b6"),
    ]:
        key = PROJ + (".weight" if name == "weight" else ".weight_scale_inv")
        path = cache / f"qwen-up-{name}.bin"
        if path.exists():
            data = path.read_bytes()
        else:
            if header is None:
                header_bytes = int.from_bytes(read_range(0, 8), "little")
                base = 8 + header_bytes
                header = json.loads(read_range(8, base))
            assert header[key]["shape"] == list(shape)
            begin, end = header[key]["data_offsets"]
            data = read_range(base + begin, base + end)
        assert hashlib.sha256(data).hexdigest() == digest
        tensors[key] = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
    return tensors


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - baseline) / 2**20
    del out
    times = []
    for _ in range(25):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return dict(peak_allocated_MiB=peak, median_cuda_ms=statistics.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path("."))
    args = parser.parse_args()
    tensors = load_projection(args.cache)
    key = PROJ + ".weight"
    scale_key = PROJ + ".weight_scale_inv"
    weight, scales = tensors[key].cuda(), tensors[scale_key].cuda()
    converter = FP8BlockDequantizer()
    eager = lambda: ImplBackend.call("_create_dequantized_weight", converter, weight, scales)
    dispatch = lambda: converter.process({key: weight, scale_key: scales})[key]
    expected = eager().cpu()
    torch.testing.assert_close(dispatch().cpu(), expected, atol=0, rtol=0)
    eager_stats, gpu_stats = measure(eager), measure(dispatch)
    assert gpu_stats["peak_allocated_MiB"] < eager_stats["peak_allocated_MiB"] * 0.5

    for fp8 in (torch.float8_e4m3fn, torch.float8_e5m2):
        bits = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
        small = bits.view(fp8).reshape(8, 32)
        c = FP8BlockDequantizer(weight_block_size=(4, 8), dtype=torch.float32)
        special = torch.tensor([[1.0, -1.0, 0.0, 0.25], [2.0, 0.0, -0.5, 1.0]], device="cuda")
        for value, scale in ((small, special), (small, special[:1]), (small.t().contiguous().t(), special)):
            actual = c.process({key: value, scale_key: scale})[key]
            ref = ImplBackend.call("_create_dequantized_weight", c, value, scale)
            torch.testing.assert_close(actual, ref, atol=0, rtol=0, equal_nan=True)
            assert torch.equal(actual.signbit()[ref == 0], ref.signbit()[ref == 0])

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src, dst = root / "source", root / "result"
        src.mkdir()
        save_file(tensors, src / "model.safetensors")
        (src / "config.json").write_text('{"model_type":"qwen3","quantization_config":{"quant_method":"fp8"}}')
        convert_checkpoint(src, dst, FP8BlockDequantizer.from_pretrained(src), device="cuda", max_workers=1)
        loaded = load_file(dst / "model.safetensors")
        assert set(loaded) == {key}
        torch.testing.assert_close(loaded[key], expected, atol=0, rtol=0)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, revision=REV,
                          shape=list(weight.shape), values=weight.numel(), equal=True,
                          eager=eager_stats, fused=gpu_stats), sort_keys=True))


if __name__ == "__main__":
    main()
