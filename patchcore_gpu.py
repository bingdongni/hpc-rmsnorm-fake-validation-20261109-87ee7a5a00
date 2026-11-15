"""Compare the actual base and candidate PatchCore search on the same CUDA tensors."""
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch

base, candidate = (Path(p).resolve() for p in sys.argv[1:])
sys.path.insert(0, str(candidate / "src"))
from anomalib.models.image.patchcore import torch_model

assert Path(torch_model.__file__).resolve().is_relative_to(candidate)
spec = importlib.util.spec_from_file_location(
    "anomalib.models.image.patchcore.base_for_benchmark",
    base / "src/anomalib/models/image/patchcore/torch_model.py",
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def measure(model, queries, k):
    model.nearest_neighbors(queries, k)
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    times = []
    for _ in range(5):
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        start.record()
        scores, locations = model.nearest_neighbors(queries, k)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated() - allocated
        result = scores.cpu(), locations.cpu()
        del scores, locations
    return result, round(peak / 2**20, 2), round(statistics.median(times), 2)


torch.manual_seed(901)
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
base_model, model = [
    cls(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=False, num_neighbors=3).eval().cuda()
    for cls in (old.PatchcoreModel, torch_model.PatchcoreModel)
]
model.load_state_dict(base_model.state_dict())
print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}), flush=True)
with torch.inference_mode():
    for n, b, d, k in [(64, 2048, 384, 1), (1280, 65536, 384, 1), (256, 131072, 384, 7)]:
        bank = torch.randn(b, d, device="cuda")
        q = torch.randn(n, d, device="cuda")
        model.memory_bank = base_model.memory_bank = bank
        # Use CUDA events for timings and report peak allocated bytes above the resident inputs.
        (ref, indices), base_mem, base_ms = measure(base_model, q, k)
        (out, locations), mem, ms = measure(model, q, k)
        torch.testing.assert_close(out, ref, rtol=2e-6, atol=2e-6)
        # torch.topk can permute tied distances. Check each reported neighbour geometrically as well.
        ids = locations.to(q.device)
        selected = bank[ids]
        actual = (q - selected if k == 1 else q[:, None] - selected).norm(dim=-1).cpu()
        torch.testing.assert_close(actual, out, rtol=2e-6, atol=2e-6)
        if b >= 65536:
            assert mem < base_mem * 0.60, (mem, base_mem)
        print(json.dumps({"patchcore": [n, b, d, k], "base_mib": base_mem, "pr_mib": mem,
                          "base_ms": base_ms, "pr_ms": ms, "different_indices": (indices != locations).sum().item(),
                          "max_score_delta": (ref - out).abs().max().item()}), flush=True)
    images = torch.rand(2, 3, 64, 64, device="cuda")
    model.memory_bank = base_model.memory_bank = torch.randn(32768, 384, device="cuda")
    a, b = base_model(images), model(images)
    torch.testing.assert_close(a.pred_score, b.pred_score)
    torch.testing.assert_close(a.anomaly_map, b.anomaly_map)
    print("Full CUDA ResNet18 PatchCore scores and anomaly maps match.", flush=True)
