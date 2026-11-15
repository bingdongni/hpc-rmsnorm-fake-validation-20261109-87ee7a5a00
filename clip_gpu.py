import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from statistics import median
import sys

from PIL import Image
from safetensors.torch import load_file
import torch


parser = argparse.ArgumentParser()
for arg in ("candidate", "baseline_layer", "weights", "image"):
    parser.add_argument(arg, type=Path)
parser.add_argument("--wide", type=Path)
args = parser.parse_args()
assets = [
    (args.weights, "95feb653fcddf239530c87f2b26bf5306fd7c3c20d8a1818d4e826099f12d677"),
    (args.image, "f3f87bb8ab3c26c7ecfd3ac60421d7f32b0503d1d6c5baf8bac42ed93d86351a"),
]
if args.wide:
    assets.append((args.wide, "3dd4009db8a4497039172dd935e92c881f73b2214f9bbf7c09bfb724c8ea4a52"))
for path, digest in assets:
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == digest

sys.path.insert(0, str(args.candidate.resolve()))
import timm
from timm.layers import attention_pool2d as candidate

spec = importlib.util.spec_from_file_location("timm.layers._upstream_pool_reference", args.baseline_layer)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
assert torch.cuda.is_available()
print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "model": "resnet50_clip.openai"}), flush=True)


def differentiation():
    count = 0
    torch.manual_seed(42)
    for cls in ("AttentionPool2d", "RotAttentionPool2d"):
        for separate in (True, False):
            for fused in (True, False):
                for pool, training, dropout, pre_logits in (
                    ("token", True, 0., False), ("token", False, .2, True),
                    ("token", True, .2, False), ("", False, 0., False),
                ):
                    kwargs = dict(num_heads=4, pool_type=pool, class_token=True, qkv_separate=separate, drop_rate=dropout)
                    old, new = (getattr(module, cls)(64, **kwargs).cuda().train(training) for module in (baseline, candidate))
                    new.load_state_dict(old.state_dict())
                    old.fused_attn = new.fused_attn = fused
                    image = torch.randn(2, 64, 5, 7, device="cuda", requires_grad=True)
                    result = []
                    for layer in (old, new):
                        torch.cuda.manual_seed(123)
                        value = layer(image, pre_logits=pre_logits)
                        grads = torch.autograd.grad(value.square().sum(), (image, *layer.parameters()), allow_unused=True)
                        result.append((value, grads, torch.cuda.get_rng_state()))
                    torch.testing.assert_close(result[0][0], result[1][0], atol=2e-5, rtol=2e-5)
                    for a, b in zip(result[0][1], result[1][1]):
                        torch.testing.assert_close(a, b, atol=4e-5, rtol=4e-5)
                    torch.testing.assert_close(result[0][2], result[1][2])
                    count += 1
    print(json.dumps({"gradient_cases": count, "dropout_rng": "identical"}), flush=True)


def timed(call, count):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def compare(model, call, stage, batch, precision):
    outputs = []
    memory = []
    layers = [baseline.AttentionPool2d, candidate.AttentionPool2d]
    for cls in layers:
        model.head.__class__ = cls
        outputs.append(call().float())
        for _ in range(8):
            call()
        torch.cuda.synchronize()
        active = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        memory.append((torch.cuda.max_memory_allocated() - active) / 2**20)
    delta = (outputs[0] - outputs[1]).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(*outputs).min().item()
    torch.testing.assert_close(*outputs, rtol=.01 if precision == "float16" else 2e-5,
                               atol=.001 if precision == "float16" else 1e-5)
    assert cos > .99999
    timings = [[], []]
    rng = random.Random(731)
    for _ in range(11):
        order = [0, 1]
        rng.shuffle(order)
        for i in order:
            model.head.__class__ = layers[i]
            timings[i].append(timed(call, 10 if stage == "head" else 5))
    print(json.dumps(dict(clip=stage, arch=model.pretrained_cfg["architecture"], batch=batch, dtype=precision,
                          baseline_MiB=round(memory[0], 3), candidate_MiB=round(memory[1], 3), baseline_ms=round(median(timings[0]), 4),
                          candidate_ms=round(median(timings[1]), 4), max_abs=delta, min_cosine=cos)), flush=True)


@torch.inference_mode()
def checkpoint(name, weights, batches, precisions):
    model = timm.create_model(name)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.eval().cuda()
    transform = timm.data.create_transform(**timm.data.resolve_model_data_config(model), is_training=False)
    with Image.open(args.image) as sample:
        image = transform(sample.convert("RGB")).unsqueeze(0).cuda()
    assert tuple(image.shape[1:]) == model.pretrained_cfg["input_size"]
    for precision in precisions:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=precision == "float16", cache_enabled=True):
            for batch in batches:
                images = image.repeat(batch, 1, 1, 1)
                features = model.forward_features(images)
                assert tuple(features.shape[-2:]) == tuple(model.head.feat_size)
                compare(model, lambda: model.forward_head(features), "head", batch, precision)
                compare(model, lambda: model(images), "image_encoder", batch, precision)


differentiation()
checkpoint("resnet50_clip.openai", args.weights, (1, 8), ("float32", "float16"))
if args.wide:
    checkpoint("resnet50x16_clip.openai", args.wide, (1, 4), ("float16",))
