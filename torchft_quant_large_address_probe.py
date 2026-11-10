"""Exercise address arithmetic with large, sparsely touched CUDA allocations."""

import gc
import importlib.util
import os
from pathlib import Path
import sys

import torch


root = Path(os.environ.get("TORCHFT_PROBE_ROOT", r"C:\PR\torchft-scout-20261110"))
path = root / "torchft/quantization.py"
spec = importlib.util.spec_from_file_location("torchft_quantization_large_probe", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

assert torch.cuda.is_available()
free, total = torch.cuda.mem_get_info()
print(f"CUDA free/total: {free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
assert free > 5 * 2**30, "Insufficient free GPU memory for the large-stride test"

# A logical huge input is enough to test host-side offset preparation.
large = torch.ones((1, 1), dtype=torch.float16, device="cuda").expand(1_048_576, 2048)
small = torch.ones((1, 1), dtype=torch.float16, device="cuda")
metadata = module._prepare_quantize_fp8([large, small], 1)
expected_offset = 1_048_576 * 2052
assert metadata[3].dtype == torch.int64
assert metadata[3].cpu().tolist() == [0, expected_offset]
print(f"Prepared large tensor offset: {expected_offset} bytes", flush=True)
del large, small, metadata

# Exercise quantize, reduce and dequantize across the 2 GiB address boundary.
# Three ranks ensure rank * per_rank overflows signed int32 even though each
# individual argument fits in int32. Only three short rows are touched.
stride = 2**30 + 16
buffer = torch.empty(2 * stride + 8, dtype=torch.uint8, device="cuda")
inputs = torch.tensor([[1.0], [3.0], [5.0]], dtype=torch.float32, device="cuda")
outputs = torch.empty_like(inputs)
prepared = module._prepare_quantize_fp8([inputs], 3)
prepared = list(prepared)
prepared[1:4] = [item.to(torch.int64) for item in prepared[1:4]]
grid = (3, 1)
module._fused_kernel_quantize_into_fp8[grid](
    *prepared[:5], buffer, stride, 3, BLOCK_SIZE=module.BLOCK_SIZE_T,
    TL_FP8_TYPE=module._get_fp8_type(), TL_FP8_MAX=module._get_fp8_max(), WIDE_INDEX=True,
)
prepared_out = module._prepare_quantize_fp8([outputs], 3)
prepared_out = list(prepared_out)
prepared_out[1:4] = [item.to(torch.int64) for item in prepared_out[1:4]]
module._fused_kernel_dequantize_from_fp8[grid](
    *prepared_out[:5], buffer, stride, 3, BLOCK_SIZE=module.BLOCK_SIZE_T,
    TL_FP8_TYPE=module._get_fp8_type(), WIDE_INDEX=True,
)
torch.testing.assert_close(outputs, inputs, rtol=0.02, atol=0.02)
module._fused_kernel_reduce_fp8[(1, 1)](
    prepared[1], prepared[3], buffer, stride, 3, 2, 3.0,
    BLOCK_SIZE=module.BLOCK_SIZE_T, TL_FP8_TYPE=module._get_fp8_type(),
    TL_FP8_MAX=module._get_fp8_max(), WIDE_INDEX=True,
)
module._fused_kernel_dequantize_from_fp8[grid](
    *prepared_out[:5], buffer, stride, 3, BLOCK_SIZE=module.BLOCK_SIZE_T,
    TL_FP8_TYPE=module._get_fp8_type(), WIDE_INDEX=True,
)
torch.cuda.synchronize()
torch.testing.assert_close(outputs[2], inputs.mean(dim=0), rtol=0.02, atol=0.04)
print("Quantize, rank-2 reduction and dequantize crossed 2 GiB correctly", flush=True)

# Exercise a second input whose encoded-data offset is beyond 2 GiB.
a = torch.tensor([[2.0]], dtype=torch.float32, device="cuda")
b = torch.tensor([[4.0]], dtype=torch.float32, device="cuda")
combined = module._prepare_quantize_fp8([a, b], 1)
combined = list(combined)
combined[1:4] = [item.to(torch.int64) for item in combined[1:4]]
combined[3][1] = 2 * stride
module._fused_kernel_quantize_into_fp8[(1, 2)](
    *combined[:5], buffer, 2 * stride + 8, 1,
    BLOCK_SIZE=module.BLOCK_SIZE_T, TL_FP8_TYPE=module._get_fp8_type(),
    TL_FP8_MAX=module._get_fp8_max(), WIDE_INDEX=True,
)
a_out, b_out = torch.empty_like(a), torch.empty_like(b)
combined_out = module._prepare_quantize_fp8([a_out, b_out], 1)
combined_out = list(combined_out)
combined_out[1:4] = [item.to(torch.int64) for item in combined_out[1:4]]
combined_out[3][1] = 2 * stride
module._fused_kernel_dequantize_from_fp8[(1, 2)](
    *combined_out[:5], buffer, 2 * stride + 8, 1,
    BLOCK_SIZE=module.BLOCK_SIZE_T, TL_FP8_TYPE=module._get_fp8_type(), WIDE_INDEX=True,
)
torch.testing.assert_close(a_out, a, rtol=0.02, atol=0.02)
torch.testing.assert_close(b_out, b, rtol=0.02, atol=0.02)
print("Quantize and dequantize handled a second-input offset above 2 GiB", flush=True)
del buffer, inputs, outputs, prepared, prepared_out, combined, combined_out
gc.collect()
torch.cuda.empty_cache()

# A two-row view with a >2^31 element row stride needs about 4 GiB, but the
# quantization kernel touches only the two populated elements.
element_stride = 2**31 + 17
backing = torch.empty(element_stride + 1, dtype=torch.float16, device="cuda")
view = torch.as_strided(backing, (2, 1), (element_stride, 1))
view[0, 0] = 2.0
view[1, 0] = 7.0
encoded = module.fused_quantize_into_fp8([view], 1)
view[0, 0] = 0.0
view[1, 0] = 0.0
module.fused_dequantize_from_fp8([view], encoded, 1)
torch.cuda.synchronize()
torch.testing.assert_close(view.cpu(), torch.tensor([[2.0], [7.0]], dtype=torch.float16), rtol=0.02, atol=0.02)
print("Quantize and dequantize handled an input stride above 2^31 elements", flush=True)
