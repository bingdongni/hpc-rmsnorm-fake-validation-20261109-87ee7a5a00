import os
import statistics
import time

os.environ['LOGURU_LEVEL'] = 'ERROR'
import torch

from checkpoint_engine.pin_memory import _normal_pin_memory

torch.set_num_threads(1)
print('device', torch.cuda.get_device_name(0), 'torch', torch.__version__, 'cuda', torch.version.cuda, flush=True)
count, dimension = 8, 2048
base = torch.arange(count * dimension * dimension, device='cuda', dtype=torch.float32).view(count, dimension, dimension)
for layout in ('contiguous', 'transposed'):
    sources = {f'w{i}': (base[i] if layout == 'contiguous' else base[i].t()) for i in range(count)}
    buffers = _normal_pin_memory(files=[], named_tensors=sources)
    elapsed = []
    peaks = []
    for _ in range(7):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        _normal_pin_memory(files=[], named_tensors=sources, shared_pin_memory=buffers)
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start_time) * 1000)
        peaks.append((torch.cuda.max_memory_allocated() - start_allocated) / 2**20)
    print(layout, 'median_ms', round(statistics.median(elapsed), 2), 'samples_ms', [round(x, 2) for x in elapsed],
          'peak_extra_MiB', max(peaks), 'samples_MiB', peaks, flush=True)
    if layout == 'transposed' and os.environ.get('EXPECTED_STAGING') == 'baseline':
        assert max(peaks) > 16
    if layout == 'transposed' and os.environ.get('EXPECTED_STAGING') == 'bounded':
        assert max(peaks) <= 16
