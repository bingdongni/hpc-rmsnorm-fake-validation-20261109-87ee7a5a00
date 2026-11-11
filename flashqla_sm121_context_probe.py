"""Exercise the real AutoCP/context code on CPU with a simulated SM121 device."""
import os
from pathlib import Path
import sys
import types

import torch

root = Path(os.environ['FLASHQLA_TEST_ROOT'])
device = os.environ.get('FLASHQLA_TEST_DEVICE', 'cpu')
if device == 'cuda':
    assert torch.cuda.is_available()
    torch.cuda.init()
    print('Physical GPU:', torch.cuda.get_device_name())
for name, path in (
    ('flash_qla', root / 'flash_qla'),
    ('flash_qla.ops', root / 'flash_qla/ops'),
    ('flash_qla.ops.gated_delta_rule', root / 'flash_qla/ops/gated_delta_rule'),
    ('flash_qla.ops.gated_delta_rule.chunk', root / 'flash_qla/ops/gated_delta_rule/chunk'),
    ('flash_qla.ops.gated_delta_rule.chunk.cp', root / 'flash_qla/ops/gated_delta_rule/chunk/cp'),
):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package
utility = types.ModuleType('flash_qla.utils')
utility.tensor_cache = lambda fn: fn
sys.modules['flash_qla.utils'] = utility
tilelang = types.ModuleType('tilelang')
tilelang.cdiv = lambda value, divisor: (value + divisor - 1) // divisor
tilelang.contrib = types.SimpleNamespace(nvcc=types.SimpleNamespace(get_target_compute_version=lambda: '12.1'))
sys.modules['tilelang'] = tilelang
torch.cuda.get_device_properties = lambda *args, **kwargs: types.SimpleNamespace(multi_processor_count=48)
sys.path.insert(0, str(root / 'tests'))

from flash_qla.ops.gated_delta_rule.chunk.cp import context
from ref_context import ref_intra_context

assert context.ARCH == 'sm121' and context.AUTOCP_MODEL
for is_train in (False, True):
    boundaries = [0, 8192, 16384]
    raw = torch.tensor(boundaries, dtype=torch.int32, device=device)
    result = context._calc_intra_cp_seqs(raw, 32, 8, is_train=is_train, force_intra_cp=True)
    actual = result.intra_cp_cu_seqlens.tolist()
    max_local = (actual[1] - actual[0]) // 32
    expected = ref_intra_context(boundaries, 32, max_local)
    assert actual == expected['cp_cu']
    assert result.seq_map_r2c.tolist() == expected['r2c']
    assert result.seq_map_c2r.tolist() == expected['c2r']
    assert result.ht_mask.tolist() == expected['ht']
    assert result.ht_mask_bwd.tolist() == expected['ht_bwd']
    print('SM121 simulated context:', 'train' if is_train else 'inference', 'reference matched')
