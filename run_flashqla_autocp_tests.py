"""Run architecture-neutral AutoCP tests without importing unsupported local GPU kernels."""
import os
from pathlib import Path
import sys
import types

import pytest

root = Path(os.environ['FLASHQLA_TEST_ROOT'])
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

chunk = sys.modules['flash_qla.ops.gated_delta_rule.chunk']
chunk.CHUNK_SIZE = 32
chunk.chunk_gated_delta_rule_fwd = lambda *args: None
chunk.chunk_gated_delta_rule_bwd = lambda *args: None
chunk._auto_intra_cp_context = lambda *args: None
utility = types.ModuleType('flash_qla.utils')
utility.tensor_cache = lambda fn: fn
utility.l2norm = lambda *args: None
utility.pack = lambda *args: None
sys.modules['flash_qla.utils'] = utility
reference = types.ModuleType('ref_gdr')
reference.chunk_gated_delta_rule_fwd = lambda *args: None
reference.chunk_gated_delta_rule_bwd = lambda *args: None
sys.modules['ref_gdr'] = reference

tilelang = types.ModuleType('tilelang')
contrib = types.ModuleType('tilelang.contrib')
nvcc = types.ModuleType('tilelang.contrib.nvcc')
nvcc.get_target_compute_version = lambda: '12.1'
tilelang.contrib = contrib
contrib.nvcc = nvcc
sys.modules.update({'tilelang': tilelang, 'tilelang.contrib': contrib, 'tilelang.contrib.nvcc': nvcc})
sys.path.insert(0, str(root / 'tests'))
raise SystemExit(pytest.main([str(root / 'tests/test_autocp_arch.py'), *sys.argv[1:]]))
