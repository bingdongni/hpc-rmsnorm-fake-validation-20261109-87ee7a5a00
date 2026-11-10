"""Run trajectory loss tests with the unavailable verl masking helper stubbed."""
import os
import ast
from pathlib import Path
import sys
import types
from typing import Optional

import torch

root = Path(os.environ.get("AGENTR1_PROBE_ROOT", r"C:\PR\agentr1-denominator-scout-20261110"))
sys.path.insert(0, str(root))
verl = types.ModuleType("verl")
utils = types.ModuleType("verl.utils")
functional = types.ModuleType("verl.utils.torch_functional")
functional_path = Path(os.environ.get("AGENTR1_VERL_FUNCTIONAL", r"C:\PR\verl070_torch_functional.py"))
functional_tree = ast.parse(functional_path.read_text(encoding="utf-8"), filename=str(functional_path))
functional_node = next(node for node in functional_tree.body if isinstance(node, ast.FunctionDef) and node.name == "masked_sum")
functional_scope = {"torch": torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[functional_node], type_ignores=[])),
             str(functional_path), "exec"), functional_scope)
functional.masked_sum = functional_scope["masked_sum"]
verl.utils = utils
utils.torch_functional = functional
sys.modules.update({"verl": verl, "verl.utils": utils, "verl.utils.torch_functional": functional})
upstream_path = Path(os.environ.get("AGENTR1_VERL_SOURCE", r"C:\PR\verl070_core_algos.py"))
upstream_tree = ast.parse(upstream_path.read_text(encoding="utf-8"), filename=str(upstream_path))
upstream_agg = next(node for node in upstream_tree.body if isinstance(node, ast.FunctionDef) and node.name == "agg_loss")
scope = {"torch": torch, "Optional": Optional, "verl_F": functional}
exec(compile(ast.fix_missing_locations(ast.Module(body=[upstream_agg], type_ignores=[])), str(upstream_path), "exec"), scope)
trainer = types.ModuleType("verl.trainer")
ppo = types.ModuleType("verl.trainer.ppo")
upstream = types.ModuleType("verl.trainer.ppo.core_algos")
upstream.agg_loss = scope["agg_loss"]
verl.trainer = trainer
trainer.ppo = ppo
ppo.core_algos = upstream
sys.modules.update({"verl.trainer": trainer, "verl.trainer.ppo": ppo, "verl.trainer.ppo.core_algos": upstream})

import pytest

raise SystemExit(pytest.main([str(root / "tests/test_trajectory_loss_normalization.py"), "-q", "--disable-warnings"]))
