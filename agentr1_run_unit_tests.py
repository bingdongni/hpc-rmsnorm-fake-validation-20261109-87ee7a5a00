"""Run focused unit tests with unavailable Ray/verl integration modules stubbed."""
import os
from pathlib import Path
import sys
import types

root = Path(os.environ.get("AGENTR1_PROBE_ROOT", r"C:\PR\agentr1-scout-20261110"))
sys.path.insert(0, str(root))
sys.path.insert(0, r"C:\PR\fla-model-deps-20261110")
sys.path.insert(0, r"C:\PR\unirl-research-deps-20261110")


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    sys.modules[name] = result
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, result)
    return result


def noop(*_args, **_kwargs):
    return None


class TensorDict(dict):
    def __init__(self, data, batch_size=None):
        super().__init__(data)
        self.batch_size = batch_size


class DataProto:
    def __init__(self, batch, non_tensor_batch, meta_info):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = meta_info


flow_pkg = module("agent_r1.agent_flow")
flow_pkg.__path__ = [str(root / "agent_r1/agent_flow")]
module("hydra", utils=types.SimpleNamespace(instantiate=noop))
ray = module("ray", remote=lambda cls: cls)
ray.actor = types.SimpleNamespace(ActorHandle=object)
module("omegaconf", DictConfig=object, OmegaConf=object)
module("tensordict", TensorDict=TensorDict)
module("transformers", AutoProcessor=object, AutoTokenizer=object)
module("agent_r1.reward_loop.reward_loop", RewardLoopWorker=object)
for name in ("verl", "verl.experimental", "verl.experimental.agent_loop", "verl.single_controller",
             "verl.single_controller.ray", "verl.utils", "verl.utils.dataset", "verl.workers", "verl.workers.rollout"):
    module(name)
module("verl.experimental.agent_loop.agent_loop", AsyncLLMServerManager=object, DictConfigWrap=object)
module("verl.experimental.agent_loop.prometheus_utils", update_prometheus_config=noop)
module("verl.experimental.agent_loop.utils", resolve_config_path=noop)
module("verl.protocol", DataProto=DataProto)
module("verl.single_controller.ray.base", RayResourcePool=object, RayWorkerGroup=object)
sys.modules["verl.utils"].hf_processor = noop
sys.modules["verl.utils"].hf_tokenizer = noop
module("verl.utils.chat_template", initialize_system_prompt=noop)
module("verl.utils.dataset.rl_dataset", RLHFDataset=object, get_dataset_class=noop)
module("verl.utils.fs", copy_to_local=noop)
module("verl.utils.model", compute_position_id_with_mask=noop)
module("verl.utils.ray_utils", get_event_loop=noop)
module("verl.utils.rollout_trace", RolloutTraceConfig=object, rollout_trace_attr=noop)
module("verl.utils.transferqueue_utils", tqbridge=lambda: lambda function: function)
module("verl.utils.torch_functional")
module("verl.workers.rollout.replica", get_rollout_replica_class=noop)

import pytest

raise SystemExit(pytest.main([str(root / "tests/test_agent_flow_rewards.py"), "-q", "--disable-warnings"]))
