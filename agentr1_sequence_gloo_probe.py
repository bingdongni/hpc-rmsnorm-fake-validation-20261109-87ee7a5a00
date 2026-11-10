"""Exercise trajectory loss normalization using two real local Gloo ranks."""
import ast
import copy
from datetime import timedelta
import os
from pathlib import Path
import sys
import types
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

root = Path(os.environ.get("AGENTR1_PROBE_ROOT", r"C:\PR\agentr1-denominator-scout-20261110"))
sys.path.insert(0, str(root))
verl = types.ModuleType("verl")
utils = types.ModuleType("verl.utils")
functional = types.ModuleType("verl.utils.torch_functional")
upstream_functional = Path(os.environ.get("AGENTR1_VERL_FUNCTIONAL", r"C:\PR\verl070_torch_functional.py"))
functional_tree = ast.parse(upstream_functional.read_text(encoding="utf-8"), filename=str(upstream_functional))
functional_nodes = [node for node in functional_tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in {"masked_sum", "masked_mean"}]
assert len(functional_nodes) == 2
# Install both functions into the same namespace so masked_mean can call masked_sum.
functional_scope = {"torch": torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=functional_nodes, type_ignores=[])),
             str(upstream_functional), "exec"), functional_scope)
functional.masked_sum = functional_scope["masked_sum"]
functional.masked_mean = functional_scope["masked_mean"]
verl.utils = utils
utils.torch_functional = functional
sys.modules.update({"verl": verl, "verl.utils": utils, "verl.utils.torch_functional": functional})

from agent_r1.trainer.ppo.core_algos import agg_loss
from agent_r1.trainer.ppo.trajectory_batching import get_mini_batch_global_info, prepare_trajectory_mini_batch


class ActorConfig:
    clip_ratio = 0.2
    clip_ratio_low = None
    clip_ratio_high = None


upstream_core = Path(os.environ.get("AGENTR1_VERL_SOURCE", r"C:\PR\verl070_core_algos.py"))
upstream_tree = ast.parse(upstream_core.read_text(encoding="utf-8"), filename=str(upstream_core))
names = {"agg_loss", "compute_policy_loss_gspo"}
nodes = [copy.deepcopy(node) for node in upstream_tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(nodes) == 2
for node in nodes:
    node.decorator_list = []
upstream_scope = {"torch": torch, "Any": Any, "Optional": Optional, "ActorConfig": ActorConfig, "verl_F": functional}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(upstream_core), "exec"), upstream_scope)


class Data:
    def __init__(self, batch, uids):
        self.batch = batch
        self.non_tensor_batch = {"trajectory_uids": np.asarray(uids)}
        self.meta_info = {}

    def __len__(self):
        return self.batch["attention_mask"].shape[0]

    def select_idxs(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        result = Data({key: value[indices.to(value.device)].clone() for key, value in self.batch.items()},
                      self.non_tensor_batch["trajectory_uids"][indices.cpu().numpy()])
        result.meta_info = dict(self.meta_info)
        return result


def worker(rank, port, device):
    store = dist.TCPStore("127.0.0.1", port, None, False, timeout=timedelta(seconds=60), use_libuv=False)
    options = dist.ProcessGroupGloo._Options()
    options._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    options._timeout = timedelta(seconds=60)
    group = dist.ProcessGroupGloo(store, rank, 2, options)
    if device == "cuda":
        torch.cuda.set_device(0)
    mask = torch.tensor([[1, 1], [0, 0], [1, 1]], device=device)
    attention = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]], device=device)
    batch = prepare_trajectory_mini_batch(Data({"response_mask": mask, "attention_mask": attention}, ["a", "b", "c"]),
                                          mini_batch_size=3, dp_size=2)
    local_size = len(batch) // 2
    local = batch.select_idxs(range(rank * local_size, (rank + 1) * local_size))
    info = get_mini_batch_global_info(local)
    try:
        for mode, expected in (("seq-mean-token-mean", 1.0), ("seq-mean-token-sum", 2.0), ("token-mean", 1.0)):
            for micro in (1, 2):
                param = torch.nn.Parameter(torch.tensor(1.0, device=device))
                optimizer = torch.optim.SGD([param], lr=0.1)
                for start in range(0, local_size, micro):
                    mask_part = local.batch["response_mask"][start:start + micro]
                    loss = agg_loss(param * torch.ones_like(mask_part), mask_part, mode, dp_size=2,
                                    global_batch_size=info["global_batch_size"],
                                    batch_num_tokens=info["batch_num_tokens"])
                    loss.backward()
                grad = param.grad.detach().cpu().clone()
                group.allreduce([grad]).wait()
                grad /= 2
                param.grad.copy_(grad.to(device))
                optimizer.step()
                torch.testing.assert_close(grad, torch.tensor(expected), rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(param.detach().cpu(), torch.tensor(1.0 - 0.1 * expected), rtol=1e-6, atol=1e-6)
                if rank == 0:
                    print(device, mode, "micro", micro, "gradient", float(grad), "parameter", float(param), "pass", flush=True)
        for micro in (1, 2):
            param = torch.nn.Parameter(torch.tensor(0.0, device=device))
            config = ActorConfig()
            config.global_batch_info = {"dp_size": 2, "global_batch_size": info["global_batch_size"]}
            for start in range(0, local_size, micro):
                mask_part = local.batch["response_mask"][start:start + micro]
                logp = param * torch.ones_like(mask_part)
                loss, _ = upstream_scope["compute_policy_loss_gspo"](
                    old_log_prob=torch.zeros_like(logp), log_prob=logp, advantages=torch.ones_like(logp),
                    response_mask=mask_part, config=config)
                loss.backward()
            grad = param.grad.detach().cpu().clone()
            group.allreduce([grad]).wait()
            grad /= 2
            torch.testing.assert_close(grad, torch.tensor(-1.0), rtol=1e-6, atol=1e-6)
            if rank == 0:
                print(device, "upstream verl 0.7 GSPO", "micro", micro, "gradient", float(grad), "pass", flush=True)
    finally:
        del group


if __name__ == "__main__":
    device = os.environ.get("AGENTR1_PROBE_DEVICE", "cpu")
    store = dist.TCPStore("127.0.0.1", 0, None, True, timeout=timedelta(seconds=60), use_libuv=False)
    mp.spawn(worker, args=(store.port, device), nprocs=2, join=True)
