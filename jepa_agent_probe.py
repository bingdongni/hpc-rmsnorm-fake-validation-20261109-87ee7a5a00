"""Exercise the evaluation agent's public goal/act path with a differentiable world model."""

import torch

from evals.simu_env_planning.planning.gc_agent import GC_Agent


class Config(dict):
    __getattr__ = dict.__getitem__


class LinearWorld(torch.nn.Module):
    action_dim = 2
    decode_unroll = None

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.5, device="cuda"))

    def encode(self, obs, act):
        return self.scale * obs

    def unroll(self, z, act_suffix):
        return z + self.scale * act_suffix.cumsum(0)


def run(kind):
    config = Config(
        local_seed=11,
        task_specification=Config(obs="rgb", goal_source="test"),
        planner=Config(
            planner_name=kind, planning_objective=Config(objective_type="L2"),
            horizon=4, iterations=5, lr=0.13, action_noise=0.035, sample_type="zero",
            num_act_stepped=2, max_norms=[0.24], max_norm_dims=[[0]],
            adam_betas=(0.5, 0.8), adam_eps=0.02,
        ),
    )
    model = LinearWorld()
    agent = GC_Agent(config, model)
    obs = torch.tensor([0.6, -1.0], device="cuda")
    goal = torch.tensor([0.1, -0.2], device="cuda")
    agent.set_goal(goal)
    for horizon in (3, 1):
        torch.manual_seed(37)
        actual = agent.act(obs, steps_left=horizon)
        assert model.scale.grad is None
        torch.manual_seed(37)
        acts = torch.zeros(1, horizon, 2, device="cuda", requires_grad=True)
        if kind == "adam":
            opt = torch.optim.Adam([acts], lr=0.13, betas=(0.5, 0.8), eps=0.02)
        else:
            opt = torch.optim.SGD([acts], lr=0.13)
        for _ in range(5):
            opt.zero_grad()
            # The model is affine, and the configured L2 objective uses the last state.
            loss = (1.5 * (obs + acts.sum(1) - goal)).square().mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                acts.add_(torch.randn_like(acts) * 0.035)
                acts[..., 0].clamp_(-0.24, 0.24)
        torch.testing.assert_close(actual, acts[0, :2].detach().cpu())
    print(f"{kind}: matched reference at horizons 3 and 1 with noise, clipping and a trainable encoder")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(), torch.__version__)
    run("adam")
    run("gd")
