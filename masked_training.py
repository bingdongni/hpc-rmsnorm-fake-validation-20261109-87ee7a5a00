"""Compare training on incomplete task-major targets with explicit Gaussian conditioning."""
from pathlib import Path
import sys
import torch

checkout, device, expectation = sys.argv[1:]
sys.path.insert(0, str(Path(checkout).resolve()))
import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal


class TwoOutputs(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood):
        super().__init__(x, y, likelihood)
        self.log_scales = torch.nn.Parameter(x.new_tensor([-0.4, 0.7]))

    def forward(self, x):
        distance = (x - x.mT).square()
        distributions = []
        for task, scale in enumerate(self.log_scales.exp()):
            k = scale * torch.exp(-distance * (task + 1))
            mean = (task + 1) * x.squeeze(-1) + task
            distributions.append(MultivariateNormal(mean, k))
        return MultitaskMultivariateNormal.from_independent_mvns(distributions)


def case(batched, dtype):
    x = torch.linspace(-1, 1, 5, device=device, dtype=dtype).unsqueeze(-1)
    y = torch.stack((x[:, 0].sin(), 2 * x[:, 0].cos()), dim=-1)
    y[0, 1] = y[3, 0] = torch.nan
    if batched:
        x = torch.stack((x, x + 0.2))
        y = torch.stack((y, y + 0.3))
        y[1, 1, 0] = torch.nan
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2, has_global_noise=False)
    model = TwoOutputs(x, y, likelihood).to(device=device, dtype=dtype)
    likelihood.task_noises = y.new_tensor([0.12, 0.65])
    model.train()
    out = model(x)
    with gpytorch.settings.observation_nan_policy("mask"):
        actual = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)(out, y)
    # Compute the density of each independently observed task in data order.
    keep = ~y.reshape(-1, 5, 2).isnan().any(dim=0)
    reference = 0
    for t, scale in enumerate(model.log_scales.exp()):
        points = x[..., keep[:, t], :]
        cov = scale * torch.exp(-(points - points.mT).square() * (t + 1))
        cov = cov + torch.eye(cov.shape[-1], device=device, dtype=dtype) * likelihood.task_noises[..., t]
        m = (t + 1) * points.squeeze(-1) + t
        reference += torch.distributions.MultivariateNormal(m, covariance_matrix=cov).log_prob(y[..., keep[:, t], t])
    reference = reference / out.event_shape.numel()
    params = [model.log_scales, likelihood.raw_task_noises]
    ga = torch.autograd.grad(actual.sum(), params)
    gr = torch.autograd.grad(reference.sum(), params)
    delta = (actual-reference).abs().max().item()
    grad_delta = max((a-b).abs().max().item() for a,b in zip(ga,gr))
    print(f"Compared {device=} {dtype=} {batched=} max loss delta={delta:.8g} gradient delta={grad_delta:.8g}", flush=True)
    if expectation == "baseline":
        assert delta > 0.01 and grad_delta > 0.001
    else:
        torch.testing.assert_close(actual, reference)
        for a,b in zip(ga,gr):
            torch.testing.assert_close(a,b)


assert Path(gpytorch.__file__).resolve().is_relative_to(Path(checkout).resolve())
print("Device", device, torch.cuda.get_device_name() if device == "cuda" else "CPU", torch.__version__, flush=True)
for dtype in (torch.float64, torch.float32):
    for batched in (False, True):
        case(batched, dtype)
