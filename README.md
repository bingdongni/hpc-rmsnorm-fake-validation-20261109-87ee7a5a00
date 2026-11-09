# hpc-ops RMSNorm fake output validation

The [GitHub Actions run](https://github.com/bingdongni/hpc-rmsnorm-fake-validation-20261109-87ee7a5a00/actions/runs/42558485336) compares hpc-ops commit `352b95a97995d41f356933f4e8d93d5a7a8e62ce` with upstream baseline `2a2e26562433a8ba4b504858f1c938eb7612c901`. The baseline failed three cases and passed one; the patch passed all four. The [job log](gpu-validation.log) records the checks.

The one-job Windows runner used an NVIDIA GeForce RTX 4060 Laptop GPU. The pinned Python wrapper and tests use the native operator's schema with a PyTorch CUDA reference implementation. The project CUDA extension was not built or run. This checks FakeTensor output metadata and `torch.compile` with the eager backend, not native kernel correctness or performance. The hpc-ops source is subject to the accompanying license.
