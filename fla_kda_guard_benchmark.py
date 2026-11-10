import statistics
import time

import torch
import torch.nn.functional as F

from fla.ops.kda.fused_recurrent import fused_recurrent_kda_fwd


torch.manual_seed(42)
torch.set_default_device('cuda')
print('device', torch.cuda.get_device_name(0), 'torch', torch.__version__, flush=True)
for packed, supplied in [(False, False), (False, True), (True, False), (True, True)]:
    B, T, H, HV, K, V = (1, 8, 4, 8, 64, 64) if packed else (8, 1, 4, 8, 64, 64)
    q = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16), dim=-1)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16)
    g = F.logsigmoid(torch.randn(B, T, HV, K))
    beta = torch.rand(B, T, HV, dtype=torch.bfloat16)
    h0 = torch.randn(8, HV, K, V)
    cu = torch.arange(9, dtype=torch.int32) if packed else None
    out = torch.empty_like(v) if supplied else None

    def call():
        return fused_recurrent_kda_fwd(
            q=q, k=k, v=v, g=g, beta=beta, initial_state=h0,
            inplace_final_state=False, output_final_state=False, cu_seqlens=cu, out=out,
        )

    for _ in range(50):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(500):
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / 500)
    print('packed', packed, 'supplied', supplied, 'median_us', round(statistics.median(samples), 3),
          'samples_us', [round(x, 3) for x in samples], flush=True)
