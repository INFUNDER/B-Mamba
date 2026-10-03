"""
Inference speed / memory benchmark for B-Mamba (batch size 1, 352x352).
Reports parameters, FPS and peak GPU memory. Run on a GPU node.
"""
import time

import torch

from polyp_segmentation.b_mamba.b_mamba_model import BMambaModel_Pro


@torch.no_grad()
def main(n_warmup=30, n_iter=200, size=352):
    device = torch.device('cuda')
    model = BMambaModel_Pro(pretrained=False, img_size=size).to(device).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    branch = sum(p.numel() for p in model.b_mamba.parameters()) / 1e6
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(n_warmup):
        model(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    print(f'GPU            : {torch.cuda.get_device_name(0)}')
    print(f'Params (total) : {params:.2f} M   | B-Mamba branch: {branch:.2f} M')
    print(f'Latency        : {dt * 1000:.2f} ms/img')
    print(f'FPS            : {1.0 / dt:.1f}')
    print(f'Peak memory    : {torch.cuda.max_memory_allocated() / 2**20:.0f} MB')


if __name__ == '__main__':
    main()
