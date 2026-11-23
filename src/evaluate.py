import time

import numpy as np
import torch

from src.metrics import psnr, ssim, lpips_score


@torch.no_grad()
def evaluate(upscale_fn, dataset, device="cuda"):
    res = {"psnr": [], "ssim": [], "lpips": [], "time_ms": []}
    torch.cuda.reset_peak_memory_stats()
    for i in range(len(dataset)):
        lr, hr = dataset[i]
        lr_b = lr.unsqueeze(0).to(device)
        torch.cuda.synchronize()
        t0 = time.time()
        sr = upscale_fn(lr_b)
        torch.cuda.synchronize()
        res["time_ms"].append((time.time() - t0) * 1000)
        sr = sr.squeeze(0).clamp(0, 1).cpu()
        res["psnr"].append(psnr(sr, hr))
        res["ssim"].append(ssim(sr, hr))
        res["lpips"].append(lpips_score(sr, hr))
    summary = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    summary["vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return summary