import glob
import time

import numpy as np
import torch
from PIL import Image

from src.data import SRDataset
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


@torch.no_grad()
def evaluate_images(upscale_fn, hr_dir, scale=4, device="cuda"):
    """Evaluation sur images entieres (recadrees a un multiple de `scale`)."""
    files = sorted(glob.glob(f"{hr_dir}/*.png"))
    res = {"psnr": [], "ssim": [], "lpips": [], "time_ms": []}
    per_image = {}
    use_cuda = device == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for f in files:
        img = Image.open(f).convert("RGB")
        w, h = img.size
        w, h = w - w % scale, h - h % scale
        hr_img = img.crop((0, 0, w, h))
        lr_img = hr_img.resize((w // scale, h // scale), Image.BICUBIC)

        hr = SRDataset._to_tensor(hr_img)
        lr = SRDataset._to_tensor(lr_img).unsqueeze(0).to(device)

        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        sr = upscale_fn(lr)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = (time.time() - t0) * 1000

        sr = sr.squeeze(0).clamp(0, 1).cpu()
        p, s, l = psnr(sr, hr), ssim(sr, hr), lpips_score(sr, hr, device)

        res["psnr"].append(p)
        res["ssim"].append(s)
        res["lpips"].append(l)
        res["time_ms"].append(elapsed)
        per_image[f.split("/")[-1]] = {"psnr": round(p, 3), "ssim": round(s, 4), "lpips": round(l, 4)}

    summary = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    summary["device"] = device
    if use_cuda:
        summary["vram_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return summary, per_image