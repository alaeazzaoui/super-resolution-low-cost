import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None


def _to_np(t):
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def psnr(sr, hr):
    return peak_signal_noise_ratio(_to_np(hr), _to_np(sr), data_range=1.0)


def ssim(sr, hr):
    return structural_similarity(_to_np(hr), _to_np(sr), data_range=1.0, channel_axis=2)


def lpips_score(sr, hr, device="cuda"):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device)
    with torch.no_grad():
        a = (sr * 2 - 1).unsqueeze(0).to(device)
        b = (hr * 2 - 1).unsqueeze(0).to(device)
        return _lpips_model(a, b).item()