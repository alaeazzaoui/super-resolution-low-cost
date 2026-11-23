import torch.nn.functional as F


def bicubic_upscale(lr, scale):
    out = F.interpolate(lr.unsqueeze(0), scale_factor=scale, mode="bicubic", align_corners=False)
    return out.squeeze(0).clamp(0, 1)