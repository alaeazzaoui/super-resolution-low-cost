import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

from src.data import SRDataset
from src.metrics import psnr


def _np(t):
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def compare_methods(hr_path, methods, out_path, scale=4, crop=64, device="cpu"):
    """Figure de comparaison avec zoom sur la zone ou l'erreur du dernier modele est maximale."""
    img = Image.open(hr_path).convert("RGB")
    w, h = img.size
    w, h = w - w % scale, h - h % scale
    hr_img = img.crop((0, 0, w, h))
    lr_img = hr_img.resize((w // scale, h // scale), Image.BICUBIC)

    hr = SRDataset._to_tensor(hr_img)
    lr = SRDataset._to_tensor(lr_img).unsqueeze(0).to(device)

    outputs = {name: fn(lr).squeeze(0).clamp(0, 1).cpu() for name, fn in methods.items()}
    last_name = list(outputs)[-1]
    last = outputs[last_name]

    err = (last - hr).abs().mean(0)
    pooled = F.avg_pool2d(err[None, None], kernel_size=crop, stride=crop // 2)
    row, col = divmod(pooled.flatten().argmax().item(), pooled.shape[-1])
    y0, x0 = row * (crop // 2), col * (crop // 2)

    names = ["Référence (HR)"] + list(outputs) + [f"Erreur ({last_name})"]
    images = [_np(hr)] + [_np(o) for o in outputs.values()]
    titles = ["Référence (HR)"] + [f"{n}\nPSNR {psnr(o, hr):.2f} dB" for n, o in outputs.items()]

    fig, axes = plt.subplots(2, len(names), figsize=(3.2 * len(names), 6.8))
    for j, (im, title) in enumerate(zip(images, titles)):
        axes[0, j].imshow(im)
        axes[0, j].set_title(title, fontsize=9)
        axes[0, j].add_patch(Rectangle((x0, y0), crop, crop, fill=False, edgecolor="red", linewidth=1.5))
        axes[1, j].imshow(im[y0:y0 + crop, x0:x0 + crop], interpolation="nearest")

    axes[0, -1].imshow(err.numpy(), cmap="gray", vmin=0, vmax=float(err.max()))
    axes[0, -1].set_title(names[-1], fontsize=9)
    axes[1, -1].imshow(err.numpy()[y0:y0 + crop, x0:x0 + crop], cmap="gray", vmin=0, vmax=float(err.max()), interpolation="nearest")

    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)