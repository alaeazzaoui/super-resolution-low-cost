import os
import sys
import time

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.models import EDSRLight

SCALE = 4
MAX_SIDE = 256
CKPT_DIR = os.environ.get("SR_CKPT_DIR", "/content/drive/MyDrive/sr_project/checkpoints")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(filename):
    path = os.path.join(CKPT_DIR, filename)
    if not os.path.exists(path):
        return None
    model = EDSRLight(scale=SCALE, n_feats=64, n_blocks=8).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()


MODELS = {"EDSR léger": load_model("edsr_light.pt"), "GAN": load_model("gan_g.pt")}
METHODS = ["Bicubique"] + [name for name, m in MODELS.items() if m is not None]


def to_tensor(img):
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def to_image(t):
    arr = (t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


@torch.no_grad()
def upscale(image, method):
    if image is None:
        return None, None, "Aucune image fournie."

    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        r = MAX_SIDE / max(w, h)
        img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BICUBIC)

    x = to_tensor(img).unsqueeze(0).to(device)
    bicubic = F.interpolate(x, scale_factor=SCALE, mode="bicubic", align_corners=False)

    t0 = time.time()
    if method == "Bicubique":
        y = bicubic
    else:
        y = MODELS[method](x)
    elapsed = (time.time() - t0) * 1000

    out = to_image(y)
    info = (
        f"Méthode : {method} | entrée {img.size[0]}×{img.size[1]} "
        f"→ sortie {out.size[0]}×{out.size[1]} | {elapsed:.0f} ms sur {device.upper()}"
    )
    return to_image(bicubic), out, info


with gr.Blocks(title="Super-résolution à faible coût") as demo:
    gr.Markdown(
        "# Super-résolution ×4\n"
        "Chargez une petite image (taille limitée à 256 pixels pour le plus grand côté) "
        "et comparez l'agrandissement bicubique avec celui d'un modèle.\n\n"
        "**Attention :** l'image de droite est générée par un modèle. "
        "Les détails ajoutés sont des estimations et peuvent être inexacts. "
        "Ne pas l'utiliser comme preuve ni pour des usages médicaux ou judiciaires."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Image d'entrée")
        with gr.Column():
            method = gr.Dropdown(METHODS, value=METHODS[-1], label="Méthode")
            btn = gr.Button("Agrandir")
    with gr.Row():
        out_bic = gr.Image(label="Bicubique (référence)")
        out_model = gr.Image(label="Résultat (image générée)")
    info = gr.Textbox(label="Informations")
    btn.click(upscale, [inp, method], [out_bic, out_model, info])

if __name__ == "__main__":
    demo.launch(share=True)