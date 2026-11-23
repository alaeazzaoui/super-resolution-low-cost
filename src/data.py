import glob
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """Paires (LR, HR) generees a la volee a partir d'images haute resolution."""

    def __init__(self, hr_dir, scale=4, patch_size=128, train=True, patches_per_image=16):
        assert patch_size % scale == 0
        self.files = sorted(glob.glob(f"{hr_dir}/*.png"))
        self.scale = scale
        self.ps = patch_size
        self.train = train
        self.n = patches_per_image

    def __len__(self):
        return len(self.files) * self.n

    def __getitem__(self, idx):
        img = Image.open(self.files[idx // self.n]).convert("RGB")
        w, h = img.size
        ps = self.ps

        if self.train:
            x, y = random.randint(0, w - ps), random.randint(0, h - ps)
        else:
            x, y = (w - ps) // 2, (h - ps) // 2

        hr = img.crop((x, y, x + ps, y + ps))
        if self.train and random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)

        lr = hr.resize((ps // self.scale, ps // self.scale), Image.BICUBIC)
        return self._to_tensor(lr), self._to_tensor(hr)

    @staticmethod
    def _to_tensor(img):
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0