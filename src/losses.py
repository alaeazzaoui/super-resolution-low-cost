import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer=35):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.l1 = nn.L1Loss()

    def forward(self, sr, hr):
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std
        return self.l1(self.vgg(sr), self.vgg(hr))