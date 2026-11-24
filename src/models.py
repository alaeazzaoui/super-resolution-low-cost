import math

import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class EDSRLight(nn.Module):
    def __init__(self, scale=4, n_feats=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_blocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        up = []
        for _ in range(int(math.log2(scale))):
            up += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2)]
        self.up = nn.Sequential(*up)
        self.tail = nn.Conv2d(n_feats, 3, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        return self.tail(self.up(x))
class Discriminator(nn.Module):
    def __init__(self, n_feats=64):
        super().__init__()

        def block(i, o, s):
            return [nn.Conv2d(i, o, 3, stride=s, padding=1), nn.LeakyReLU(0.2, inplace=True)]

        layers = (
            block(3, n_feats, 1) + block(n_feats, n_feats, 2)
            + block(n_feats, n_feats * 2, 1) + block(n_feats * 2, n_feats * 2, 2)
            + block(n_feats * 2, n_feats * 4, 1) + block(n_feats * 4, n_feats * 4, 2)
        )
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(n_feats * 4, 1)
        )

    def forward(self, x):
        return self.classifier(self.features(x))