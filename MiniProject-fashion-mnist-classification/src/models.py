"""Model zoo: a linear baseline, an MLP, two CNNs and a small Vision Transformer."""
from __future__ import annotations

import torch
from torch import nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LogReg(nn.Module):
    """Softmax regression on raw pixels - the simplest baseline."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, num_classes))

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    """784-512-256-10 fully connected network with dropout."""

    def __init__(self, num_classes: int = 10, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Classic two-block CNN (conv-conv-pool x2 -> FC), no batch norm."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNetLite(nn.Module):
    """Small ResNet (He et al., 2016): stem + 3 residual stages (16/32/64 ch)."""

    def __init__(self, num_classes: int = 10, width: int = 16, p: float = 0.2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)
        )
        self.layers = nn.Sequential(
            ResBlock(w, w), ResBlock(w, 2 * w, stride=2), ResBlock(2 * w, 4 * w, stride=2)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p), nn.Linear(4 * w, num_classes)
        )

    def forward(self, x):
        return self.head(self.layers(self.stem(x)))


class TinyViT(nn.Module):
    """Small Vision Transformer (Dosovitskiy et al., 2021) for 28x28 inputs.

    4x4 patches -> 49 tokens + [CLS], pre-norm encoder blocks, learned positions.
    """

    def __init__(self, num_classes: int = 10, patch: int = 4, dim: int = 64,
                 depth: int = 4, heads: int = 4, mlp_ratio: int = 2, p: float = 0.1):
        super().__init__()
        n_tokens = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio, dropout=p,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)           # B, N, dim
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], dim=1) + self.pos
        t = self.norm(self.encoder(t))
        return self.head(t[:, 0])


MODEL_FACTORIES = {
    "logreg": LogReg,
    "mlp": MLP,
    "cnn": SimpleCNN,
    "resnet": ResNetLite,
    "resnet_noaug": ResNetLite,   # ablation: same net, no data augmentation
    "vit": TinyViT,
}
