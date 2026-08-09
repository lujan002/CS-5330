"""CNN embedder and the ArcFace head used to train it.

Only CardEmbedder is exported to ONNX: image in, L2-normalized vector out. The
ArcFace head exists purely to shape the embedding space during training and is
dropped at inference, which is what lets new sets be added later by embedding
their scans instead of retraining.

No attention layers anywhere, so the graph exports to plain ONNX conv/gemm ops.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .common import EMBED_DIM

BACKBONES = ("mobilenet_v3_large", "resnet18", "resnet34")


class CardEmbedder(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenet_v3_large",
        embed_dim: int = EMBED_DIM,
        pretrained: bool = True,
    ):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}")
        self.backbone_name = backbone
        self.embed_dim = embed_dim

        weights = "DEFAULT" if pretrained else None
        if backbone == "mobilenet_v3_large":
            net = models.mobilenet_v3_large(weights=weights)
            in_features = net.classifier[0].in_features
            net.classifier = nn.Identity()
        else:
            net = getattr(models, backbone)(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        self.trunk = net

        # BN after the projection keeps the pre-normalization activations well
        # scaled, which matters a lot when every class has a single source image.
        self.head = nn.Sequential(
            nn.Linear(in_features, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.trunk(x)), p=2, dim=1)


class ArcFaceHead(nn.Module):
    """Additive angular margin softmax over card ids."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 32.0,
        margin: float = 0.30,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(embeddings, F.normalize(self.weight, p=2, dim=1))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        theta = torch.acos(cosine)
        target = torch.zeros_like(cosine)
        target.scatter_(1, labels.view(-1, 1), 1.0)

        # Past pi - margin the margin would fold the angle back around, so the
        # penalty is applied linearly there instead (the "easy margin" guard).
        margined = torch.where(
            theta + self.margin < math.pi,
            torch.cos(theta + self.margin),
            cosine - self.margin * math.sin(torch.tensor(self.margin)),
        )
        return (target * margined + (1.0 - target) * cosine) * self.scale


def build(backbone: str, num_classes: int, scale: float, margin: float, pretrained=True):
    embedder = CardEmbedder(backbone=backbone, pretrained=pretrained)
    head = ArcFaceHead(embedder.embed_dim, num_classes, scale=scale, margin=margin)
    return embedder, head


class OrientationNet(nn.Module):
    """Binary upright vs upside-down classifier on the same 224x312 crop as matching.

    MobileNetV3-Small trunk; logits[0]=upright, logits[1]=upside_down. Live score
    is logit0 - logit1 (higher => more upright).
    """

    NUM_CLASSES = 2

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        in_features = net.classifier[0].in_features  # 576
        net.classifier = nn.Identity()
        self.trunk = net
        self.head = nn.Linear(in_features, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def _conv_bn_relu(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CardnessNet(nn.Module):
    """Tiny binary full-card vs not-a-card classifier (same 224x312 input).

    Four stride-2 conv blocks + one stride-1 refine (~390k params).
    logits[0]=card, logits[1]=not_card. Live score is logit0 - logit1.
    """

    NUM_CLASSES = 2

    def __init__(self, pretrained: bool = False):
        # pretrained kept for API symmetry with OrientationNet; unused (trained from scratch).
        super().__init__()
        del pretrained
        self.trunk = nn.Sequential(
            _conv_bn_relu(3, 32, stride=2),     # 156 x 112
            _conv_bn_relu(32, 64, stride=2),    # 78 x 56
            _conv_bn_relu(64, 128, stride=2),   # 39 x 28
            _conv_bn_relu(128, 256, stride=2),  # 20 x 14
            _conv_bn_relu(256, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Linear(256, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))
