from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class MultitaskOutput:
    ring_boundary: torch.Tensor
    ring_distance: torch.Tensor
    defect_mask: torch.Tensor
    classification_logits: torch.Tensor


class ResNetEncoder(nn.Module):
    def __init__(self, backbone: str = "resnet34") -> None:
        super().__init__()
        if backbone not in {"resnet18", "resnet34", "resnet50"}:
            raise ValueError(f"unsupported backbone: {backbone}")
        base = getattr(models, backbone)(weights=None)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.out_channels = 512 if backbone in {"resnet18", "resnet34"} else 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        logits = self.layers(x)
        return torch.nn.functional.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class SharedMultitaskBaseline(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet34",
        ring_channels: int = 2,
        defect_channels: int = 3,
        classification_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(backbone)
        self.ring_head = SegmentationHead(self.encoder.out_channels, ring_channels)
        self.defect_head = SegmentationHead(self.encoder.out_channels, defect_channels)
        self.classifier = ClassificationHead(self.encoder.out_channels, classification_classes)

    def forward(
        self,
        image: torch.Tensor,
        tasks: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        output_size = (int(image.shape[-2]), int(image.shape[-1]))
        requested = tasks or {"ring", "defect", "classification"}
        outputs: dict[str, torch.Tensor] = {}
        if "ring" in requested:
            ring_logits = self.ring_head(features, output_size)
            outputs["ring_boundary"] = ring_logits[:, :1]
            outputs["ring_distance"] = ring_logits[:, 1:2]
        if "defect" in requested:
            outputs["defect_mask"] = self.defect_head(features, output_size)
        if "classification" in requested:
            outputs["classification_logits"] = self.classifier(features)
        return outputs


def build_shared_multitask_model(
    backbone: str = "resnet34",
    ring_channels: int = 2,
    defect_channels: int = 3,
    classification_classes: int = 3,
) -> SharedMultitaskBaseline:
    return SharedMultitaskBaseline(
        backbone=backbone,
        ring_channels=ring_channels,
        defect_channels=defect_channels,
        classification_classes=classification_classes,
    )


def tensor_summary(outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: list(tensor.shape) for name, tensor in outputs.items()}
