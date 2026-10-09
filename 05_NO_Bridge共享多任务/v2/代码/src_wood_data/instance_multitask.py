from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign


class RingFPNHead(nn.Module):
    def __init__(self, fpn_channels: int = 256, hidden_channels: int = 64) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                level: nn.Sequential(
                    nn.Conv2d(fpn_channels, hidden_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(8, hidden_channels),
                    nn.ReLU(inplace=True),
                )
                for level in ("0", "1", "2", "3")
            }
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
        )

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_size = features["0"].shape[-2:]
        projected = []
        for level, projection in self.projections.items():
            feature = projection(features[level])
            if feature.shape[-2:] != base_size:
                feature = F.interpolate(feature, size=base_size, mode="bilinear", align_corners=False)
            projected.append(feature)
        logits = self.fuse(torch.cat(projected, dim=1))
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits[:, :1], logits[:, 1:2]


def build_resnet_fpn(backbone_name: str) -> BackboneWithFPN:
    if backbone_name not in {"resnet34", "resnet50"}:
        raise ValueError(f"unsupported instance multitask backbone: {backbone_name}")
    body = getattr(models, backbone_name)(weights=None)
    if backbone_name == "resnet34":
        channels = [64, 128, 256, 512]
    else:
        channels = [256, 512, 1024, 2048]
    return BackboneWithFPN(
        body,
        return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
        in_channels_list=channels,
        out_channels=256,
    )


def build_anchor_generator(profile: str) -> AnchorGenerator:
    if profile == "default":
        sizes = ((16,), (32,), (64,), (128,), (256,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    elif profile == "small_crack":
        sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0),) * 5
    elif profile == "thin_crack":
        sizes = ((8,), (16,), (32,), (64,), (128,))
        aspect_ratios = ((0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),) * 5
    else:
        raise ValueError(f"unsupported anchor profile: {profile}")
    return AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)


class SharedInstanceMultitaskModel(nn.Module):
    """Shared FPN model for rings, boxes, instance masks, and ROI classification."""

    def __init__(
        self,
        backbone_name: str = "resnet34",
        image_size: int = 320,
        num_detection_classes: int = 3,
        anchor_profile: str = "default",
        box_roi_output: int = 7,
    ) -> None:
        super().__init__()
        backbone = build_resnet_fpn(backbone_name)
        anchor_generator = build_anchor_generator(anchor_profile)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=box_roi_output, sampling_ratio=2
        )
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
        )
        self.detector = MaskRCNN(
            backbone,
            num_classes=num_detection_classes,
            min_size=image_size,
            max_size=image_size,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=box_roi_pool,
            mask_roi_pool=mask_roi_pool,
            box_score_thresh=0.001,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
        )
        self.ring_head = RingFPNHead(fpn_channels=backbone.out_channels)
        self.classification_roi_pool = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
        )
        self.instance_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.out_channels * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_detection_classes),
        )
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.anchor_profile = anchor_profile

    def _transform_and_encode(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None,
    ) -> tuple[Any, list[dict[str, torch.Tensor]] | None, OrderedDict[str, torch.Tensor]]:
        image_list, transformed_targets = self.detector.transform(images, targets)
        features = self.detector.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        return image_list, transformed_targets, features

    def _classification_logits(
        self,
        features: OrderedDict[str, torch.Tensor],
        image_sizes: list[tuple[int, int]],
        boxes: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if boxes is None:
            # Full-image classification for pre-cropped patches (classification task)
            boxes = [
                features["0"].new_tensor([[0.0, 0.0, float(width), float(height)]])
                for height, width in image_sizes
            ]
        pooled = self.classification_roi_pool(features, boxes, image_sizes)
        return self.instance_classifier(pooled)

    def forward(
        self,
        images: list[torch.Tensor],
        *,
        targets: list[dict[str, torch.Tensor]] | None = None,
        ring: bool = False,
        detection: bool = False,
        classification: bool = False,
        classification_boxes: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("images must not be empty")
        if detection and self.training and targets is None:
            raise ValueError("detection targets are required while training")

        image_list, transformed_targets, features = self._transform_and_encode(
            images, targets if detection else None
        )
        outputs: dict[str, Any] = {}

        if ring:
            boundary, distance = self.ring_head(features, image_list.tensors.shape[-2:])
            outputs["ring_boundary"] = boundary
            outputs["ring_distance"] = distance

        if classification:
            roi_boxes = classification_boxes
            if roi_boxes is None and detection and transformed_targets is not None:
                roi_boxes = [target["boxes"] for target in transformed_targets]
            outputs["classification_logits"] = self._classification_logits(
                features, image_list.image_sizes, roi_boxes
            )

        if detection:
            proposals, proposal_losses = self.detector.rpn(
                image_list, features, transformed_targets
            )
            detections, roi_losses = self.detector.roi_heads(
                features, proposals, image_list.image_sizes, transformed_targets
            )
            losses = {}
            losses.update(proposal_losses)
            losses.update(roi_losses)
            outputs["detection_losses"] = losses
            if not self.training:
                original_sizes = [tuple(image.shape[-2:]) for image in images]
                outputs["detections"] = self.detector.transform.postprocess(
                    detections, image_list.image_sizes, original_sizes
                )

        return outputs


def load_semantic_encoder_checkpoint(
    model: SharedInstanceMultitaskModel,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    destination = model.state_dict()
    prefixes = {
        "encoder.stem.0.": "detector.backbone.body.conv1.",
        "encoder.stem.1.": "detector.backbone.body.bn1.",
        "encoder.layer1.": "detector.backbone.body.layer1.",
        "encoder.layer2.": "detector.backbone.body.layer2.",
        "encoder.layer3.": "detector.backbone.body.layer3.",
        "encoder.layer4.": "detector.backbone.body.layer4.",
    }
    transferred: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for source_name, value in source.items():
        destination_name = None
        for old_prefix, new_prefix in prefixes.items():
            if source_name.startswith(old_prefix):
                destination_name = new_prefix + source_name[len(old_prefix) :]
                break
        if (
            destination_name is not None
            and destination_name in destination
            and destination[destination_name].shape == value.shape
        ):
            transferred[destination_name] = value
        elif source_name.startswith("encoder."):
            skipped.append(source_name)
    incompatible = model.load_state_dict(transferred, strict=False)
    return {
        "source": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
        "transferred_tensors": len(transferred),
        "skipped_encoder_tensors": skipped,
        "missing_after_partial_load": len(incompatible.missing_keys),
        "unexpected_after_partial_load": incompatible.unexpected_keys,
    }


__all__ = [
    "SharedInstanceMultitaskModel",
    "build_anchor_generator",
    "build_resnet_fpn",
    "load_semantic_encoder_checkpoint",
]
