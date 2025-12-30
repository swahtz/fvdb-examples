# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass, field

import numpy as np
import torch
from fvdb import GaussianSplat3d
from fvdb_reality_capture.transforms import (
    Compose,
    CropScene,
    CropSceneToPoints,
    DownsampleImages,
    FilterImagesWithLowPoints,
    Identity,
    PercentileFilterPoints,
    TransformScene,
)

from garfvdb.scene_transforms import ComputeImageSegmentationMasksWithScales


@dataclass
class GARfVDBModelConfig:
    """Configuration parameters specific to the model."""

    depth_samples: int = 24
    use_grid: bool = True
    use_grid_conv: bool = False
    enc_feats_one_idx_per_ray: bool = False
    grid_feature_dim: int = 8
    gs_features: int = 192
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 4
    mlp_output_dim: int = 256


@dataclass
class GaussianSplatSegmentationTrainingConfig:
    """Configuration parameters for the training process."""

    # Random seed
    seed: int = 42
    # Number of training iterations
    max_steps: int | None = None
    max_epochs: int = 100
    sample_pixels_per_image: int = 256
    batch_size: int = 8
    accumulate_grad_steps: int = 1
    model: GARfVDBModelConfig = field(default_factory=GARfVDBModelConfig)
    log_test_images: bool = False

    # Percentage of total epochs at which we perform evaluation on the validation set. i.e. 10 means perform evaluation after 10% of the epochs.
    eval_at_percent: list[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 75, 100])
    # Percentage of total epochs at which we save the model checkpoint. i.e. 10 means save a checkpoint after 10% of the epochs.
    save_at_percent: list[int] = field(default_factory=lambda: [10, 20, 100])


@dataclass
class SfmSceneSegmentationTransformConfig:
    """
    Configuration for the transforms to apply to the SfmScene for segmentation training.
    """

    # Downsample images by this factor
    image_downsample_factor: int = 1
    # JPEG quality to use when resaving images after downsampling
    rescale_jpeg_quality: int = 95
    # Percentile of points to filter out based on their distance from the median point
    points_percentile_filter: float = 0.0
    # Optional bounding box (in the normalized space) to crop the scene to (xmin, xmax, ymin, ymax, zmin, zmax)
    crop_bbox: tuple[float, float, float, float, float, float] | None = None
    # Whether to crop the scene to the bounding box or not
    crop_to_points: bool = False
    # Minimum number of 3D points that must be visible in an image for it to be included in training
    min_points_per_image: int = 5
    # Whether to compute segmentation masks with scales
    compute_segmentation_masks: bool = True
    # SAM2 model parameters
    sam2_points_per_side: int = 40
    sam2_pred_iou_thresh: float = 0.80
    sam2_stability_score_thresh: float = 0.80
    # Device to use for the SAM2 model
    device: torch.device | str = "cuda"

    def build_scene_transforms(self, gs3d: GaussianSplat3d, normalization_transform: torch.Tensor | None):
        # SfmScene transform
        transforms = [
            TransformScene(normalization_transform.numpy()) if normalization_transform is not None else Identity(),
            PercentileFilterPoints(
                percentile_min=np.full((3,), self.points_percentile_filter),
                percentile_max=np.full((3,), 100.0 - self.points_percentile_filter),
            ),
            DownsampleImages(
                image_downsample_factor=self.image_downsample_factor,
                rescaled_jpeg_quality=self.rescale_jpeg_quality,
            ),
            FilterImagesWithLowPoints(min_num_points=self.min_points_per_image),
            (
                ComputeImageSegmentationMasksWithScales(
                    gs3d=gs3d,
                    checkpoint="large",
                    points_per_side=self.sam2_points_per_side,
                    pred_iou_thresh=self.sam2_pred_iou_thresh,
                    stability_score_thresh=self.sam2_stability_score_thresh,
                    device=self.device,
                )
                if self.compute_segmentation_masks
                else Identity()
            ),
        ]
        if self.crop_bbox is not None:
            transforms.append(CropScene(self.crop_bbox))
        if self.crop_to_points:
            transforms.append(CropSceneToPoints(margin=0.0))
        return Compose(*transforms)
