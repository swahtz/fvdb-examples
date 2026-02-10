# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Sequence, Sized, TypedDict, cast

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from fvdb_reality_capture.radiance_fields import SfmDataset
from fvdb_reality_capture.sfm_scene import SfmScene

logger = logging.getLogger(__name__)


class LangSplatV2DataItem(TypedDict):
    """Type definition for a single item in the LangSplatV2 dataset.

    Attributes:
        image: RGB image tensor of shape ``[H, W, 3]``.
        projection: Camera projection matrix of shape ``[3, 3]``.
        camera_to_world: Camera-to-world transformation ``[4, 4]``.
        world_to_camera: World-to-camera transformation ``[4, 4]``.
        gt_features: Ground truth CLIP features ``[H, W, clip_n_dims]``.
        feature_mask: Boolean mask ``[H, W]`` of valid pixels.
        image_h: Image height.
        image_w: Image width.
    """

    image: torch.Tensor
    projection: torch.Tensor
    camera_to_world: torch.Tensor
    world_to_camera: torch.Tensor
    gt_features: torch.Tensor
    feature_mask: torch.Tensor
    image_h: int
    image_w: int


class LangSplatV2Dataset(SfmDataset):
    """Dataset for LangSplatV2 training that loads CLIP features from cache.

    Extends SfmDataset to load pre-computed CLIP features and segmentation
    maps. For each image, constructs the ground truth feature map by looking
    up CLIP features using the segmentation map at the configured scale level.

    Attributes:
        feature_level: Which SAM scale to use (0=default, 1=s, 2=m, 3=l).
        clip_n_dims: Dimensionality of CLIP features.
    """

    def __init__(
        self,
        sfm_scene: SfmScene,
        feature_level: int = 1,
        clip_n_dims: int = 512,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
        cache_features: bool = True,
        cache_images: bool = True,
    ):
        """Initialize the LangSplatV2 dataset.

        Args:
            sfm_scene: The SfmScene with CLIP features in its cache.
            feature_level: Which SAM scale level to use (1=s, 2=m, 3=l).
            clip_n_dims: CLIP feature dimensionality.
            dataset_indices: Optional indices to subset the dataset.
            cache_features: If True, cache CLIP features in memory.
            cache_images: If True, cache decoded images in memory.
        """
        super().__init__(sfm_scene, dataset_indices)
        self.feature_level = feature_level
        self.clip_n_dims = clip_n_dims
        self._cache_features = cache_features
        self._cache_images = cache_images
        self._features_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._images_cache: dict[int, dict] = {}

    @property
    def num_zeropad(self) -> int:
        return len(str(self._sfm_scene.num_images)) + 2

    def warmup_cache(self) -> None:
        """Pre-load all feature and image data into cache.

        Call this BEFORE creating a DataLoader with ``num_workers > 0``.
        Workers forked after warmup inherit the populated cache.
        """
        import tqdm

        if self._cache_features:
            for idx in tqdm.tqdm(range(len(self)), desc="Warming up feature cache"):
                index = self._indices[idx]
                if index not in self._features_cache:
                    self._get_feature_data(index)

        if self._cache_images:
            for idx in tqdm.tqdm(range(len(self)), desc="Warming up image cache"):
                index = self._indices[idx]
                if index not in self._images_cache:
                    sfm_item = super().__getitem__(idx)
                    self._images_cache[index] = sfm_item

    def read_feature_data(self, index: int) -> dict[str, torch.Tensor]:
        """Read feature data from disk cache."""
        cache_filename = f"features_{index:0{self.num_zeropad}}"
        _, data = self.sfm_scene.cache.read_file(cache_filename)
        return data

    def _get_feature_data(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load CLIP features and build GT feature map for an image.

        Args:
            index: Image index in the SfmScene.

        Returns:
            Tuple of (gt_features, feature_mask) where:
                - gt_features: ``[H, W, clip_n_dims]`` CLIP feature map
                - feature_mask: ``[H, W]`` boolean mask of valid pixels
        """
        if self._cache_features and index in self._features_cache:
            return self._features_cache[index]

        data = self.read_feature_data(index)

        features = data["features"]  # [N_total, clip_n_dims]
        seg_maps = data["seg_maps"]  # [4, H, W]

        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        if isinstance(seg_maps, np.ndarray):
            seg_maps = torch.from_numpy(seg_maps)

        features = features.float()  # Ensure float32

        # Select the segmentation map for the configured scale level
        seg_map = seg_maps[self.feature_level]  # [H, W] with indices or -1
        H, W = seg_map.shape

        # Build GT feature map by looking up features using seg_map indices
        feature_mask = seg_map >= 0  # [H, W]

        # Create the GT feature map
        gt_features = torch.zeros(H, W, self.clip_n_dims, dtype=torch.float32)
        if feature_mask.any():
            valid_indices = seg_map[feature_mask].long()  # [N_valid]
            gt_features[feature_mask] = features[valid_indices]

        result = (gt_features, feature_mask)

        if self._cache_features:
            self._features_cache[index] = result

        return result

    def _get_sfm_item(self, idx: int) -> dict:
        """Get SfmDataset item, using cache if enabled."""
        index = self._indices[idx]
        if self._cache_images and index in self._images_cache:
            return self._images_cache[index]

        sfm_item = super().__getitem__(idx)

        if self._cache_images:
            self._images_cache[index] = sfm_item

        return sfm_item

    def __getitem__(self, idx: int) -> LangSplatV2DataItem:  # type: ignore[override]
        with nvtx.range("LangSplatV2Dataset.__getitem__"):
            sfm_item = self._get_sfm_item(idx)
            index = self._indices[idx]

            gt_features, feature_mask = self._get_feature_data(index)

            return LangSplatV2DataItem(
                image=torch.from_numpy(sfm_item["image"]),
                projection=sfm_item["projection"],
                camera_to_world=sfm_item["camera_to_world"],
                world_to_camera=sfm_item["world_to_camera"],
                gt_features=gt_features,
                feature_mask=feature_mask,
                image_h=sfm_item["image"].shape[0],
                image_w=sfm_item["image"].shape[1],
            )


class LangSplatV2Input(dict):
    """Batched input dictionary for the LangSplatV2 model.

    A typed dictionary wrapper that supports device transfer.
    """

    def to(self, device: torch.device, non_blocking: bool = True) -> "LangSplatV2Input":
        result = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device, non_blocking=non_blocking)
            else:
                result[k] = v
        return LangSplatV2Input(result)


def LangSplatV2CollateFn(batch: list[LangSplatV2DataItem]) -> LangSplatV2Input:
    """Collate LangSplatV2DataItems into a batched input.

    Args:
        batch: List of data items to collate.

    Returns:
        Batched LangSplatV2Input dictionary.
    """
    return LangSplatV2Input(
        image=torch.stack([cast(torch.Tensor, b["image"]) for b in batch]),
        projection=torch.stack([cast(torch.Tensor, b["projection"]) for b in batch]),
        camera_to_world=torch.stack([cast(torch.Tensor, b["camera_to_world"]) for b in batch]),
        world_to_camera=torch.stack([cast(torch.Tensor, b["world_to_camera"]) for b in batch]),
        gt_features=torch.stack([cast(torch.Tensor, b["gt_features"]) for b in batch]),
        feature_mask=torch.stack([cast(torch.Tensor, b["feature_mask"]) for b in batch]),
        image_h=[cast(int, b["image_h"]) for b in batch],
        image_w=[cast(int, b["image_w"]) for b in batch],
    )


class InfiniteSampler(torch.utils.data.Sampler):
    """Sampler that yields dataset indices infinitely.

    Avoids DataLoader iterator recreation between epochs.
    Epoch boundaries must be tracked by counting samples.
    """

    def __init__(self, dataset: Sized, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        while True:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self._epoch)
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                indices = list(range(len(self.dataset)))
            yield from indices
            self._epoch += 1

    def __len__(self):
        raise NotImplementedError("InfiniteSampler yields indices indefinitely.")
