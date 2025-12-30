# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#

from typing import Dict, List, NotRequired, Sequence, TypedDict, Union, cast

import fvdb
import numpy as np
import torch
from fvdb_reality_capture.radiance_fields import SfmDataset
from fvdb_reality_capture.sfm_scene import SfmScene


class SegmentationDataItem(TypedDict):
    """Type definition for a single item in the SegmentationDataset for linting convenience."""

    image: torch.Tensor  # [H, W, 3] or [num_samples, 3]
    projection: torch.Tensor  # [3, 3]
    camera_to_world: torch.Tensor  # [4, 4]
    world_to_camera: torch.Tensor  # [4, 4]
    scales: torch.Tensor  # [NM] i.e. [0.6053, 0.4358, 0.2108, 0.2107, 0.2090, 0.1880, 0.1320,
    mask_cdf: torch.Tensor  # [H, W, MM] or [num_samples, MM] i.e. [0.5014, 1., 1., 1.]
    mask_ids: torch.Tensor  # [H, W, MM] or [num_samples, MM] i.e. [12, 14, -1, -1],
    image_h: torch.Tensor  # [1]
    image_w: torch.Tensor  # [1]
    image_full: NotRequired[torch.Tensor]  # [H, W, 3]
    pixel_coords: NotRequired[torch.Tensor]  # [num_samples, 2]


class SegmentationDataset(SfmDataset):
    """Dataset for loading the SegmentationDataset which loads the images, intrinsics, cam_to_worlds,
    scales, mask_cdfs, mask_ids from disk.  Members of this class can then be modified by further
    data transforms."""

    def __init__(
        self,
        sfm_scene: SfmScene,
        dataset_indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    ):
        """
        Args:
            segmentation_dataset_path: Path to the segmentation dataset.
        """
        super().__init__(sfm_scene, dataset_indices)

    @property
    def num_zeropad(self) -> int:
        return len(str(self._sfm_scene.num_images)) + 2

    def __getitem__(self, idx: int) -> SegmentationDataItem:  # pyright: ignore[reportIncompatibleMethodOverride]
        sfm_item = super().__getitem__(idx)

        index = self._indices[idx]
        num_zeropad = self.num_zeropad
        _, data = self.sfm_scene.cache.read_file(f"masks_{index:0{num_zeropad}}")

        return SegmentationDataItem(
            image=torch.from_numpy(sfm_item["image"]),
            projection=sfm_item["projection"],
            camera_to_world=sfm_item["camera_to_world"],
            world_to_camera=sfm_item["world_to_camera"],
            scales=data["scales"],
            mask_cdf=data["mask_cdf"],
            mask_ids=data["pixel_to_mask_id"],
            image_h=torch.tensor(sfm_item["image"].shape[0], dtype=torch.int32),
            image_w=torch.tensor(sfm_item["image"].shape[1], dtype=torch.int32),
        )

    @property
    def scales(self) -> torch.Tensor:
        scales = []
        for index in self._indices:
            _, data = self.sfm_scene.cache.read_file(f"masks_{index:0{self.num_zeropad}}")
            scales.append(data["scales"])
        return torch.cat(scales)

    @property
    def camera_to_world_matrices(self) -> np.ndarray:
        """
        Get the camera to world matrices for all images in the dataset.

        This returns the camera to world matrices as a numpy array of shape (N, 4, 4) where N is the number of images.

        Returns:
            np.ndarray: An Nx4x4 array of camera to world matrices for the cameras in the dataset.
        """
        return self.sfm_scene.camera_to_world_matrices[self._indices]

    @property
    def projection_matrices(self) -> np.ndarray:
        """
        Get the projection matrices mapping camera to pixel coordinates for all images in the dataset.

        This returns the undistorted projection matrices as a numpy array of shape (N, 3, 3) where N is the number of images.

        Returns:
            np.ndarray: An Nx3x3 array of projection matrices for the cameras in the dataset.
        """
        # in fvdb_3dgs/training/sfm_dataset.py
        return np.stack(
            [self._sfm_scene.images[i].camera_metadata.projection_matrix for i in self._indices],
            axis=0,
        )

    @property
    def indices(self) -> np.ndarray:
        """
        Return the indices of the images in the SfmScene used in the dataset.

        Returns:
            np.ndarray: The indices of the images in the SfmScene used in the dataset.
        """
        return self._indices


class GARfVDBInput(Dict[str, Union[torch.Tensor, fvdb.JaggedTensor, None]]):
    """Dictionary with custom behavior for 3D Gaussian splatting inputs."""

    def __repr__(self):
        return f"GARfVDBInput({super().__repr__()})"

    def to(self, device: torch.device) -> "GARfVDBInput":
        return GARfVDBInput(
            {k: v.to(device) if (type(v) in (torch.Tensor, fvdb.JaggedTensor)) else v for k, v in self.items()}  # type: ignore
        )


def GARfVDBInputCollateFn(batch: List[SegmentationDataItem]) -> GARfVDBInput:
    """Collate function for a DataLoader to stack the SegmentationDataItems into a GARfVDBInput.
    Args:
        batch: List of SegmentationDataItems.
    Returns:
        GARfVDBInput: A dictionary of tensors that is expected as input to the GARfVDB model.
    """

    kwargs = {
        "image": torch.stack([cast(torch.Tensor, b["image"]) for b in batch]),
        "projection": torch.stack([cast(torch.Tensor, b["projection"]) for b in batch]),
        "camera_to_world": torch.stack([cast(torch.Tensor, b["camera_to_world"]) for b in batch]),
        "world_to_camera": torch.stack([cast(torch.Tensor, b["world_to_camera"]) for b in batch]),
        "image_h": torch.tensor([cast(int, b["image_h"]) for b in batch]),
        "image_w": torch.tensor([cast(int, b["image_w"]) for b in batch]),
        "scales": torch.stack([cast(torch.Tensor, b["scales"]) for b in batch]),
        # "mask_cdf": torch.nested.nested_tensor([cast(torch.Tensor, b["mask_cdf"]) for b in batch]),
        "mask_ids": torch.stack([cast(torch.Tensor, b["mask_ids"]) for b in batch]),
    }

    if "image_full" in batch[0]:
        kwargs["image_full"] = torch.stack([cast(torch.Tensor, b.get("image_full")) for b in batch])

    if "pixel_coords" in batch[0]:
        kwargs["pixel_coords"] = torch.stack([cast(torch.Tensor, b.get("pixel_coords")) for b in batch])

    return GARfVDBInput(**kwargs)
