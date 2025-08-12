# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import logging

import point_cloud_utils as pcu
import torch
import torch.utils.data
import tqdm
import tyro
from sun_3d_dataset import Sun3dDataset

from fvdb import Grid


def reconstruct_mesh_from_depth_maps(
    dataset: Sun3dDataset,
    voxel_size: float,
    device: torch.device,
    dtype: torch.dtype,
    color_dtype: torch.dtype = torch.uint8,
):
    """
    Reconstruct a mesh from a dataset of depth maps, RGB images and camera poses using the
    TSDF fusion algorithm in fVDB.

    Args:
        dataset (Sun3dDataset): Dataset containing RGB images, depth maps and camera poses.
        voxel_size (float): Size of the voxels in the TSDF grid in world units.
        device (torch.device): Device to run the reconstruction on (e.g., 'cuda' or 'cpu').
        dtype (torch.dtype): Data type for the TSDF, weights, and color tensors.
            Supports torch.float16, torch.float32, and torch.float64.
        color_dtype (torch.dtype): Data type to use for colors during integration. Defaults to torch.uint8.
            Supports torch.uint8, torch.float16, torch.float32, and torch.float64.

    Returns:
        mesh_vertices (np.ndarray): Vertices of the reconstructed mesh.
        mesh_faces (np.ndarray): Faces of the reconstructed mesh.
        mesh_vertex_normals (np.ndarray): Normals of the reconstructed mesh vertices.
        mesh_colors (np.ndarray): Colors of the reconstructed mesh.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Initialize an empty grid as well as TSDF, weights, and colors which will will integrate
    # depth maps and images into
    dtype = torch.float16
    voxel_trunc_margin = 2.0
    trunc_margin = voxel_size * voxel_trunc_margin
    accum_grid = Grid.from_zero_voxels(device=device, voxel_size=voxel_size)
    weights = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
    tsdf = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
    colors = torch.zeros(accum_grid.num_voxels, 3, device=device, dtype=color_dtype)

    with tqdm.tqdm(dataloader, desc="Integrating depth maps", unit="frames") as pbar:
        # For each image and depth map in the dataset, get it's camera pose and integrate it into the grid.
        # This returns a new grid and updated tsdf, weights, and colors tensors.
        for i, data in enumerate(pbar):
            rgb = data["rgb"].to(device=device, dtype=torch.uint8).squeeze(0)
            depth = data["depth"].to(device=device, dtype=dtype).squeeze(0)
            cam_to_world_matrix = data["cam_to_world_matrix"].to(device=device, dtype=dtype).squeeze(0)
            projection_matrix = data["projection_matrix"].to(device=device, dtype=dtype).squeeze(0)

            cam_to_world_det = torch.linalg.det(cam_to_world_matrix.to(torch.float32))
            if cam_to_world_det < 1e-6 or cam_to_world_det.isnan().any() or cam_to_world_det.isinf().any():
                logging.info(f"Skipping image {i} due to invalid camera pose.")
                continue

            accum_grid, tsdf, weights, colors = accum_grid.integrate_tsdf_with_features(
                truncation_distance=trunc_margin,
                projection_matrix=projection_matrix,
                cam_to_world_matrix=cam_to_world_matrix,
                tsdf=tsdf,
                features=colors,
                weights=weights,
                depth_image=depth,
                feature_image=rgb,
            )
            pbar.set_postfix({"accumulated_voxels": f"{accum_grid.num_voxels:,}"})

    # Prune any voxels from the grid with zero weights. These were added during integration but never updated.
    # Correspondingly prune their TSDF and color values.
    logging.info(f"Done accumulating depth maps. Total accumulated voxels: {accum_grid.num_voxels:,}")
    logging.info(f"Pruning zero-weight voxels from the accumulated grid.")
    new_grid = accum_grid.pruned_grid(weights > 0.0)
    filter_tsdf = torch.zeros(new_grid.num_voxels, device=new_grid.device, dtype=dtype)
    filter_colors = torch.zeros(new_grid.num_voxels, 3, device=new_grid.device, dtype=torch.uint8)
    new_grid.inject_from(accum_grid, tsdf, filter_tsdf)
    new_grid.inject_from(accum_grid, colors, filter_colors)
    logging.info(f"Done pruning grid. Total voxels after pruning: {new_grid.num_voxels:,}")

    # Extract a mesh from the accumulated grid using marching cubes.
    logging.info(f"Extracting mesh from the accumulated grid using marching cubes.")
    vertices, faces, vertex_normals = new_grid.marching_cubes(filter_tsdf, 0.0)
    colors = new_grid.sample_trilinear(vertices, filter_colors.to(dtype)) / 255.0
    colors.clip_(min=0.0, max=1.0)
    logging.info(f"Mesh extraction complete. Got {vertices.shape[0]:,} Vertices, and {faces.shape[0]:,} Faces.")

    return vertices.cpu().numpy(), faces.cpu().numpy(), vertex_normals.cpu().numpy(), colors.cpu().numpy()


def main(data_path: str, voxel_size: float, output_path: str = "mesh.ply", device: str = "cuda"):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    dataset = Sun3dDataset(
        dataset_path=data_path,
    )

    # torch.device constructor requires a device index
    if device == "cuda":
        device = "cuda:0"

    v, f, n, c = reconstruct_mesh_from_depth_maps(
        dataset=dataset,
        voxel_size=voxel_size,
        device=torch.device(device),
        dtype=torch.float16,
    )
    logging.info(f"Saving mesh to {output_path}")
    pcu.save_mesh_vfnc(output_path, v, f, n, c)
    logging.info("Done.")


if __name__ == "__main__":
    tyro.cli(main)
