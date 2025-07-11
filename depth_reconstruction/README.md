fVDB Depth Map Reconstruction
---

This example shows how to reconstruct a mesh from a series of depth maps using fVDB's TSDF
fusion implementation.

To run the example, first download some example data. Here we use a sequence from the SUN3D dataset
published in ["3DMatch: Learning Local Geometric Descriptors from RGB-D Reconstructions" (Zeng et.al. 2017)](https://vision.princeton.edu/projects/2016/3DMatch/)
```
python download_example_data.py
```

Then to reconstruct a colored mesh from the data, run
```
python reconstruct_depth_map.py --data-path ./data/sun3d/sun3d-mit_76_studyroom-76-1studyroom2 --voxel-size 0.025 --output-path mesh.ply
```
which reconstructs the scene with a 2.5cm voxel resolution. The reconstruction is saved as mesh.ply.
