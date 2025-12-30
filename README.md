## Setup Environment

```bash
conda env create -f ./garfvdb_environment.yml
```

### Build and Install ƒVDB wheel

```bash
# Clone ƒVDB repository
git clone https://github.com/openvdb/fvdb-core.git
cd fvdb-core
# Build ƒVDB wheel in build environment
conda env create -f env/build_environment.yml
conda activate fvdb_build
./build.sh wheel verbose

# Install ƒVDB wheel in garfvdb environment
conda activate fvdb_garfvdb
pip install ./dist/[fvdb wheel]

# Clone ƒVDB Reality Capture repository
git clone https://github.com/openvdb/fvdb-reality-capture.git
cd ../fvdb-reality-capture
# Install ƒVDB Reality Capture wheel in garfvdb environment
pip install .

```

## GARƒVDB on ƒVDB Reality Capture dataset

### Train Gaussian Splat Scene on Scene

1. Download
    ```bash
    frgs download safety_park
    ```
1. Train 3d gaussian splatting on safety_park scene
    ```bash
    frgs reconstruct  --run-name safety_park_1 --tx.image-downsample-factor 1  data/safety_park/
    ```


### Train GARƒVDB

1. Run the following command to train GARƒVDB
    ```bash
    python train_segmentation.py --sfm-dataset-path data/safety_park/ --reconstruction-path frgs_logs/safety_park_1/checkpoints/00024800/reconstruct_ckpt.pt --use-every-n-as-val 80 --io.use-tensorboard --io.save-images-to-tensorboard --io.save-images  --config.max-epochs 300 --tx.image-downsample-factor 2
    ```
1. Launch Tensorboard to monitor the training progress
    ```bash
    tensorboard --logdir ./garfvdb_logs
    ```
1. Go to http://localhost:6006/ to monitor the training progress

### Visualize Results of GARƒVDB
1. Start the viewer.  Choose a blend factor between 0 and 1 to control the opacity of the segmentation mask.  Choose a scale between 0 and 1 to control the scale of the segmentation mask (as a fraction of the max scale of the scene).
```bash
python view_checkpoint.py  --reconstruction-path frgs_logs/safety_park_1/checkpoints/00024800/reconstruct_ckpt.pt  --segmentation-path garfvdb_logs/[datetime directory]/checkpoints/00036600/train_ckpt.pt --initial-blend 0.2 --initial-scale 0.1
```
1. Go to http://localhost:8080/ to visualize the results
