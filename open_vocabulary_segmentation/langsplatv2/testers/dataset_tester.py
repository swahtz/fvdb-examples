from fvdb_reality_capture.sfm_scene import SfmScene
from langsplatv2.config import LangSplatV2PreprocessConfig
from langsplatv2.training.dataset import LangSplatV2Dataset

# Load your scene
scene = SfmScene.from_colmap("/ai/segmentation_datasets/nvos/scenes/room_undistort/")

# Create configuration with default settings
config = LangSplatV2PreprocessConfig(image_downsample_factor=4)

# Build and apply the transform pipeline
transforms = config.build_scene_transforms()
preprocessed_scene = transforms(scene)

# Create dataset
dataset = LangSplatV2Dataset(preprocessed_scene, feature_level=1, clip_n_dims=512)

for i in range(len(dataset)):
    item = dataset[i]
    print(item)
