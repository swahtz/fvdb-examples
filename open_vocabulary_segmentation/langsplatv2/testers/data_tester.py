from fvdb_reality_capture.sfm_scene import SfmScene
from langsplatv2.config import LangSplatV2PreprocessConfig

# Load your scene
scene = SfmScene.from_colmap("/ai/segmentation_datasets/nvos/scenes/room_undistort/")

# Create configuration with default settings
config = LangSplatV2PreprocessConfig(image_downsample_factor=4)

# Build and apply the transform pipeline
transforms = config.build_scene_transforms()
preprocessed_scene = transforms(scene)
