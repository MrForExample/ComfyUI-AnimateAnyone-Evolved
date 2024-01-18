import torch

from comfy.model_patcher import ModelPatcher

from .unet_3d import UNet3DConditionModel

class AnimateAnyoneModelPatcher(ModelPatcher):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model: UNet3DConditionModel = self.model
        
    