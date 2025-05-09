from typing import List
import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.models.detectors.two_stage import TwoStageDetector
from typing import Iterable

from rein.peft import *


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class FrozenBackboneTwoStageDetector(TwoStageDetector):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class FrozenNSTwoStageDetector(TwoStageDetector):
    def train(self, mode=True):
        super().train(mode)
        
        # Freeze specific layers
        for name, param in self.backbone.named_parameters():
            if any(keyword in name for keyword in ['norm1', 'norm2', 'ls1', 'ls2',
                                                   'pos_embed', 'patch_embed', 'blocks.0.',
                                                   'blocks.1.', 'blocks.2.', 'blocks.3.',
                                                   'blocks.4.', 'blocks.5.', 'blocks.6.', 'blocks.7.']):
                param.requires_grad = False



@MODELS.register_module()
class PEFTBackboneTwoStageDetector(TwoStageDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        cfg = self.backbone.lora_cfg
        Config = SoraConfig
        lora_config = Config(**cfg)
        
        self.backbone = SoraModel(lora_config, self.backbone)
        
    def train(self, mode=True):
        super().train(mode)
        
        # Freeze specific layers
        for name, param in self.backbone.named_parameters():
            if any(keyword in name for keyword in ['bn1', 'bn2', 'bn3']):
                param.requires_grad = False
        
        