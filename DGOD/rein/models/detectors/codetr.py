import copy
from typing import Tuple, Union, Iterable, List

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

from projects.CO_DETR.codetr import *
from rein.peft import *


def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything


@MODELS.register_module()
class PEFTBackboneCoDETR(CoDETR):
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
            if any(keyword in name for keyword in ['norm1', 'norm2', 'ls1', 'ls2',
                                                   'pos_embed', 'patch_embed']):
                param.requires_grad = False


@MODELS.register_module()
class FrozenBackboneCoDETR(CoDETR):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(batch_inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x