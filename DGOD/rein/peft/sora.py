import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank

from peft.utils.integrations import gather_params_ctx

# Define the necessary utility functions and classes

@dataclass
class SoraConfig:
    type: str = field(default='sora', metadata={"help": "Type of Low-rank Adaptation."})
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
        },
    )
    first_eigen: bool = field(default=False, metadata={"help": "Utilizing first r-th eigen components."})
    rank_ramp: bool = field(default=False, metadata={"help": "Utilizing Rank ramp structure, progressively scailing up rank of SoRA."})
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0, metadata={"help": "Lora dropout"})
    lora_weight_init: str = field(default="sora_niter_4", 
        metadata={"help": "Initialization scheme for lora_A and lora_B, Initialize the SoRA with fast SVD, which completes in just a few seconds."})
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False, metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="all", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."}
    )
    start_lora_idx: int = field(default=0, metadata={"help": "Index of the block from which to start applying LoRA"})

    def __post_init__(self):
        self.peft_type = "SoRA"


class SoraModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.peft_config = config
        if config.rank_ramp:
            self.ramp_idx = [12, 16, 20]
        self.model = model
        self._find_and_replace()
        self.mark_only_lora_as_trainable()
        self.forward = self.model.forward

    def _find_and_replace(self):
        kwargs = {
            "first_eigen": self.peft_config.first_eigen,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": self.peft_config.merge_weights,
        }
        for name, module in self.model.named_modules():
            target_module_found = False
            block_index = self._extract_block_index(name)
            if block_index is not None and block_index >= self.peft_config.start_lora_idx:
                if isinstance(self.peft_config.target_modules, str):
                    if re.fullmatch(self.peft_config.target_modules, name):
                        target_module_found = True
                else:
                    if any(name.endswith(target) for target in self.peft_config.target_modules):
                        target_module_found = True

            if target_module_found:
                parent, target_name = self._get_parent_module(name)
                if isinstance(module, nn.Linear):
                    if self.peft_config.rank_ramp and block_index >= self.ramp_idx[2]:
                        rank = self.peft_config.r * 8
                    elif self.peft_config.rank_ramp and block_index >= self.ramp_idx[1]:
                        rank = self.peft_config.r * 4
                    elif self.peft_config.rank_ramp and block_index >= self.ramp_idx[0]:
                        rank = self.peft_config.r * 2
                    else:
                        rank = self.peft_config.r
                    new_module = Linear(module.in_features, module.out_features, bias=module.bias is not None, r = rank, **kwargs)
                    self._replace_module(parent, target_name, new_module, module)
                    with gather_params_ctx(new_module.weight):
                        new_module.sora_init(self.peft_config.lora_weight_init)

    def _extract_block_index(self, module_name):
        match = re.search(r'blocks\.(\d+)\.', module_name)
        if match:
            return int(match.group(1))
        return None

    def _get_parent_module(self, module_name):
        module_names = module_name.split('.')
        parent_module = self.model
        for name in module_names[:-1]:
            parent_module = getattr(parent_module, name)
        return parent_module, module_names[-1]

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def mark_only_lora_as_trainable(self):
        for name, param in self.model.named_parameters():
            if "lora_" not in name and "mask_token" not in name and "register_tokens" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"{name} is trainable!")

        if self.peft_config.bias == "all":
            for name, param in self.model.named_parameters():
                block_index = self._extract_block_index(name)
                if block_index is not None and block_index >= self.peft_config.start_lora_idx:
                    if any(bias + "." in name for bias in self.peft_config.target_modules) and "bias" in name:
                        param.requires_grad = True
        elif self.peft_config.bias == "lora_only":
            for module in self.model.modules():
                if isinstance(module, LoraLayer) and hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = True
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference=False):
        config = {k: v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config


class LoraLayer:
    def __init__(self, r, lora_alpha, lora_dropout, first_eigen, merge_weights):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else lambda x: x
        self.first_eigen = first_eigen
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoraLayer):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0.0, 
                 first_eigen = True, fan_in_fan_out=False, merge_weights=False, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, first_eigen=first_eigen, 
                           lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
    
    def sora_init(self, init):
        weight = self.weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize SoRA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        
        if init == "sora":
            # Perform SVD and initialize SoRA with the factors
            U, S, Vh = torch.linalg.svd(weight.data, full_matrices=False)
            if self.first_eigen:
                Ur = U[:, :self.r]
                Sr = S[:self.r]
                Sr /= self.scaling
                Vhr = Vh[:self.r, :]
            else:
                Ur = U[:, -self.r:]
                Sr = S[-self.r:]
                Sr /= self.scaling
                Vhr = Vh[-self.r:, :]

        elif len(init.split("_niter_")) == 2:
            niter = int(init.split("_niter_")[-1])
            Ur, Sr, Vr = svd_lowrank(weight.data, self.r, niter=niter)
            Sr /= self.scaling
            Vhr = Vr.t()
            
        else:
            raise ValueError(
                f"init should be 'sora' or 'sora_niter_[number of iters]', got {init} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Vhr
        lora_B = Ur @ torch.diag(torch.sqrt(Sr))

        self.lora_A.weight.data = lora_A
        self.lora_B.weight.data = lora_B

        weight = weight.data - self.scaling * lora_B @ lora_A
        weight = weight.to(dtype)
        self.weight.data = weight
    
    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        
        if not mode and self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += (self.lora_B.weight @ self.lora_A.weight * self.scaling)
            self.merged = True
        elif self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= (self.lora_B.weight @ self.lora_A.weight * self.scaling)
            self.merged = False
    
    def eval(self):
        nn.Linear.eval(self)
        self.lora_A.eval()
        self.lora_B.eval()

    def forward(self, x):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias = self.bias)
            if self.r > 0:
                result += ((x @ self.lora_A.weight.T) @ self.lora_B.weight.T) * self.scaling
        else:
            result = F.linear(x, self.weight, bias = self.bias)
            
        return result