# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from einops import rearrange


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class DownSampler(nn.Module):
    def __init__(self, *args, **kwargs):
        ''' Required kwargs: down_factor, down_shortcut'''
        super().__init__()
        self.down_factor = kwargs.pop('down_factor')
        self.down_shortcut = kwargs.pop('down_shortcut')
        # self.dwconv = nn.Conv2d(*args, **kwargs)
        
        # Initialize the weights and biases of the layer to zero
        # nn.init.constant_(self.dwconv.weight, 0)
        # if self.dwconv.bias is not None:
        #     nn.init.constant_(self.dwconv.bias, 0)

    def forward(self, x):
        # x = self.dwconv(x) + (x if self.down_shortcut else 0)
        return rearrange(x, 'b d (h dh) (w dw) -> (b dh dw) (h w) d', dh=self.down_factor, dw=self.down_factor)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_type: str = 'vanilla',
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        
        if attn_type.startswith("window"):
            self.window_size = int(attn_type.replace("window", ""))
            self.down_factor = None
        elif attn_type.startswith("downsample"):
            self.window_size = None
            self.down_factor = int(attn_type.replace("downsample", ""))
            self.downsampler = DownSampler(dim, dim, kernel_size=5, stride=1, padding=5//2, groups=dim,
                                           down_factor=self.down_factor, down_shortcut=True)
        else:
            self.window_size = None
            self.down_factor = None
        assert not (self.window_size is not None and self.down_factor is not None)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias = qkv_bias)
        self.k = nn.Linear(dim, dim, bias = qkv_bias)
        self.v = nn.Linear(dim, dim, bias = qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = self.q(x), self.k(x), self.v(x)
        
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        q = q * self.scale

        # q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        N -= 1
        if self.down_factor is not None:
            H, W = int(N**0.5), int(N**0.5)
            x_cls, x = x[:, 0:1, :], x[:, 1:, :]
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            x = self.downsampler(x)
            B_, N_, _ = x.shape
            # for cls token insertion
            x_cls = x_cls.unsqueeze(1).repeat(1, self.downsampler.down_factor**2, 1, 1).flatten(0, 1)
            x = torch.concat([x_cls, x], dim = 1)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = self.q(x), self.k(x), self.v(x)
        
        if self.window_size is not None:
            N_ = self.window_size * self.window_size
            H, W = int(N**0.5), int(N**0.5)
            q_cls, q = q[:, 0:1, :], q[:, 1:, :]
            k_cls, k = k[:, 0:1, :], k[:, 1:, :]
            v_cls, v = v[:, 0:1, :], v[:, 1:, :]
            q = q.transpose(-2, -1).reshape(B, C, H, W)
            k = k.transpose(-2, -1).reshape(B, C, H, W)
            v = v.transpose(-2, -1).reshape(B, C, H, W)
            q = F.unfold(q, kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size))
            k = F.unfold(k, kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size))
            v = F.unfold(v, kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size))
            B, C_Wh_Ww, nW = q.shape
            q = q.reshape(B, self.num_heads, C // self.num_heads, N_, nW).permute(0, 4, 1, 3, 2).contiguous()
            k = k.reshape(B, self.num_heads, C // self.num_heads, N_, nW).permute(0, 4, 1, 3, 2).contiguous()
            v = v.reshape(B, self.num_heads, C // self.num_heads, N_, nW).permute(0, 4, 1, 3, 2).contiguous()
            q, k, v = q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1)
            # for cls token insertion
            q_cls = q_cls.reshape(B, 1, self.num_heads, C // self.num_heads).unsqueeze(1).repeat(1, nW, 1, 1, 1).flatten(0, 1).transpose(1, 2)
            k_cls = k_cls.reshape(B, 1, self.num_heads, C // self.num_heads).unsqueeze(1).repeat(1, nW, 1, 1, 1).flatten(0, 1).transpose(1, 2)
            v_cls = v_cls.reshape(B, 1, self.num_heads, C // self.num_heads).unsqueeze(1).repeat(1, nW, 1, 1, 1).flatten(0, 1).transpose(1, 2)
            q, k, v = torch.concat([q_cls, q], dim=-2), torch.concat([k_cls, k], dim=-2), torch.concat([v_cls, v], dim=-2)
        elif self.down_factor is not None:
            q = q.reshape(B_, N_ + 1, self.num_heads, C // self.num_heads)
            k = k.reshape(B_, N_ + 1, self.num_heads, C // self.num_heads)
            v = v.reshape(B_, N_ + 1, self.num_heads, C // self.num_heads)
        else:
            q = q.reshape(B, N + 1, self.num_heads, C // self.num_heads)
            k = k.reshape(B, N + 1, self.num_heads, C // self.num_heads)
            v = v.reshape(B, N + 1, self.num_heads, C // self.num_heads)

        # q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        
        if self.window_size is not None: # (B*nW, N_, num_heads, head_dim) -> (B, num_heads, c/num_heads, N_, nW)
            x = x.reshape(B, nW, N_+1, self.num_heads, C // self.num_heads).permute(0, 3, 4, 2, 1)
            x_cls, x = x[:,:,:,0:1,:], x[:,:,:,1:,:]
            x = x.reshape(B, C_Wh_Ww, nW)
            x = F.fold(x, output_size=(H, W),
                   kernel_size=(self.window_size, self.window_size),
                   stride=(self.window_size, self.window_size))  # [B, C, H, W]
            x = x.reshape(B, C, N).transpose(-2, -1)
            x_cls = x_cls.mean(dim = -1)
            x = torch.concat([x_cls.reshape(B, C, 1).transpose(-2, -1), x], dim = 1)
        elif self.down_factor is not None:  # (B*4, N_, num_heads, head_dim) 
            x_cls, x = x[:, 0:1, :, :], x[:, 1:, :, :]
            x = rearrange(x, '(b dh dw) (h w) nh hd -> b (h dh) (w dw) (nh hd)',
                          h=H//self.down_factor, w=W//self.down_factor, dh=self.down_factor, dw=self.down_factor)
            x = x.reshape(B, N, C)
            x_cls = x_cls.reshape(B, self.downsampler.down_factor ** 2, 1, C).mean(1)
            x = torch.concat([x_cls, x], dim = 1)
        else:
            x = x.reshape(B, N+1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
