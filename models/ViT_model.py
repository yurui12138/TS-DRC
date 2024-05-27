
import math
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            # ("drop", nn.Dropout(0.5)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, in_channels: int, out_channels: int, width: int, layers: int, heads: int, output_dim: int, mode: str, sensors_Number: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.mode = mode
        self.sensors_Number = sensors_Number
        self.pool = nn.AdaptiveAvgPool2d(8)
        scale = width ** -0.5
        if self.mode == 'mci':
            self.class_embedding = nn.Parameter(scale * torch.randn(width))
            self.proi = nn.Linear(output_dim, 1)
        if self.mode == 'scs':
            self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        if self.mode == 'mci':
            # x = x.reshape(x.shape[0], -1, x.shape[1])
            x = x.permute(0, 3, 2, 1)
            x = self.proi(x)
            x = x.permute(0, 3, 2, 1)

        x = self.pool(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        if self.mode == 'mci':
            x = x.reshape(-1, self.sensors_Number, x.shape[1])
            x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        if self.mode == 'scs':
            x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        if self.mode == 'mci':
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TIENet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 training_mode: str,
                 configs
                 ):
        super().__init__()
        self.training_mode = training_mode

        vision_heads = 8

        self.visual_scs = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            in_channels=1,
            out_channels=vision_width,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            sensors_Number=configs.sensors_Number,
            mode='scs'
        )


        self.visual_mci = VisionTransformer(
            input_resolution=(image_resolution//vision_patch_size),
            patch_size=1,
            in_channels=embed_dim,
            out_channels=1,
            width=(image_resolution//vision_patch_size)**2,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            sensors_Number=configs.sensors_Number,
            mode='mci'
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # if self.training_mode == 'self_supervised':
        self.mlp_pro = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(embed_dim, embed_dim * 3)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(embed_dim * 3, embed_dim))
        ]))

    @property
    def dtype_scs(self):
        return self.visual_scs.proj.dtype

    def encode_image_scs(self, image):
        return self.visual_scs(image.type(self.dtype_scs))

    @property
    def dtype_mci(self):
        return self.visual_mci.proj.dtype

    def encode_image_mci(self, image):
        return self.visual_mci(image.type(self.dtype_mci))

    def forward(self, image_feature):
        batch = image_feature.shape[0]
        dims = image_feature.shape[1]
        channel = image_feature.shape[2]
        width = image_feature.shape[3]
        height = image_feature.shape[4]

        image_feature_scs = self.encode_image_scs(image_feature.view(-1, channel, width, height))
        batch_feature = image_feature_scs.shape[0]
        width_height = int(math.sqrt(image_feature_scs.shape[1]))
        channel_feature = image_feature_scs.shape[2]
        image_feature_mci = self.encode_image_mci(image_feature_scs.view(batch_feature, channel_feature, width_height, width_height))

        # normalized features   (batch,dim)
        image_feature_mci = image_feature_mci / image_feature_mci.norm(dim=1, keepdim=True)

        return self.mlp_pro(image_feature_mci)


