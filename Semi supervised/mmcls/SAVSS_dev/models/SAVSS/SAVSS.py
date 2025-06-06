'''
Author: Hui Liu
Github: https://github.com/Karl1109
Email: liuhui@ieee.org
'''

from typing import Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import ModuleList
from mmcls.models.builder import BACKBONES
from mmcls.models.utils import resize_pos_embed, to_2tuple
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.SAVSS_dev.models.modules.patch_embed import ConvPatchEmbed
from mmcls.SAVSS_dev.models.SAVSS.SAVSS_layer import SAVSS_Layer
from models.GBC import BottConv

@BACKBONES.register_module()
class SAVSS(BaseBackbone):
    arch_zoo = {
        'Crack': {
            'patch_size': 8,
            'embed_dims': 256,
            'num_layers': 4,
            'num_convs_patch_embed': 2,
            'layers_with_dwconv': [],
            'layer_cfgs': {
                'use_rms_norm': False,
                'mamba_cfg': {
                    'd_state': 16,
                    'expand': 2,
                    'conv_size': 7,
                    'dt_init': "random",
                    'conv_bias': True,
                    'bias': True,
                    'default_hw_shape': (512 // 8, 512 // 8)
                }
            }
        }
    }

    def __init__(self,
                 img_size=224,
                 in_channels=3,
                 arch=None,
                 patch_size=16,
                 embed_dims=192,
                 num_layers=20,
                 num_convs_patch_embed=1,
                 with_pos_embed=True,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 interpolate_mode='bicubic',
                 layer_cfgs=dict(),
                 layers_with_dwconv=[],
                 init_cfg=None,
                 test_cfg=dict(),
                 convert_syncbn=False,
                 freeze_patch_embed=False,
                 **kwargs):
        super(SAVSS, self).__init__(init_cfg)

        self.test_cfg = test_cfg
        self.img_size = to_2tuple(img_size)
        self.convert_syncbn = convert_syncbn
        self.arch = arch

        if self.arch is None:
            self.embed_dims = embed_dims
            self.num_layers = num_layers
            self.patch_size = patch_size
            self.num_convs_patch_embed = num_convs_patch_embed
            self.layers_with_dwconv = layers_with_dwconv
            _layer_cfgs = layer_cfgs
        else:
            assert self.arch in self.arch_zoo.keys()
            self.embed_dims = self.arch_zoo[self.arch]['embed_dims']
            self.num_layers = self.arch_zoo[self.arch]['num_layers']
            self.patch_size = self.arch_zoo[self.arch]['patch_size']
            self.num_convs_patch_embed = self.arch_zoo[self.arch]['num_convs_patch_embed']
            self.layers_with_dwconv = self.arch_zoo[self.arch]['layers_with_dwconv']
            _layer_cfgs = self.arch_zoo[self.arch]['layer_cfgs']

        self.with_pos_embed = with_pos_embed
        self.interpolate_mode = interpolate_mode
        self.freeze_patch_embed = freeze_patch_embed
        _drop_path_rate = drop_path_rate

        self.patch_embed = ConvPatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            num_convs=self.num_convs_patch_embed,
            patch_size=self.patch_size,
            stride=self.patch_size
        )
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        if with_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
            trunc_normal_(self.pos_embed, std=0.02)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        dpr = np.linspace(0, _drop_path_rate, self.num_layers)
        self.drop_path_rate = _drop_path_rate

        self.layer_cfgs = _layer_cfgs
        self.layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [copy.deepcopy(_layer_cfgs) for _ in range(self.num_layers)]

        for i in range(self.num_layers):
            _layer_cfg_i = layer_cfgs[i]
            _layer_cfg_i.update({
                "embed_dims": self.embed_dims,
                "drop_path_rate": dpr[i]
            })
            if i in self.layers_with_dwconv:
                _layer_cfg_i.update({"with_dwconv": True})
            else:
                _layer_cfg_i.update({"with_dwconv": False})
            self.layers.append(
                SAVSS_Layer(**_layer_cfg_i)
            )

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, self.embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        for i in out_indices:
            if i != self.num_layers - 1:
                if norm_cfg is not None:
                    norm_layer = build_norm_layer(norm_cfg, self.embed_dims)[1]
                else:
                    norm_layer = nn.Identity()
                self.add_module(f'norm_layer{i}', norm_layer)

        self.conv256to128 = BottConv(in_channels=256, out_channels=128, mid_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv256to64 = BottConv(in_channels=256, out_channels=64, mid_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv256to32 = BottConv(in_channels=256, out_channels=32, mid_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv256to16 = BottConv(in_channels=256, out_channels=16, mid_channels=4, kernel_size=1, stride=1, padding=0)
        self.gn128 = nn.GroupNorm(num_channels=128, num_groups=8)
        self.gn64 = nn.GroupNorm(num_channels=64, num_groups=4)
        self.gn32 = nn.GroupNorm(num_channels=32, num_groups=2)
        self.gn16 = nn.GroupNorm(num_channels=16, num_groups=2)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        super(SAVSS, self).init_weights()
        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.with_pos_embed:
                trunc_normal_(self.pos_embed, std=0.02)
        self.set_freeze_patch_embed()

    def set_freeze_patch_embed(self):
        if self.freeze_patch_embed:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    def forward(self, x):
        x, patch_resolution = self.patch_embed(x)
        if self.with_pos_embed:
            pos_embed = resize_pos_embed(
                self.pos_embed,
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0
            )
            x = x + pos_embed
        x = self.drop_after_pos(x)

        outs_before = []
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, hw_shape=patch_resolution)
            if i == len(self.layers) - 1 and self.final_norm:
                x = self.norm1(x)

            if i in self.out_indices:
                B, _, C = x.shape
                patch_token = x.reshape(B, *patch_resolution, C)
                if i != self.num_layers - 1:
                    norm_layer = getattr(self, f'norm_layer{i}')
                    patch_token = norm_layer(patch_token)
                patch_token = patch_token.permute(0, 3, 1, 2)
                outs_before.append(patch_token)

                if i == self.out_indices[0]:
                    patch_token_mid = self.gn128(self.conv256to128(patch_token))
                    patch_token_mid = nn.Upsample(size=(64, 64), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[1]:
                    patch_token_mid = self.gn64(self.conv256to64(patch_token))
                    patch_token_mid = nn.Upsample(size=(128, 128), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[2]:
                    patch_token_mid = self.gn32(self.conv256to32(patch_token))
                    patch_token_mid = nn.Upsample(size=(256, 256), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                elif i == self.out_indices[3]:
                    patch_token_mid = self.gn16(self.conv256to16(patch_token))
                    patch_token_mid = nn.Upsample(size=(256, 256), mode="bilinear")(patch_token_mid)
                    outs.append(patch_token_mid)
                else:
                    continue

        return outs