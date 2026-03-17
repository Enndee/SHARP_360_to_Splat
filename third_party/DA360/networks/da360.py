# by Hualie Jiang @Insta360

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import os

import numpy as np
import torch
import torch.nn as nn

from depth_anything_v2.dpt import DepthAnythingV2
from .layers import MultiLayerMLP, modify_conv_layers

try:
    autocast = torch.amp.autocast
except AttributeError:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, *_args, **_kwargs):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class DA360(nn.Module):
    """ Depth Anything in 360°
    """
    def __init__(self, equi_h=518, equi_w=1036, dinov2_encoder="vits", frozen=[None], mixed_precision=False, **kwargs):
        super(DA360, self).__init__()
        
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.dinov2_encoder = dinov2_encoder
        self.mixed_precision = mixed_precision
        self.frozen = frozen
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.depth_anything = DepthAnythingV2(**model_configs[self.dinov2_encoder])
        if os.path.exists(f'/home/jianghualie/checkpoints/depth_anything_v2_{dinov2_encoder}.pth'):
            self.depth_anything.load_state_dict(
                torch.load(f'/home/jianghualie/checkpoints/depth_anything_v2_{dinov2_encoder}.pth', map_location='cpu'))
        
        if "vit" in frozen:
            for param in self.depth_anything.pretrained.parameters():
                param.requires_grad = False
        elif "dpt" in frozen:
            for param in self.depth_anything.depth_head.parameters():
                param.requires_grad = False
            # The last depth prediction layers cannot be frozen.
            for param in self.depth_anything.depth_head.scratch.output_conv2.parameters():
                param.requires_grad = True
            for param in self.depth_anything.depth_head.scratch.output_conv1.parameters():
                param.requires_grad = True
                        
        self.depth_anything.eval()
        self.depth_anything.depth_head.apply(modify_conv_layers)
        
        vit_dim = model_configs[self.dinov2_encoder]['out_channels'][-1]
        self.shift_mlp = MultiLayerMLP(input_dim=vit_dim, hidden_dims=[vit_dim//2, vit_dim//4], output_activation='softplus')
        self.eps = 1e-4

    def forward(self, input_equi_image):
        
        with autocast(device_type=input_equi_image.device.type, enabled=self.mixed_precision):
            ssidisp, cls_token = self.depth_anything(input_equi_image, return_cls_token=True)
        
        shift = self.shift_mlp(cls_token).unsqueeze(-1).unsqueeze(-1)+self.eps

        outputs = {}
        outputs["pred_disp"] = ssidisp+shift
        
        return outputs
