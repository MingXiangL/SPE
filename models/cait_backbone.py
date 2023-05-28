# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from timm.models import create_model as creat_transformer_backbone
import pdb

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, args=None):
        super().__init__()
        if args.dataset_file == 'coco':
            num_classes = 90
        elif 'voc' in args.dataset_file:
            num_classes = 20

        self.body, num_channels = creat_transformer_backbone(args.backbone, 
                                                pretrained=True, 
                                                num_classes=num_classes,
                                                drop_rate=args.backbone_drop_rate,
                                                drop_path_rate=args.drop_path_rate, 
                                                drop_block_rate=None, 
                                                attn_drop_rate=args.drop_attn_rate,
                                                layer_to_det=args.layer_to_det)
        self.num_channels = num_channels
        args.hidden_dim = num_channels

    def forward(self, tensor_list: NestedTensor):
        backbone_out = self.body(tensor_list)
        x = backbone_out['x_patch']
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        backbone_out['x_patch'] = NestedTensor(x, mask)
        return backbone_out



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        backbone_out = self[0](tensor_list)
        x = backbone_out['x_patch']
        pos = []
        # position encoding
        pos.append(self[1](x).to(x.tensors.dtype))

        return backbone_out, pos


def build_backbone(args):
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    print(f'use backbone:{args.backbone}')
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args=args)
    backbone.body.finetune_det()
    position_embedding = build_position_encoding(args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
