# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
import pdb
import numpy as np
import pdb

__all__ = [
    'cait_M48', 'cait_M36',
    'cait_S36', 'cait_S24','cait_S24_224',
    'cait_XS24','cait_XXS24','cait_XXS24_224',
    'cait_XXS36','cait_XXS36_224'
]


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, **kwargs):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        self.save_attention_map(attn)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls     

class Class_Attention_Return_Attn(Class_Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_classes = num_classes

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn1 = self.attn_drop(attn)

        x_cls = (attn1 @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls



class Multi_Class_Attention(Class_Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_classes = num_classes

        self.attn_gradients = None
        self.attention_map = None        

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        num_cls_tokens = self.num_classes + 1
        
        q = self.q(x[:,:num_cls_tokens]).unsqueeze(1).reshape(B, num_cls_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # attn_mask = torch.zeros(num_cls_tokens,N).to(x.get_device())
        # attn_mask[:,1:num_cls_tokens] = float('-inf')
        # range_list = torch.arange(num_cls_tokens)
        # attn_mask[range_list, range_list] = 0
        # attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        
        attn = q @ k.transpose(-2, -1)
        # attn += attn_mask.expand_as(attn)
        attn = attn.softmax(dim=-1)
        self.save_attention_map(attn)# 这个attn 放在drop后面，对性能影响是好还是坏
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, num_cls_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        register_hook = kwargs['register_hook'] if 'register_hook' in kwargs else False
        if register_hook:
            attn.register_hook(self.save_attn_gradients)
        return x_cls


class Multi_Class_Attention_WithoutCT0(Class_Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                         proj_drop=proj_drop)
        self.num_classes = num_classes

        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        num_cls_tokens = self.num_classes

        q = self.q(x[:, :num_cls_tokens]).unsqueeze(1).reshape(B, num_cls_tokens, self.num_heads,
                                                               C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        
        attn = attn.softmax(dim=-1)
        self.save_attention_map(attn)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, num_cls_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        register_hook = kwargs['register_hook'] if 'register_hook' in kwargs else False
        if register_hook:
            attn.register_hook(self.save_attn_gradients)
        return x_cls


class Multi_Class_Attention_Query(Class_Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_classes = num_classes

        self.attn_gradients = None
        self.attention_map = None        

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        num_cls_tokens = self.num_classes + 1
        
        q = self.q(x[:,:num_cls_tokens]).unsqueeze(1).reshape(B, num_cls_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_k = torch.cat((x[:,[0]], x[:,num_cls_tokens:]),dim=1)
        k = self.k(x_k).reshape(B, N-num_cls_tokens+1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x_k).reshape(B, N-num_cls_tokens+1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # attn_mask = torch.zeros(num_cls_tokens,N).to(x.get_device())
        # attn_mask[:,1:num_cls_tokens] = float('-inf')
        # range_list = torch.arange(num_cls_tokens)
        # attn_mask[range_list, range_list] = 0
        # attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        
        attn = q @ k.transpose(-2, -1)
        # attn += attn_mask.expand_as(attn)
        attn = attn.softmax(dim=-1)
        if not self.training:
            self.save_attention_map(attn)

        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, num_cls_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class Multi_Class_Attention_TokenToPatch(Class_Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_classes=20):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.num_classes = num_classes
        
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        register_hook = kwargs['register_hook']
        num_cls_tokens = self.num_classes + 1
        q = self.q(x[:,:num_cls_tokens]).unsqueeze(1).reshape(B, num_cls_tokens, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x[:,num_cls_tokens:]).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x[:,num_cls_tokens:]).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        self.save_attention_map(attn)
        attn = self.attn_drop(attn)

        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, num_cls_tokens, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
    
    def forward(self, x, x_cls, **kwargs):
        
        u = torch.cat((x_cls,x),dim=1)
        
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)), **kwargs)
        
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 

class LayerScale_Block_CA_MultiClass(LayerScale_Block_CA):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Multi_Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4, num_classes=20):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, \
                attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, \
                Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_values)
        self.num_classes = num_classes
        self.attn = Attention_block(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                    attn_drop=attn_drop, proj_drop=drop, num_classes=self.num_classes)
    
    def forward(self, x, x_cls, **kwargs):
        
        u = torch.cat((x_cls,x),dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u), **kwargs))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 
    
    # def forward_eval(self, x, x_cls, **kwargs):
    #     u = torch.cat((x_cls,x),dim=1)
    #     u, attn = self.attn(self.norm1(u), **kwargs)
    #     x_cls = x_cls + self.drop_path(self.gamma_1 * u)
    #     x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    #     return x_cls, attn

    # def forward(self, x, x_cls, **kwargs):
    #     if self.training:
    #         return self.forward_train(x, x_cls, **kwargs)
    #     else:
            # return self.forward_eval(x, x_cls, **kwargs)


class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.attention_map = None

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)
        weights = attn.clone()

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        self.save_attention_map(weights)
        return x
    
    
class LayerScale_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):        
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    
    
    
    
class cait_models(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention_talking_head,Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Class_Attention,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0):
        super().__init__()
        
            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        
        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, tensor_list):
        x, mask = tensor_list.decompose()
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x_ca = x * (1 - mask)
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x_ca, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
                
        x = self.norm(x)
        return x[:, 0]

    def forward(self, tensor_list):
        x = self.forward_features(tensor_list)
        
        x = self.head(x)

        return x 
        
class PatchEmbedMine(PatchEmbed):
    """ Image to Patch Embedding
    """

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TSCAM_cait(cait_models):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA_MultiClass,
                 Patch_layer=PatchEmbedMine,act_layer=nn.GELU,
                 Attention_block = Attention_talking_head, Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Multi_Class_Attention,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0, 
                layer_to_det = 23):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, \
                        num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, \
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, \
                        attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, \
                        global_pool=global_pool, block_layers=block_layers, block_layers_token=block_layers_token, \
                        Patch_layer=Patch_layer, act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, \
                        init_scale=init_scale, Attention_block_token_only=Attention_block_token_only, Mlp_block_token_only=Mlp_block_token_only, \
                        depth_token_only=depth_token_only, mlp_ratio_clstk=mlp_ratio_clstk)

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale, num_classes=num_classes)
            for i in range(depth_token_only)])

        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        trunc_normal_(self.extra_cls_token, std=.02)
        self.cls_head = nn.Linear(self.embed_dim, 1)
        self.cls_head_multi_cls = nn.Linear(self.embed_dim, self.num_classes)
        self.patch_size = patch_size
        self.norm_to_det = norm_layer(embed_dim)
        self.layer_to_det = layer_to_det
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)

    def finetune_det(self, img_size=[800, 1344], use_checkpoint=False):
        # import pdb;pdb.set_trace()

        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        print(f'patch_pos_embed.shape:{patch_pos_embed.shape}')
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(patch_pos_embed)
        self.img_size = img_size
        # if mid_pe_size == None:
        #     self.has_mid_pe = False
        #     print('No mid pe')
        # else:
        #     print('Has mid pe')
        #     self.mid_pos_embed = nn.Parameter(torch.zeros(self.depth - 1, 1, 1 + (mid_pe_size[0] * mid_pe_size[1] // self.patch_size ** 2) + 100, self.embed_dim))
        #     trunc_normal_(self.mid_pos_embed, std=.02)
        #     self.has_mid_pe = True
        #     self.mid_pe_size = mid_pe_size
        # self.use_checkpoint=use_checkpoint

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        patch_pos_embed = pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        return patch_pos_embed

    def forward_features(self, tensor_list):
        x, mask = tensor_list.decompose()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        org_cls_tokens = self.cls_token.expand(B, -1, -1)  
        extra_cls_tokens = self.extra_cls_token.expand(B, -1, -1)
        cls_tokens = torch.cat((org_cls_tokens, extra_cls_tokens), dim=1)
        temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.layer_to_det:
                x_feat = self.norm_to_det(x.clone())
        N, _, C = x.shape
        x_patch = x.transpose(1,2).view(N, C, H // self.patch_size, W // self.patch_size)
        x_patch_det = x_feat.transpose(1,2).view(N, C, H // self.patch_size, W // self.patch_size)
        attn_weights = []
        for i , blk in enumerate(self.blocks_token_only):
            # if self.training:
            cls_tokens = blk(x,cls_tokens)
            # else:
            #     cls_tokens = blk(x, cls_tokens)
            #     attn = blk.attn.get_attention_map()
            #     attn_weights.append(attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x, x_patch_det

    def forward(self, tensor_list):
        # with torch.autograd.profiler.profile(enabled=True) as prof:

        x, mask = tensor_list.decompose()
        x_patch, x_feat = self.forward_features(tensor_list)
        x_logits = self.cls_head(x_patch[:,1:1+self.num_classes]).squeeze(-1)
        x_cls_logits = self.cls_head_multi_cls(x_patch[:,0])
        # if self.training:
        #     return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'x_patch':x_feat}
        # else:
        attn_weights = torch.stack([self.blocks_token_only[0].attn.get_attention_map()]) # 
        H, W = x.shape[2], x.shape[3]
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        B = attn_weights.shape[1]
        h = H // self.patch_size
        w = W // self.patch_size
        attn_weights = attn_weights.sum(0)
        # cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        cams_cls = cams_cls.reshape([B,self.num_classes, h, w])

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return {'x_logits':x_logits, 'x_cls_logits': x_cls_logits, 'cams_cls':cams_cls, 'x_patch':x_feat}



class TSCAM_cait_two_branch(cait_models):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA_MultiClass,
                 Patch_layer=PatchEmbedMine,act_layer=nn.GELU,
                 Attention_block = Attention_talking_head, Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Multi_Class_Attention,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0, 
                layer_to_det = 23,
                **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, \
                        num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, \
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, \
                        attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, \
                        global_pool=global_pool, block_layers=block_layers, block_layers_token=block_layers_token, \
                        Patch_layer=Patch_layer, act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, \
                        init_scale=init_scale, Attention_block_token_only=Attention_block_token_only, Mlp_block_token_only=Mlp_block_token_only, \
                        depth_token_only=depth_token_only, mlp_ratio_clstk=mlp_ratio_clstk)

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale, num_classes=num_classes)
            for i in range(depth_token_only)])
        dpr = [drop_path_rate for i in range(depth)]
        self.layer_to_det = layer_to_det
        self.blocks_det = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(self.layer_to_det, depth)])
        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        trunc_normal_(self.extra_cls_token, std=.02)
        
        self.cls_head = nn.Linear(self.embed_dim, 1)
        self.cls_head_multi_cls = nn.Linear(self.embed_dim, self.num_classes)
        self.patch_size = patch_size
        self.norm_det = norm_layer(embed_dim)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        print('length of blocks_det:{}'.format(len(self.blocks_det)))

    def init_blocks_det_weight(self):
        for i in range(1,1+len(self.blocks_det)):
            self.blocks_det[-i].load_state_dict(self.blocks[-i].state_dict(), strict=True)

    def finetune_det(self, img_size=[800, 1344], use_checkpoint=False):
        # import pdb;pdb.set_trace()

        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        print(f'patch_pos_embed.shape:{patch_pos_embed.shape}')
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(patch_pos_embed)
        self.img_size = img_size

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        patch_pos_embed = pos_embed.transpose(1,2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H//self.patch_size, W//self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H,new_P_W), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        return patch_pos_embed

    def forward_features(self, tensor_list):
        x, mask = tensor_list.decompose()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        org_cls_tokens = self.cls_token.expand(B, -1, -1)  
        extra_cls_tokens = self.extra_cls_token.expand(B, -1, -1)
        cls_tokens = torch.cat((org_cls_tokens, extra_cls_tokens), dim=1)
        temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H,W))
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            # if i == self.layer_to_det:
            if i + 1 == self.layer_to_det:
                x_feat = x.clone()

        for i, blk in enumerate(self.blocks_det):
            x_feat = blk(x_feat)

        x_feat = self.norm_det(x_feat)
        N, _, C = x.shape
        x_patch = x.transpose(1,2).view(N, C, H // self.patch_size, W // self.patch_size)
        x_patch_det = x_feat.transpose(1,2).view(N, C, H // self.patch_size, W // self.patch_size)
        attn_weights = []
        for i , blk in enumerate(self.blocks_token_only):
            # if self.training:
            cls_tokens = blk(x,cls_tokens)
            # else:
            #     cls_tokens = blk(x, cls_tokens)
            #     attn = blk.attn.get_attention_map()
            #     attn_weights.append(attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        
        return x, x_patch_det

    def std_reweighting(self, cam):
        std = torch.std(cam, dim=-1, keepdim=True)
        std -= std.min(dim=1, keepdim=True)[0]
        std /= std.max(dim=1, keepdim=True)[0]
        cam = (cam * std).sum(1)
        return cam

    def forward(self, tensor_list):
        # with torch.autograd.profiler.profile(enabled=True) as prof:

        x, mask = tensor_list.decompose()
        x_patch, x_feat = self.forward_features(tensor_list)
        x_logits = self.cls_head(x_patch[:,1:1+self.num_classes]).squeeze(-1)
        x_cls_logits = self.cls_head_multi_cls(x_patch[:,0])
        # if self.training:
        #     return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'x_patch':x_feat}
        # else:
        attn_weights = self.blocks_token_only[0].attn.get_attention_map() # 
        H, W = x.shape[2], x.shape[3]
        # attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        B = x.shape[0]
        h = H // self.patch_size
        w = W // self.patch_size
        # attn_weights = attn_weights.sum(0)
        # cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        # cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        cams_cls = self.std_reweighting(attn_weights[..., 1:1+self.num_classes, 1+self.num_classes:])
        cams_cls = cams_cls.reshape([B,self.num_classes, h, w])

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return {'x_logits':x_logits, 'x_cls_logits': x_cls_logits, 'cams_cls':cams_cls, 'x_patch':x_feat}

    
class TSCAM_cait_two_branch_conv_cls_attn_woct0head(cait_models):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=LayerScale_Block,
                 block_layers_token=LayerScale_Block_CA_MultiClass,
                 Patch_layer=PatchEmbedMine, act_layer=nn.GELU,
                 Attention_block=Attention_talking_head, Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=Multi_Class_Attention,
                 Mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_clstk=4.0,
                 layer_to_det=23):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, \
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, \
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, \
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, \
                         global_pool=global_pool, block_layers=block_layers, block_layers_token=block_layers_token, \
                         Patch_layer=Patch_layer, act_layer=act_layer, Attention_block=Attention_block,
                         Mlp_block=Mlp_block, \
                         init_scale=init_scale, Attention_block_token_only=Attention_block_token_only,
                         Mlp_block_token_only=Mlp_block_token_only, \
                         depth_token_only=depth_token_only, mlp_ratio_clstk=mlp_ratio_clstk)

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale, num_classes=num_classes)
            for i in range(depth_token_only)])
        dpr = [drop_path_rate for i in range(depth)]
        self.layer_to_det = layer_to_det
        self.blocks_det = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(self.layer_to_det, depth)])
        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes + 1, self.embed_dim))
        trunc_normal_(self.extra_cls_token, std=.02)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(self.embed_dim, self.num_classes, 3, padding=1)
        self.cls_head = nn.Linear(self.embed_dim, 1)
        #         self.cls_head_multi_cls = nn.Linear(self.embed_dim, self.num_classes)
        self.patch_size = patch_size
        self.norm_det = norm_layer(embed_dim)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        print('length of blocks_det:{}'.format(len(self.blocks_det)))

    def init_blocks_det_weight(self):
        for i in range(1, 1 + len(self.blocks_det)):
            self.blocks_det[-i].load_state_dict(self.blocks[-i].state_dict(), strict=True)

    def finetune_det(self, img_size=[800, 1344], use_checkpoint=False):
        # import pdb;pdb.set_trace()

        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        print(f'patch_pos_embed.shape:{patch_pos_embed.shape}')
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(patch_pos_embed)
        self.img_size = img_size

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        patch_pos_embed = pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        return patch_pos_embed

    def forward_features(self, tensor_list):
        x, mask = tensor_list.decompose()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        cls_tokens = self.extra_cls_token.expand(B, -1, -1)
        temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H, W))
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # if i == self.layer_to_det:
            if i + 1 == self.layer_to_det:
                x_feat = x.clone()

        for i, blk in enumerate(self.blocks_det):
            x_feat = blk(x_feat)

        x_feat = self.norm_det(x_feat)
        N, _, C = x.shape
        x_patch = x.transpose(1, 2).view(N, C, H // self.patch_size, W // self.patch_size)
        x_patch_det = x_feat.transpose(1, 2).view(N, C, H // self.patch_size, W // self.patch_size)
        attn_weights = []
        for i, blk in enumerate(self.blocks_token_only):
            # if self.training:
            cls_tokens = blk(x, cls_tokens)
            # else:
            #     cls_tokens = blk(x, cls_tokens)
            #     attn = blk.attn.get_attention_map()
            #     attn_weights.append(attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)

        return x, x_patch_det

    def forward(self, tensor_list):
        # with torch.autograd.profiler.profile(enabled=True) as prof:

        x, mask = tensor_list.decompose()
        x_patch, x_feat = self.forward_features(tensor_list)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x_patch_cls = x_patch[:, self.num_classes + 1:].reshape((B, h, w, -1))
        x_patch_cls = x_patch_cls.permute([0, 3, 1, 2]).contiguous()
        x_patch_map = self.conv_head(x_patch_cls)
        x_cls_logits = self.avg_pool(x_patch_map).squeeze(3).squeeze(2)

        x_logits = self.cls_head(x_patch[:, 1:1 + self.num_classes]).squeeze(-1)
        # x_cls_logits = self.cls_head_multi_cls(x_patch[:,0])
        # if self.training:
        #     return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'x_patch':x_feat}
        # else:
        attn_weights = torch.stack([self.blocks_token_only[0].attn.get_attention_map()])  #
        H, W = x.shape[2], x.shape[3]
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        B = attn_weights.shape[1]
        h = H // self.patch_size
        w = W // self.patch_size
        attn_weights = attn_weights.sum(0)
        # cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        cams_cls = attn_weights[:, 1:1 + self.num_classes, 1 + self.num_classes:]
        cams_cls = cams_cls.reshape([B, self.num_classes, h, w])
        cams_cls = F.relu(cams_cls * x_patch_map)

        attn_patch_weights = torch.stack([blk.attn.get_attention_map() for blk in self.blocks])
        attn_patch_weights = attn_patch_weights.mean(dim=2).mean(dim=0)
        attn_cls_patch = torch.einsum('bci,bij->bcj', (attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:], attn_patch_weights))
        attn_cls_patch = attn_cls_patch.reshape([B, self.num_classes, h, w])
        cams_cls_patch = F.relu(attn_cls_patch * x_patch_map)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'cams_cls': cams_cls, 'x_patch': x_feat, 'cams_cls_patch': cams_cls_patch}


class TSCAM_cait_two_branch_conv_cls_attn_woct0head_v2(cait_models):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=LayerScale_Block,
                 block_layers_token=LayerScale_Block_CA_MultiClass,
                 Patch_layer=PatchEmbedMine, act_layer=nn.GELU,
                 Attention_block=Attention_talking_head, Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=Multi_Class_Attention_WithoutCT0,
                 Mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_clstk=4.0,
                 layer_to_det=23):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, \
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, \
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, \
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, \
                         global_pool=global_pool, block_layers=block_layers, block_layers_token=block_layers_token, \
                         Patch_layer=Patch_layer, act_layer=act_layer, Attention_block=Attention_block,
                         Mlp_block=Mlp_block, \
                         init_scale=init_scale, Attention_block_token_only=Attention_block_token_only,
                         Mlp_block_token_only=Mlp_block_token_only, \
                         depth_token_only=depth_token_only, mlp_ratio_clstk=mlp_ratio_clstk)

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale, num_classes=num_classes)
            for i in range(depth_token_only)])
        dpr = [drop_path_rate for i in range(depth)]
        self.layer_to_det = layer_to_det
        self.blocks_det = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(self.layer_to_det, depth)])
        # self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes + 1, self.embed_dim))
        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        trunc_normal_(self.extra_cls_token, std=.02)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(self.embed_dim, self.num_classes, 3, padding=1)
        self.cls_head = nn.Linear(self.embed_dim, 1)
        #         self.cls_head_multi_cls = nn.Linear(self.embed_dim, self.num_classes)
        self.patch_size = patch_size
        self.norm_det = norm_layer(embed_dim)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        print('length of blocks_det:{}'.format(len(self.blocks_det)))

    def init_blocks_det_weight(self):
        for i in range(1, 1 + len(self.blocks_det)):
            self.blocks_det[-i].load_state_dict(self.blocks[-i].state_dict(), strict=True)

    def finetune_det(self, img_size=[800, 1344], use_checkpoint=False):
        # import pdb;pdb.set_trace()

        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        print(f'patch_pos_embed.shape:{patch_pos_embed.shape}')
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(patch_pos_embed)
        self.img_size = img_size

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        patch_pos_embed = pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        return patch_pos_embed

    def forward_features(self, tensor_list):
        x, mask = tensor_list.decompose()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        cls_tokens = self.extra_cls_token.expand(B, -1, -1)
        temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H, W))
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # if i == self.layer_to_det:
            if i + 1 == self.layer_to_det:
                x_feat = x.clone()

        for i, blk in enumerate(self.blocks_det):
            x_feat = blk(x_feat)

        x_feat = self.norm_det(x_feat)
        N, _, C = x.shape
        x_patch = x.transpose(1, 2).view(N, C, H // self.patch_size, W // self.patch_size)
        x_patch_det = x_feat.transpose(1, 2).view(N, C, H // self.patch_size, W // self.patch_size)
        attn_weights = []
        for i, blk in enumerate(self.blocks_token_only):
            # if self.training:
            cls_tokens = blk(x, cls_tokens)
            # else:
            #     cls_tokens = blk(x, cls_tokens)
            #     attn = blk.attn.get_attention_map()
            #     attn_weights.append(attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)

        return x, x_patch_det

    def forward(self, tensor_list):
        # with torch.autograd.profiler.profile(enabled=True) as prof:

        x, mask = tensor_list.decompose()
        x_patch, x_feat = self.forward_features(tensor_list)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        # x_patch_cls = x_patch[:, self.num_classes + 1:].reshape((B, h, w, -1))
        x_patch_cls = x_patch[:, self.num_classes:].reshape((B, h, w, -1))
        x_patch_cls = x_patch_cls.permute([0, 3, 1, 2]).contiguous()
        x_patch_map = self.conv_head(x_patch_cls)
        x_cls_logits = self.avg_pool(x_patch_map).squeeze(3).squeeze(2)

        # x_logits = self.cls_head(x_patch[:, 1:1 + self.num_classes]).squeeze(-1)
        x_logits = self.cls_head(x_patch[:, :self.num_classes]).squeeze(-1)
        # x_cls_logits = self.cls_head_multi_cls(x_patch[:,0])
        # if self.training:
        #     return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'x_patch':x_feat}
        # else:
        attn_weights = torch.stack([self.blocks_token_only[0].attn.get_attention_map()])  #
        H, W = x.shape[2], x.shape[3]
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        B = attn_weights.shape[1]
        h = H // self.patch_size
        w = W // self.patch_size
        attn_weights = attn_weights.sum(0)
        # cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        # cams_cls = attn_weights[:, 1:1 + self.num_classes, 1 + self.num_classes:]
        cams_cls = attn_weights[:, :self.num_classes, self.num_classes:]
        cams_cls = cams_cls.reshape([B, self.num_classes, h, w])
        cams_cls = F.relu(cams_cls * x_patch_map)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'cams_cls': cams_cls, 'x_patch': x_feat}


class TSCAM_cait_two_branch_conv_cls_attn_woct0head_v3(cait_models):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=LayerScale_Block,
                 block_layers_token=LayerScale_Block_CA_MultiClass,
                 Patch_layer=PatchEmbedMine, act_layer=nn.GELU,
                 Attention_block=Attention_talking_head, Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=Multi_Class_Attention,
                 Mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_clstk=4.0,
                 layer_to_det=23):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, \
                         num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, \
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, \
                         attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, \
                         global_pool=global_pool, block_layers=block_layers, block_layers_token=block_layers_token, \
                         Patch_layer=Patch_layer, act_layer=act_layer, Attention_block=Attention_block,
                         Mlp_block=Mlp_block, \
                         init_scale=init_scale, Attention_block_token_only=Attention_block_token_only,
                         Mlp_block_token_only=Mlp_block_token_only, \
                         depth_token_only=depth_token_only, mlp_ratio_clstk=mlp_ratio_clstk)

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale, num_classes=num_classes)
            for i in range(depth_token_only)])
        dpr = [drop_path_rate for i in range(depth)]
        self.layer_to_det = layer_to_det
        self.blocks_det = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale)
            for i in range(self.layer_to_det, depth)])
        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        trunc_normal_(self.extra_cls_token, std=.02)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(self.embed_dim, self.num_classes, 3, padding=1)
        self.cls_head = nn.Linear(self.embed_dim, 1)
        #         self.cls_head_multi_cls = nn.Linear(self.embed_dim, self.num_classes)
        self.patch_size = patch_size
        self.norm_det = norm_layer(embed_dim)
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        print('length of blocks_det:{}'.format(len(self.blocks_det)))

    def init_blocks_det_weight(self):
        for i in range(1, 1 + len(self.blocks_det)):
            self.blocks_det[-i].load_state_dict(self.blocks[-i].state_dict(), strict=True)

    def finetune_det(self, img_size=[800, 1344], use_checkpoint=False):
        # import pdb;pdb.set_trace()

        patch_pos_embed = self.pos_embed
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        print(f'patch_pos_embed.shape:{patch_pos_embed.shape}')
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)
        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        self.pos_embed = nn.Parameter(patch_pos_embed)
        self.img_size = img_size

    def InterpolateInitPosEmbed(self, pos_embed, img_size=(800, 1344)):
        # import pdb;pdb.set_trace()
        patch_pos_embed = pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape

        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        # patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        H, W = img_size
        new_P_H, new_P_W = H // self.patch_size, W // self.patch_size
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                    align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        return patch_pos_embed

    def forward_features(self, tensor_list):
        x, mask = tensor_list.decompose()
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        x = self.patch_embed(x)

        org_cls_tokens = self.cls_token.expand(B, -1, -1)
        extra_cls_tokens = self.extra_cls_token.expand(B, -1, -1)
        cls_tokens = torch.cat((org_cls_tokens, extra_cls_tokens), dim=1)
        temp_pos_embed = self.InterpolateInitPosEmbed(self.pos_embed, img_size=(H, W))
        x = x + temp_pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            # if i == self.layer_to_det:
            if i + 1 == self.layer_to_det:
                x_feat = x.clone()

        for i, blk in enumerate(self.blocks_det):
            x_feat = blk(x_feat)

        x_feat = self.norm_det(x_feat)
        N, _, C = x.shape
        x_patch = x.transpose(1, 2).view(N, C, H // self.patch_size, W // self.patch_size)
        x_patch_det = x_feat.transpose(1, 2).view(N, C, H // self.patch_size, W // self.patch_size)
        attn_weights = []
        for i, blk in enumerate(self.blocks_token_only):
            # if self.training:
            cls_tokens = blk(x, cls_tokens)
            # else:
            #     cls_tokens = blk(x, cls_tokens)
            #     attn = blk.attn.get_attention_map()
            #     attn_weights.append(attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)

        return x, x_patch_det

    def forward(self, tensor_list):
        # with torch.autograd.profiler.profile(enabled=True) as prof:

        x, mask = tensor_list.decompose()
        x_patch, x_feat = self.forward_features(tensor_list)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x_patch_cls = x_patch[:, self.num_classes + 1:].reshape((B, h, w, -1))
        x_patch_cls = x_patch_cls.permute([0, 3, 1, 2]).contiguous()
        x_patch_map = self.conv_head(x_patch_cls)
        x_cls_logits = self.avg_pool(x_patch_map).squeeze(3).squeeze(2)

        x_logits = self.cls_head(x_patch[:, 1:1 + self.num_classes]).squeeze(-1)
        # x_cls_logits = self.cls_head_multi_cls(x_patch[:,0])
        # if self.training:
        #     return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'x_patch':x_feat}
        # else:
        attn_weights = torch.stack([self.blocks_token_only[0].attn.get_attention_map()])  #
        H, W = x.shape[2], x.shape[3]
        attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
        B = attn_weights.shape[1]
        h = H // self.patch_size
        w = W // self.patch_size
        attn_weights = attn_weights.sum(0)
        # cams_cls = attn_weights[:, 1:1+self.num_classes, 1+self.num_classes:]
        cams_cls = attn_weights[:, 1:1 + self.num_classes, 1 + self.num_classes:]
        cams_cls = cams_cls.reshape([B, self.num_classes, h, w])
        cams_cls = F.relu(cams_cls * x_patch_map)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        return {'x_logits': x_logits, 'x_cls_logits': x_cls_logits, 'cams_cls': cams_cls, 'x_patch': x_feat}


@register_model
def cait_XXS24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model, 192

@register_model
def TSCAM_cait_conv_XXS24_224(pretrained=False, **kwargs):
    model = TSCAM_cait_conv(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model, 192

@register_model
def TSCAM_cait_XXS24_224(pretrained=False, **kwargs):
    model = TSCAM_cait(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.'+k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
        
    return model, 192


@register_model
def cait_XXS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model, 192

@register_model
def TSCAM_cait_XXS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model, 192


@register_model
def TSCAM_cait_XXS24(pretrained=False, **kwargs):
    model = TSCAM_cait(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.'+k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
    return model, 192

@register_model
def TSCAM_cait_XXS24_two_attn(pretrained=False, **kwargs):
    model = TSCAM_cait_two_attn(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.'+k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
    return model 


@register_model
def cait_XXS36_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_XXS36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.'+k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
        
    return model 

@register_model
def TSCAM_cait_XXS36(pretrained=False, **kwargs):
    model = TSCAM_cait(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.'+k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
    return model, 192


@register_model
def TSCAM_cait_XXS36_Two_Branch(pretrained=False, **kwargs):
    model = TSCAM_cait_two_branch(
        img_size= 384, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.'+k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
        model.init_blocks_det_weight()
    return model, 192


@register_model
def TSCAM_cait_XXS36_Two_Branch_conv_cls_attn_woct0head(pretrained=False, **kwargs):
    model = TSCAM_cait_two_branch_conv_cls_attn_woct0head(
        img_size=384, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.' + k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()

        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.' + k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
        model.init_blocks_det_weight()
    return model, 192


@register_model
def TSCAM_cait_XXS36_Two_Branch_conv_cls_attn_woct0head_v2(pretrained=False, **kwargs):
    model = TSCAM_cait_two_branch_conv_cls_attn_woct0head_v2(
        img_size=384, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.' + k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()

        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.' + k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
        model.init_blocks_det_weight()
    return model, 192


@register_model
def TSCAM_cait_XXS36_Two_Branch_conv_cls_attn_woct0head_v3(pretrained=False, **kwargs):
    model = TSCAM_cait_two_branch_conv_cls_attn_woct0head_v3(
        img_size=384, patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            tmp_key = 'module.' + k
            if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
                print(f"Removing key {tmp_key} from pretrained checkpoint")
                del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()

        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.' + k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
        model.init_blocks_det_weight()
    return model, 192


@register_model
def TSCAM_cait_XXS36_concat_heads(pretrained=False, **kwargs):
    model = TSCAM_cait_concat_heads(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

#         for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
#             tmp_key = 'module.'+k
#             if tmp_key in checkpoint and checkpoint[tmp_key].shape != model_dict[k].shape:
#                 print(f"Removing key {tmp_key} from pretrained checkpoint")
#                 del checkpoint[tmp_key]
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # pdb.set_trace()
        
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            new_key = 'module.'+k
            if new_key in checkpoint:
                checkpoint_no_module[k] = checkpoint['module.' + k]
        model.load_state_dict(checkpoint_no_module, strict=False)
    return model 


@register_model
def cait_XS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 


@register_model
def cait_S24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 


@register_model
def cait_S24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_S36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 


@register_model
def cait_M36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 


@register_model
def cait_M48(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 448 , patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M48_448.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model         
