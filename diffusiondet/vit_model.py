# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/diffusiondet/vit_model.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet/diffusiondet
# Created Date: Tuesday, December 5th 2023, 11:00:04 am
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-01-02 21:22:07
# Modified By: Xinyu Li
# -----
# Copyright (c) 2023 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import ml_collections
# from . import vit_seg_configs as configs
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    cfg = Config.fromfile('configs/pk/faster_rcnn_r50_fpn_1x_coco.py')

    
    config.transformer = ml_collections.ConfigDict()
    config.transformer.token_length=vit_config['token_length']
    config.hidden_size = vit_config['hidden_size']
    config.transformer.mlp_dim = vit_config['mlp_dim']
    config.transformer.num_heads = vit_config['num_heads']
    config.transformer.num_layers = vit_config['num_layers']
    config.transformer.attention_dropout_rate = vit_config['attention_dropout_rate']
    config.transformer.dropout_rate = vit_config['dropout_rate']

    # config.hidden_size = 512
    # config.transformer = ml_collections.ConfigDict()
    # config.transformer.token_length=25
    # config.transformer.mlp_dim = 1024
    # config.transformer.num_heads = 4
    # config.transformer.num_layers = 10
    # config.transformer.attention_dropout_rate = 0
    # config.transformer.dropout_rate = 0.1

    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

CONFIGS = {
    'ViT-B_16':get_b16_config(),
    'testing': get_testing(),
}


logger = logging.getLogger(__name__)




def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)#
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, vis=False):
        super(Transformer, self).__init__()
        self.config=CONFIGS['ViT-B_16']
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.config.transformer.token_length, self.config.hidden_size))#x.size()[1]=9
        self.dropout = Dropout(self.config.transformer["dropout_rate"])
        self.encoder = Encoder(self.config, vis)

    def forward(self, x):#torch.Size([1500, 9, 512])
         #torch.Size([1, 9, 512])
        embeddings = x + self.position_embeddings        
        # embeddings = self.dropout(embeddings)
        encoded, attn_weights = self.encoder(embeddings)  # (B, n_patch, hidden)
        return encoded



#可删
# class VisionTransformer(nn.Module):
#     def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(VisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.classifier = config.classifier
#         self.transformer = Transformer(config, img_size, vis)
#         self.decoder = DecoderCup(config)
#         self.segmentation_head = SegmentationHead(
#             in_channels=config['decoder_channels'][-1],
#             out_channels=config['n_classes'],
#             kernel_size=3,
#         )
#         self.config = config

#     def forward(self, x):
#         if x.size()[1] == 1:#torch.Size([24, 1, 160, 160])
#             x = x.repeat(1,3,1,1)#torch.Size([24, 3, 160, 160])
#         x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
#         x = self.decoder(x, features)
#         logits = self.segmentation_head(x)#([24, 4, 160, 160])
#         return logits




