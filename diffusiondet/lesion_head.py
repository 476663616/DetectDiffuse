# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/diffusiondet/lesion_head.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet/diffusiondet
# Created Date: Monday, December 4th 2023, 3:10:55 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-03-21 22:02:48
# Modified By: Xinyu Li
# -----
# Copyright (c) 2023 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###

import copy
import math
from einops import rearrange

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module): #绝对位置编码，正弦曲线位置编码
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)  #SPE= sin(pos/10000^(2i/dim))
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)
    

class Lesion_DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        rcnn_head = MultiImg_RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, ref_feas, init_bboxes, t, init_features, is_ref = False, cross_feature = None):
        # assert t shape (batch_size)
        time = self.time_mlp(t) # 先使用全连接层对时间步长进行采样，得到可学习的时间步参数

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, ref_feas, bboxes, proposal_features, self.box_pooler, time, is_ref, cross_feature)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class MultiImg_RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multi_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.box_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.pre_fusion = GALayer(cfg)
        self.layer_attention = ELA_Layer(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout_mha = nn.Dropout(dropout)
        self.dropout_box = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.DiffusionDet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.DiffusionDet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, ref_feas, bboxes, pro_features, pooler, time_emb, is_ref, cross_feature):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2] # N: batch_size or number of input images; nr_boxes: number of rotated boxes;

        features, ref_feas = self.feature_fusion(features, ref_feas)
        _, C, D, H, W = ref_feas[0].shape
                    
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b])) # split bounding boxes of the batch into a list of single images.

        # input: feature maps list witch shapes are [B, C, W, H] and split boxes ilst which shapes are [N, 4]
        # output: A tensor of shape (M, C, output_size, output_size) where M is the total number of boxes aggregated over all N batch images.
        roi_features = pooler(features, proposal_boxes) #[500*9, C, 7, 7]
        roi_refs = []
        for i in range(D):
            sub_rois = []
            for ref in ref_feas:
                sub_rois.append(ref[:,:,i])
            roi_refs.append(pooler(sub_rois, proposal_boxes)) #[500*9, C, 7, 7]
        
        roi_features = self.ela_enhanced(roi_features, roi_refs)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1) #[500*9, C, 7, 7] --> [9, 500, 256, 49] --> [9, 500, 256]

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1) #[500*9, C, 7, 7] --> [4500, 256, 49] --> [49, 4500, 256]

        # self_att.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2) # [9, 500, 256] --> [500, 9, 256]
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0] #MHA
        pro_features = pro_features + self.dropout1(pro_features2) # nn.Dropout(0.1) 施加注意力
        pro_features = self.norm1(pro_features) #使用的是layernorm

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2) #再次增强特征
        obj_features = self.norm2(pro_features) #最终将增强特征经过层归一化转化为检测目标的特征

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features) #施加了注意力的目标特征
        if is_ref and cross_feature == None:
            obj_features = self.cross_all_img(obj_features, N)
        if is_ref and cross_feature != None:
            obj_features = self.cross_wrong_img(obj_features, N, cross_feature)

        # from matplotlib import pyplot as plt
        # v = obj_features
        # v = v.data.detach().cpu()
        # plt.figure(figsize=(10, 10))
        # # for channel in range(v.shape[0]):
        # #     ax = plt.subplot(3, 6, channel+1,)
        #     # plt.imshow(v[channel, :, :])
        # plt.imshow(v[0,:,:])
        # plt.show()
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb) #nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2)) [9, 1024] --> [9, 512]
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0) # 重复nr_boxes次。[9, 512] --> [4500, 512], 与特征维度相匹配
        scale, shift = scale_shift.chunk(2, dim=1) #沿着通道维度分成两块，分别作为尺度变化与偏移（？）
        fc_feature = fc_feature * (scale + 1) + shift #这是因为原先的分布为[-1,1]，通过加一来解决负数的问题。

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature) # Linear + LayerNorm +ReLU 分类头，对输入特征进行分类
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature) # Linear + LayerNorm +ReLU 回归头，回归对检测框尺寸调整的信息
        class_logits = self.class_logits(cls_feature) # Linear(256, class_number)
        bboxes_deltas = self.bboxes_delta(reg_feature) # Linear(256, 4)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features
    
    def cross_wrong_img(self, features, N, wrong_idx):
        num_per_img = int(features.shape[1]/N)
        new_features = []
        for i in range(N):
            for j in range(num_per_img):
                if not i in wrong_idx:
                    new_features.append(features[:, int(i*num_per_img+j), :])
                    continue
                elif i == 0:
                    now_feature = features[:, int(i*num_per_img+j), :]
                    before_feature = features[:, int((i+1)*num_per_img+j), :]
                    after_feature = features[:, int((i+2)*num_per_img+j), :]
                elif i == N-1:
                    now_feature = features[:, int(i*num_per_img+j), :]
                    before_feature = features[:, int((i-1)*num_per_img+j), :]
                    after_feature = features[:, int((i-2)*num_per_img+j), :]
                else:
                    now_feature = features[:, int(i*num_per_img+j), :]
                    before_feature = features[:, int((i-1)*num_per_img+j), :]
                    after_feature = features[:, int((i-2)*num_per_img+j), :]
                pro_feature1 = self.box_attn(now_feature, before_feature, now_feature)[0]
                pro_feature2 = self.box_attn(now_feature, after_feature, now_feature)[0]
                now_feature = now_feature + self.dropout_box(pro_feature1) + self.dropout_box(pro_feature2)
                new_features.append(now_feature)
        new_features = torch.stack(new_features).permute(1, 0, 2).to(features.device)
        return new_features

    
    def cross_all_img(self, features, N):
        num_per_img = int(features.shape[1]/N)
        new_features = []
        for i in range(N):
            for j in range(num_per_img):
                if i == 0:
                    now_feature = features[:, int(i*num_per_img+j), :]
                    before_feature = features[:, int((i+1)*num_per_img+j), :]
                    after_feature = features[:, int((i+2)*num_per_img+j), :]
                elif i == N-1:
                    now_feature = features[:, int(i*num_per_img+j), :]
                    before_feature = features[:, int((i-1)*num_per_img+j), :]
                    after_feature = features[:, int((i-2)*num_per_img+j), :]
                else:
                    now_feature = features[:, int(i*num_per_img+j), :]
                    before_feature = features[:, int((i-1)*num_per_img+j), :]
                    after_feature = features[:, int((i-2)*num_per_img+j), :]
                pro_feature1 = self.box_attn(now_feature, before_feature, now_feature)[0]
                pro_feature2 = self.box_attn(now_feature, after_feature, now_feature)[0]
                now_feature = now_feature + self.dropout_box(pro_feature1) + self.dropout_box(pro_feature2)
                new_features.append(now_feature)
        new_features = torch.stack(new_features).permute(1, 0, 2).to(features.device)
                # features[:, int(i*num_per_img+j), :] = now_feature
        return new_features        


    
    def feature_fusion(self, features, ref_feas):
        # multi head attention between the feature maps 1-2-3-4-5, now I try to enhance the feature by use 3 pictures

        # for feature_layer in features:
        b, c, h, w = features[-1].shape
        #首先把图像按照batch分开, 并与参考帧特征进行合并
        new_batchs = []
        for batch in range(b):
            now_batch = []
            for i in range(len(features)):
                now_feature = features[i][batch].unsqueeze(0).transpose(1,0) #[256, 1, x, x]
                ref_feature = ref_feas[i][batch] #[256, 4, x, x]
                now_batch.append(torch.cat([now_feature, ref_feature],dim=1))
            new_batchs.append(now_batch)
        
        #进行特征注意力计算
        att_batchs = []
        for batch in new_batchs:
            att_batch = []
            for layer in batch:
                B, C, H, W = layer.shape
                att_batch.append(self.pre_fusion(layer.view(B*C, H, W)).view(B, C, H, W))
            att_batchs.append(att_batch)
        
        #最后再把这些还原
        features = []
        ref_feas = []
        for att_batch in att_batchs:
            sub_fea = []
            sub_ref = []
            for layer in att_batch:                
                sub_fea.append(layer[:,0])
                sub_ref.append(layer[:,1:])
            features.append(sub_fea)
            ref_feas.append(sub_ref)
        new_feature, new_ref = [], []
        if len(ref_feas) > 1:
            for i in range(len(features[0])):
                cat_feature = []
                cat_ref = []
                for j in range(len(ref_feas)):
                    cat_feature.append(features[j][i])
                    cat_ref.append(ref_feas[j][i])
                new_feature.append(torch.stack(cat_feature).squeeze())
                new_ref.append(torch.stack(cat_ref).squeeze())
        else:
            new_feature = features[0]
            new_ref = ref_feas[0]
            for i in range(len(new_feature)):
                new_feature[i] = new_feature[i].unsqueeze(0)
                new_ref[i] = new_ref[i].unsqueeze(0)

        return new_feature, new_ref

                    
    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
    
    def ela_enhanced(self, ori, refs):
        '''
        拼接并增强信息, 最终融合.
        '''
        refs =torch.stack(refs)
        all_rois = torch.cat([ori.unsqueeze(0), refs]).permute(1,2,0,3,4)
        enhanced_rois = self.layer_attention(all_rois).permute(2,0,1,3,4)
        return enhanced_rois[0]

    


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC #64
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC #2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2) # [49, 4500, 256] --> [4500, 49, 256]
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2) # [1, 4500, 256] --> [4500, 1, 32768]

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1) #特征相乘，从而聚合roi特征与增强特征，使网络关注度在关键特征上
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features
    
class GALayer(nn.Module):
    '''
    This is the global attention module which inspared by the FFA-net
    It contains a average pooling, a channel attention and a pixel attention.
    '''
    def __init__(self, cfg):
        super(GALayer, self).__init__()
        self.channel = (cfg.MODEL.DiffusionDet.REF_NUM + 1) *256
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(self.channel, self.channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channel // 8, self.channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.pa = nn.Sequential(
                nn.Conv2d(self.channel, self.channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.avg_pool(x)
        x1 = x1 * self.ca(x1)
        x1 = x1 * self.pa(x1)
        return x + x1


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)

class ELA_Layer(nn.Module):
    '''
    This is the Efficient Local Attention, 
    '''
    def __init__(self, cfg, reduction=32, kernel_size=3):
        super(ELA_Layer, self).__init__()
        in_ch = cfg.MODEL.DiffusionDet.IN_CH
        out_ch = cfg.MODEL.DiffusionDet.OUT_CH
        self.pool_d = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, None))
        self.pool_w = nn.AdaptiveAvgPool3d((None, None, 1))
        pad = kernel_size // 2
        self.conv_w = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, groups=in_ch, bias=False)
        self.conv_h = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, groups=in_ch, bias=False)
        self.conv_d = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, groups=in_ch, bias=False)
        self.GN = nn.GroupNorm(reduction, in_ch)
        self.act = h_swish()
    
    def forward(self, x):
        b,c,d,h,w = x.shape
        identity = x

        x_w = self.pool_w(x).view(b,c,d,h)
        x_h = self.pool_h(x).view(b,c,d,w)
        x_d = self.pool_d(x).view(b,c,h,w)

        x_w = self.conv_w(x_w)
        x_h = self.conv_h(x_h)
        x_d = self.conv_d(x_d)

        x_w = self.GN(x_w)
        x_h = self.GN(x_h)
        x_d = self.GN(x_d)

        x_w = self.act(x_w).view(b,c,d,h,1)
        x_h = self.act(x_h).view(b,c,d,1,w)
        x_d = self.act(x_d).view(b,c,1,h,w)

        x = identity * x_w * x_d * x_h
        return x

class con_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(con_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x    

class Window_Layer(nn.Module):
    def __init__(self, img_ch=3, out_ch=2):
        super(Window_Layer, self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.conv1 = con_block(img_ch, filters[0], 3)
        self.conv2 = con_block(filters[0], filters[0], 3)
        self.conv3 = con_block(filters[0], filters[1], 3)
        self.conv4 = con_block(filters[1], filters[1], 3)
        self.fc1 = nn.Linear(filters[3]*2, filters[2])
        self.fc2 = nn.Linear(filters[2], filters[0])
        self.fc3 = nn.Linear(filters[0], out_ch)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=4)
        self.conv5 = con_block(filters[1], filters[2], 3)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=3, stride=4)

    def forward(self,x):
        x = self.conv1(x)

        x = self.Maxpool1(x)
        x = self.conv2(x)

        x = self.Maxpool2(x)
        x = self.conv3(x)

        x = self.Maxpool3(x)
        x = self.conv4(x)

        x = self.Maxpool4(x)
        x = self.conv5(x)

        x = self.Maxpool5(x)

        b,c,h,w = x.shape
        x = self.fc1(x.view(b,c*h*w))
        x = self.fc2(x)
        x = self.fc3(x)
        return x
