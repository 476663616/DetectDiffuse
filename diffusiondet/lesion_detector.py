# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/diffusiondet/detector copy.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet/diffusiondet
# Created Date: Tuesday, January 2nd 2024, 11:06:50 am
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-05-05 18:03:18
# Modified By: Xinyu Li
# -----
# Copyright (c) 2024 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###

import math
import random
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .lesion_head import Lesion_DynamicHead, Window_Layer
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list

__all__ = ["LesionDiffusionDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t) #获取第t个时间步对应的余弦采样值
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64) #torch.linspace(start, end, steps=100, out=None, dtype=None,layout=torch.strided, device=None, requires_grad=False)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

@META_ARCH_REGISTRY.register()
class LesionDiffusionDet(nn.Module):
    """
    Implement DiffusionDet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # build diffusion
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True


        ###########################
        #### Added by Xinyu Li ####
        ###########################
        self.cfg = cfg
        self.is_lesion = cfg.MODEL.DiffusionDet.IS_LESION
        self.thr = cfg.MODEL.DiffusionDet.KEEP_THRESHOLD
        self.noise = cfg.MODEL.DiffusionDet.NOISE
        self.test_visual = False
        self.naborhood_transformer = True
        self.adaptive_window = cfg.MODEL.DiffusionDet.ADAPTIVE_WINDOW
        if self.adaptive_window == True:
            self.adapter = Window_Layer(3, 3).cuda()
            self.windows = cfg.MODEL.DiffusionDet.WINDOWS

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        if not self.is_lesion:
            self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        else:
            self.head = Lesion_DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        # Loss parameters:
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.antinormalizer = lambda x: x * pixel_std + pixel_mean
        self.to(self.device)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, ref_feas, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.head(backbone_feats, ref_feas, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = box_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord
    
    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filterpip
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                # img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
                img = torch.cat((img, torch.randn(batch, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    @torch.no_grad()
    def lesion_sample(self, batched_inputs, backbone_feats, ref_feas, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, ref_feas, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                # img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
                img = torch.cat((img, torch.randn(batch, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    # forward diffusion
    #来自DDPM的噪声扩散过程
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape) #获取当前生成时间步对应的α余弦采样值
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) #获取当前生成时间步对应的β余弦采样值

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise #返回Sqrt(α(t))*x + Sqrt(1-α(t))*x_noise
    
    def uniform_noise(self, x_boxes, noise = None, t = None):
        batch = x_boxes.shape[0]
        if self.training:
            for i in range(0, batch):
                x_boxes[i,] = x_boxes[0,]
                noise[i, ] = noise[0,]
                t[i] = t[0]
            return x_boxes, noise, t
        else:
            for i in range(0, batch):
                x_boxes[i,] = x_boxes[0,]
            return x_boxes

    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        if self.test_visual:
            self.test_visualization()
        images, ref_images, images_whwh = self.preprocess_image(batched_inputs)  #对图像进行预处理，包括对图像像素进行归一化，同时获取图像的尺寸信息
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)  #backbone,特征提取 [B, C, W, H]: [9, 3, 512, 512] --> [9, 256, 128, 128] --> [9, 256, 64, 64, 64] --> [9, 256, 32, 32, 32] --> [9, 256, 16, 16, 16] --> [9, 256, 8, 8, 8] 对应p1, p2, p3, p4, p5, p6
        features = list()
        for f in self.in_features: #将所需几个阶段的特征提取出来
            feature = src[f]
            features.append(feature)
        
        # Reference feature 组织格式为: 一个列表, 列表内部按照一个batch内所含的数据个数划分为若干子列表, 每个子列表包含由resnet50提取的不同层级的特征图, [b, c, w, h], b = 4 

        ref_feas = []
        for ref in ref_images:
            ref_fea = self.backbone(ref.tensor)
            ref_fea_list = []
            for f in self.in_features:
                feature = ref_fea[f].transpose(1,0)
                ref_fea_list.append(feature)
            ref_feas.append(ref_fea_list)

        new_feas = []
        for j in range(len(self.in_features)):
            sub_feas = []
            for i in range(len(ref_feas)):
                sub_feas.append(ref_feas[i][j])
            new_feas.append(torch.stack(sub_feas))
        del ref_feas

        # Prepare Proposals.
        if not self.training:
            if not self.is_lesion:
                results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            else:
                results = self.lesion_sample(batched_inputs, features, new_feas, images_whwh, images)
            return results

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances) #根据金标准生成对应的噪声框

            t = t.squeeze(-1)
            x_boxes = x_boxes * images_whwh[:, None, :] #将归一化后的噪声金标准框复原

            outputs_class, outputs_coord = self.head(features, new_feas, x_boxes, t, None) #检测头，输入特征与噪声框，输出框的分类以及坐标featurs:[9, 256, x, x] *4, x_boxes:[9, 500, 4] output: [6, 9, 500, 4], 因为有6个检测头，所以输出的第一维是6

            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            if self.deep_supervision: 
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long() #随机生成一个时间步——整数
        if self.noise == 'normal':
            noise = torch.randn(self.num_proposals, 4, device=self.device) #随机生成服从标准正态分布的检测框
        elif self.noise == 'poisson':
            noise = torch.rand(self.num_proposals, 4)*512.#生成0~512的随机数
            noise = torch.poisson(noise)/512. #随机生成服从泊松分布的检测框,同时进行归一化。
            noise = noise.to(device=self.device)
        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6 ？？？？？？
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4) # 长宽小于0.0001的框被删除
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale  #这里的Scale指的是SNR信噪比。为什么选择2在文中4.4小节有详细的阐述

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise) #生成真正的噪声采样框

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale) #将噪声采样框进行标准化
        x = ((x / self.scale) + 1) / 2. #497行的逆过程，不过为什么要逆一下呢？？？又为什么要变化过去呢？——也是来自前人的工作

        diff_boxes = box_cxcywh_to_xyxy(x) #框的坐标cxcywh-->xyxy 之前变过去主要是为了能够更好地进行噪声采样，wh在尺度上相对xy更小

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size # 512, 512
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy # 把框的坐标按照图像增强后的尺寸进行归一化
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes) #框的坐标xyxy-->cxcywh
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes) #输入金标准检测框，返回噪声框、经过噪声采样后的金标准框、噪声时间步
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device)
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def stenosis_inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
        else:
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            result = Instances(image_size)
            scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
            labels_per_image = labels[topk_indices]
            box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
            box_pred_per_image = box_pred_per_image[topk_indices]

            # if self.use_ensemble and self.sampling_timesteps > 1:
            #     return box_pred_per_image, scores_per_image, labels_per_image

            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result.pred_boxes = box_pred_per_image
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results
    

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        ref_images = [[self.normalizer(ref_img.to(self.device)) for ref_img in x['ref_image'][0]] for x in batched_inputs]
        # self.visualization(images)

        if self.adaptive_window:
            images_tensor = torch.stack(images)
            s1 = nn.Softmax(1)
            n_prob = self.adapter(images_tensor)  if images_tensor.shape[2] == 512 else self.adapter(F.interpolate(images_tensor.unsqueeze(0),size=(512), mode='bilinear', align_corners=False))
            n_prob = s1(n_prob)
            fit_n_idx = torch.argmax(n_prob, -1)
            n_window = [self.windows[x] for x in fit_n_idx]
            r_window = []
            for batch in ref_images:
                sub_prob = [self.adapter(x.unsqueeze(0)) for x in batch] if batch[0].shape[1] == 512 else [self.adapter(F.interpolate(x.unsqueeze(0),size=(512), mode='bilinear', align_corners=False)) for x in batch]
                sub_prob = [s1(x) for x in sub_prob]
                sub_window = [self.windows[torch.argmax(x)] for x in sub_prob]
                r_window.append(sub_window)
            
            images, ref_images = self.window_adjust(n_window, r_window, images_tensor, ref_images)

                
        images = ImageList.from_tensors(images, self.size_divisibility)
        ref_images = [ImageList.from_tensors(x, self.size_divisibility) for x in ref_images]

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, ref_images, images_whwh
    
    def visualization(self,img):
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        test_img_save_path = './datasets/test_save/test.png'
        img = img.cpu().detach().numpy().transpose(1, 2, 0)
        cv2.imwrite(test_img_save_path,img)
    
    def test_visualization(self):
        import cv2
        import numpy as np
        import os
        from matplotlib import pyplot as plt
        test_img_path = './test_img/'
        test_img = [cv2.imread(test_img_path+img).transpose(2,0,1) for img in os.listdir(test_img_path) if 'png' in img]
        test_img = torch.tensor(np.array(test_img)).float().to(self.device)
        src = self.backbone(test_img)  #backbone,特征提取 [B, C, W, H]: [9, 3, 512, 512] --> [9, 256, 128, 128] --> [9, 256, 64, 64, 64] --> [9, 256, 32, 32, 32] --> [9, 256, 16, 16, 16] --> [9, 256, 8, 8, 8] 对应p1, p2, p3, p4, p5, p6
        features = list()
        for f in self.in_features: #将所需几个阶段的特征提取出来
            feature = src[f]
            features.append(feature)
        v = features[0][0]
        v = v.data.detach().cpu()
        channel_num = 0
        plt.figure(figsize=(16, 16))
        plt.imshow(v[channel_num, :, :])
        channel_num = random_num(9,v.shape[0])
        plt.figure(figsize=(16, 16))
        for index, channel in enumerate(channel_num):
            ax = plt.subplot(3, 3, index+1,)
            plt.imshow(v[channel, :, :])
        plt.show()
        print()

    def cut_window(self, img, min, max):
        thr_img = torch.zeros_like(img)
        c, x, y = img.shape
        # self.visualization(img)
        img = self.antinormalizer(img)
        for i in range(c):
            c_max, c_min = torch.max(img[i]), torch.min(img[i])
            scale_factor = c_max/(max-min)
        #     for j in range(x):
        #         for k in range(y):
        #             thr_img[i,j,k] = torch.min(c_max, (img[i,j,k]-min)/(max-min)*c_max)
            thr_img[i] = torch.clip(img[i], min, max)
            thr_img[i] = (thr_img[i] - min) / (max - min)
            thr_img[i] = thr_img[i] * (c_max + c_min) + c_min
        # self.visualization(thr_img)
        thr_img = self.normalizer(thr_img)
        return thr_img


    def window_adjust(self, n_window, r_window, images, ref_images):
        '''
        调整图像的窗宽与窗位👽👽👽☑️☑️☑️✔️✅❎💹
        '''
        b, r = len(r_window), len(r_window[0])
        new_images = []
        new_ref = []
        for i in range(b):
            [n_min, n_max] = n_window[i]
            tmp_window = torch.tensor([[0, 0]]).to(self.device)
            # tmp_window = [0, 0]
            for j in range(r):
                tmp_window += torch.tensor(r_window[i][j]).to(self.device)
            [r_min, r_max] = tmp_window[0]/r
            w_min, w_max = (n_min+r_min)/2, (n_max+r_max)/2
            image = self.cut_window(images[i], w_min, w_max)
            sub_new_ref = [self.cut_window(x, w_min, w_max) for x in ref_images[i]]
            # self.visualization(self.antinormalizer(image))
            new_images.append(image)
            new_ref.append(sub_new_ref)
        # new_images = torch.stack(new_images)
        return new_images, new_ref
