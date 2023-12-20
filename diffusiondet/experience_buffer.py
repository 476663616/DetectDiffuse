'''
File: /home/xinyul/python_exercises/DiffusionDet/diffusiondet/experience_buffer.py
Project: /home/xinyul/python_exercises/DiffusionDet/diffusiondet
Created Date: Wednesday, July 5th 2023, 3:13:06 pm
Author: pangpang li
-----
Last Modified: 2023-07-22 20:54:25
Modified By: pangpang li
-----
Copyright (c) 2023 Beijing Institude of Technology.
------------------------------------
Have a nice day!
'''

import torch
import torch.nn as nn
import math
from typing import List
import torch
import numpy as np
from collections import Counter

from detectron2.layers import ROIAlign, nonzero_tuple
from detectron2.structures import Boxes
from detectron2.utils.tracing import assert_fx_safe, is_fx_tracing
from detectron2.modeling.poolers import _create_zeros, convert_boxes_to_pooler_format, assign_boxes_to_levels 


def compute_iou_tensor(bbox, box_list):
    cnt = len(box_list)
    for ori_bbox in box_list:
        S_rec1 = (ori_bbox[2] - ori_bbox[0]) * (ori_bbox[3] - ori_bbox[1])
        S_rec2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(ori_bbox[1], bbox[1])
        right_line = min(ori_bbox[3], bbox[3])
        top_line = max(ori_bbox[0], bbox[0])
        bottom_line = min(ori_bbox[2], bbox[2])
        intersect = (right_line - left_line) * (bottom_line - top_line)

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            cnt -= 1
            # return 0
        elif (intersect / (sum_area - intersect)) * 1.0 < 0.01:
            cnt -= 1
    if cnt == 0:
        return 0
    else:
        
        return 1



class Stenosis_Buffer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.threshold = cfg.MODEL.DiffusionDet.KEEP_THRESHOLD
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        self.buffer = self.init_memory()

    def init_memory(self):
        memory_buffer = []
        for i in range(self.num_heads):
            buffer_dict = {'features': [],
                        'bboxes': [],
                        'scores': []}
            memory_buffer.append(buffer_dict)
        return memory_buffer
    
    def update_memory(self, i, features, bboxes, scores):
        self.buffer[i]['features'].append(features)
        self.buffer[i]['bboxes'].append(bboxes)
        self.buffer[i]['scores'].append(scores)
    
    def filter_and_expand(self, thr=0.5):
        output_buffer = self.buffer[-1]
        output_scores = output_buffer['scores'][0]
        self.device = output_scores.device
        output_bboxes = output_buffer['bboxes'][0]
        prob_scores = output_scores.sigmoid()
        filter_idx = torch.argwhere(prob_scores>=thr).detach().cpu().numpy().tolist()
        self.filter_scores = []
        self.filter_bboxes = []
        for i in range(output_scores.shape[0]):
            sub_filter_scores = [prob_scores[x[0], x[1], x[2]] for x in filter_idx if x[0]==i]
            sub_filter_bboxes = [output_bboxes[x[0], x[1], :] for x in filter_idx if x[0]==i]
            sub_filter_scores, sub_filter_bboxes = self.del_same_bbox(sub_filter_scores, sub_filter_bboxes)
            self.filter_scores.append(sub_filter_scores)
            self.filter_bboxes.append(sub_filter_bboxes)
        
        num_loss, wrong_number, wrong_idx = self.cal_num_loss(self.filter_bboxes)
        excessive_idx = [wrong_idx[x] for x in range(len(wrong_idx)) if wrong_number[x]>0]

        if not wrong_number == []:
            self.expend_bbox(self.filter_bboxes, wrong_idx, wrong_number)
            return True, num_loss, excessive_idx

        else:
            return False, num_loss, excessive_idx
    
    def cal_num_loss(self, filter_bboxes):
        len_x = [len(x) for x in filter_bboxes]
        if sum(len_x) == 0:
            sum_dif = torch.as_tensor(9., device=self.device)
            sum_dif.requires_grad_(True)
            return 1/len(len_x)*sum_dif, [], []
        #test
        # len_x = [1, 1, 2, 3, 1, 1, 1, 1, 2]
        wrong_idx, right_num = self.vote_the_max(len_x)
        dif_x = [len_x[x]-right_num for x in wrong_idx]
        
        abs_dif_x = np.abs(dif_x)
        sum_dif = torch.as_tensor(np.sum(abs_dif_x) * 1., device=self.device)
        sum_dif.requires_grad_(True)
        # len_x = np.asarray(len_x)
        # dif_x = np.abs(np.diff(len_x))
        # sum_dif = np.sum(dif_x)/2
        return 1/len(len_x)*sum_dif, dif_x, wrong_idx
    
    def find_nearest_right_idx(self, idx, wrong_idx):
        for i in range(9):
            new_idx_1 = idx + i
            new_idx_2 = idx - i
            if not new_idx_1 in wrong_idx and new_idx_1 <= 8:
                return new_idx_1
            elif not new_idx_2 in wrong_idx and new_idx_2 >= 0:
                return new_idx_2
        
    
    def find_which_excessive(self, wrong_idx, filter_bboxes):
        tmp_excessive_dict = []
        excessive_dict = []
        for idx in wrong_idx:
            # excessive_dict[str(idx)] = []
            nri = self.find_nearest_right_idx(idx, wrong_idx)
            right_boxes = filter_bboxes[nri]
            for bbox in filter_bboxes[idx]:
                if compute_iou_tensor(bbox, right_boxes) <= 0.01:
                    tmp_excessive_dict.append(bbox)
        for box in tmp_excessive_dict:
            if excessive_dict == []:
                excessive_dict.append(box)
            else:
                if compute_iou_tensor(box, excessive_dict) >= 0.1:
                    continue
                else:
                    excessive_dict.append(box)
        return excessive_dict


    def vote_the_max(self, bbox_number):#按照少数服从多数的规则，通过投票选出本序列图像应有的狭窄数，并筛出出错的帧，若缺少，则补roi，若多余，则查找多余的roi并在其它图像上再次检测。
        number_item = Counter(bbox_number)
        right_num = [x for x in number_item.keys()][[x for x in number_item.values()].index(max([x for x in number_item.values()]))] #通过投票法决定正确的检测框数量
        if right_num == 0:
            new_number_item = number_item
            new_number_item.pop(0)
            right_num = [x for x in new_number_item.keys()][[x for x in new_number_item.values()].index(max([x for x in new_number_item.values()]))]
        wrong_idx = [i for i in range(len(bbox_number)) if bbox_number[i] != right_num]
        return wrong_idx, right_num
            
    
    def del_same_bbox(self, sub_filter_scores, sub_filter_bboxes):
        tmp_filter_bboxes = []
        tmp_filter_scores = []
        ####test####
        # sub_filter_bboxes = [[247.9984, 162.0222, 276.0217, 218.0125],
        #                      [337.2237,  99.3132, 366.8825, 122.1607],
        #                      [319.9865,  81.8731, 354.0383,  99.3314],
        #                      [319.9865,  81.8731, 354.0383,  99.3314],
        #                      [319.9865,  81.8731, 354.0383,  99.3314]]
        ############
        for i in range(len(sub_filter_bboxes)):
            if tmp_filter_bboxes == []:
                tmp_filter_bboxes.append(sub_filter_bboxes[i])
                tmp_filter_scores.append(sub_filter_scores[i])
            else:
                if compute_iou_tensor(sub_filter_bboxes[i], tmp_filter_bboxes) >= 0.1:
                    continue
                else:
                    tmp_filter_bboxes.append(sub_filter_bboxes[i])
                    tmp_filter_scores.append(sub_filter_scores[i])
        return tmp_filter_scores, tmp_filter_bboxes
    
    def expend_bbox(self, filter_bboxes, wrong_idx, wrong_number):
        self.expend_bboxes = []
        self.excessive_dict = []
        if min(wrong_number) < 0:#如果某帧缺少框，就找正确的框，并把它的检测结果延伸后作为新的 roi
            for i in range(len(filter_bboxes)):
                if i not in wrong_idx:
                    for bbox in filter_bboxes[i]:
                        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                        if w >= h:
                            h_expand = (w-h)/2+20
                            expend_arr = torch.tensor([-20, -h_expand, 20, h_expand], device=bbox.device)
                            self.expend_bboxes.append(bbox+expend_arr)
                        else:
                            w_expand = (h-w)/2+20
                            expend_arr = torch.tensor([-w_expand, -20, w_expand, 20], device=bbox.device)
                            self.expend_bboxes.append(bbox+expend_arr)
                    break
        if max(wrong_number) > 0:
            # wrong_img = filter_bboxes[wrong_idx[wrong_number.index(max(wrong_number))]]
            # for bbox in wrong_img:
            #     w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
            #     if w >= h:
            #         h_expand = (w-h)/2+20
            #         expend_arr = torch.tensor([-20, -h_expand, 20, h_expand], device=bbox.device)
            #         self.expend_bboxes.append(bbox+expend_arr)
            #     else:
            #         w_expand = (h-w)/2+20
            #         expend_arr = torch.tensor([-w_expand, -20, w_expand, 20], device=bbox.device)
            #         self.expend_bboxes.append(bbox+expend_arr)
            self.excessive_dict = self.find_which_excessive(wrong_idx, filter_bboxes)
            for i in range(len(self.excessive_dict)):
                bbox = self.excessive_dict[i]
                w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
                if w >= h:
                    h_expand = (w-h)/2+20
                    expend_arr = torch.tensor([-20, -h_expand, 20, h_expand], device=bbox.device)
                    self.excessive_dict[i] = (bbox+expend_arr)
                else:
                    w_expand = (h-w)/2+20
                    expend_arr = torch.tensor([-w_expand, -20, w_expand, 20], device=bbox.device)
                    self.excessive_dict[i] = (bbox+expend_arr)
class ROIPooler(nn.Module):
    """
    Region of interest feature map pooler that supports pooling from one or more
    feature maps.
    """

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
        """
        Args:
            output_size (int, tuple[int] or list[int]): output size of the pooled region,
                e.g., 14 x 14. If tuple or list is given, the length must be 2.
            scales (list[float]): The scale for each low-level pooling op relative to
                the input image. For a feature map with stride s relative to the input
                image, scale is defined as 1/s. The stride must be power of 2.
                When there are multiple scales, they must form a pyramid, i.e. they must be
                a monotically decreasing geometric sequence with a factor of 1/2.
            sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
            pooler_type (string): Name of the type of pooling operation that should be applied.
                For instance, "ROIPool" or "ROIAlignV2".
            canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
                is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
                pre-training).
            canonical_level (int): The feature map level index from which a canonically-sized box
                should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
                i.e., a box of size 224x224 will be placed on the feature with stride=16.
                The box placement for all boxes will be determined from their sizes w.r.t
                canonical_box_size. For example, a box whose area is 4x that of a canonical box
                should be used to pool features from feature level ``canonical_level+1``.

                Note that the actual input feature maps given to this module may not have
                sufficiently many levels for the input boxes. If the boxes are too large or too
                small for the input feature maps, the closest level will be used.
        """
        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size
        self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )


        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -(math.log2(scales[0]))
        max_level = -(math.log2(scales[-1]))
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 <= self.min_level and self.min_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size
    
    # def extension_roi(self, box_lists):
    #     for box_list in box_lists:
            


    def forward(self, x: List[torch.Tensor], box_lists: List[Boxes]):
        """
        Args:
            x (list[Tensor]): A list of feature maps of NCHW shape, with scales matching those
                used to construct this module.
            box_lists (list[Boxes] | list[RotatedBoxes]):
                A list of N Boxes or N RotatedBoxes, where N is the number of images in the batch.
                The box coordinates are defined on the original image and
                will be scaled by the `scales` argument of :class:`ROIPooler`.

        Returns:
            Tensor:
                A tensor of shape (M, C, output_size, output_size) where M is the total number of
                boxes aggregated over all N batch images and C is the number of channels in `x`.
        """

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        # box_lists = self.extension_roi(box_lists)

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        output = _create_zeros(pooler_fmt_boxes, num_channels, output_size, output_size, x[0])

        for level, pooler in enumerate(self.level_poolers):
            inds = nonzero_tuple(level_assignments == level)[0]
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            # Use index_put_ instead of advance indexing, to avoid pytorch/issues/49852
            output.index_put_((inds,), pooler(x[level], pooler_fmt_boxes_level))

        return output