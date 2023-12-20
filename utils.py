# -*- coding:utf-8 -*-
###
# File: e:\G\图像\LiTS\utils.py
# Project: e:\G\图像\LiTS
# Created Date: Saturday, October 7th 2023, 3:00:23 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2023-11-01 18:06:25
# Modified By: Xinyu Li
# -----
# Copyright (c) 2023 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
import numpy as np
import os
from scipy.ndimage import label, find_objects
from shapely.geometry import Polygon
from skimage import measure

def range_normalization(data, rnge=(0, 1), per_channel=True, eps=1e-8):
    data_normalized = np.zeros(data.shape, dtype=data.dtype)
    for b in range(data.shape[0]):
        if per_channel:
            for c in range(data.shape[1]):
                data_normalized[b, c] = min_max_normalization(data[b, c], eps)
        else:
            data_normalized[b] = min_max_normalization(data[b], eps)

    data_normalized *= (rnge[1] - rnge[0])
    data_normalized += rnge[0]
    return data_normalized

def crop_img_by_label(image: np.array, label: np.array, extend_slice: int =0, only_z: bool =True) -> np.array:
    '''
    根据label对图像进行裁切，去除纯背景部分，防止样本极度不均衡\n
    !!! IMPORTANT!!!
    If the input is 2-D, please make sure that the shape of label is w*h, not c*w*h
    !!! IMPORTANT!!!
    input: A image and a label with type of np.array, can be 2-D or 3-D
           Extend index (optional), default=0, add some pure background.
           only_z (optional), default=True, in 3-D images, we only crop the z-axis
    return: The cropped image and label as np.array
    '''
    mask_range = np.nonzero(label)
    demension = len(label.shape)
    if demension == 2:
        image_left = mask_range[0][0] - extend_slice if mask_range[0][0] >= extend_slice else 0
        image_top = mask_range[1][0] - extend_slice if mask_range[1][0] >= extend_slice else 0
        image_right = mask_range[0][-1] + extend_slice if mask_range[0][-1] + extend_slice <= label.shape[0] - 1 else label.shape[0] - 1
        image_bottom = mask_range[-1][-1] + extend_slice if mask_range[-1][-1] + extend_slice <= label.shape[-1] - 1 else label.shape[-1] - 1
        new_img = image[:, image_left:image_right+1, image_top:image_bottom+1] if len(image.shape) == 3 else image[image_left:image_right+1, image_top:image_bottom+1]
        new_label = label[image_left:image_right+1, image_top:image_bottom+1]
    elif demension == 3:
        z_start, z_end = mask_range[0][0] - extend_slice if mask_range[0][0] >= extend_slice else 0, mask_range[0][-1] + extend_slice if mask_range[0][-1] + extend_slice <= label.shape[0] - 1 else label.shape[0] - 1
        x_start, x_end = mask_range[1][0] - extend_slice if mask_range[1][0] >= extend_slice else 0, mask_range[1][-1] + extend_slice if mask_range[1][-1] + extend_slice <= label.shape[1] - 1 else label.shape[1] - 1
        y_start, y_end = mask_range[2][0] - extend_slice if mask_range[2][0] >= extend_slice else 0, mask_range[2][-1] + extend_slice if mask_range[2][-1] + extend_slice <= label.shape[2] - 1 else label.shape[2] - 1
        new_img = image[z_start:z_end+1, :, :] if only_z else image[z_start:z_end+1, x_start:x_end+1, y_start:y_end+1]
        new_label = label[z_start:z_end+1, :, :] if only_z else label[z_start:z_end+1, x_start:x_end+1, y_start:y_end+1]
    
    return new_img, new_label

def split_label(label):
    '''
    split the mask in different labels, such as [0, 1, 2, 3, 1, 2] --> [0, 1, 0, 0, 1, 0], [0, 0, 2, 0, 0, 2], [0, 0, 0, 3, 0, 0]
    input: the multi-labeled mask
    ouput: a list of single-labeled mask
    '''
    unique_label = np.unique(label).tolist()
    unique_label.remove(0)
    split_label = []
    for item in unique_label:
        tmp_label = np.zeros_like(label)
        tmp_label[label==item]=item
        split_label.append(tmp_label)
    return split_label

def remove_small_island(mask):
    return None
        
def count_connected_components_2D(mask):
    label_array, _ = label(mask)
    cc_edge = find_objects(label_array) if label_array.max() != 0 else []
    largest_cc = np.argmax(np.bincount(label_array.ravel())[1:])+1 if label_array.max() != 0 else 0  #np.bincount统计一维数组中，从0开始各个数字出现的次数。
    return cc_edge, largest_cc

def slice_to_xywh(x_slice, y_slice, padding=2):
    x_start, x_end = x_slice.start, x_slice.stop
    y_start, y_end = y_slice.start, y_slice.stop
    return [x_start-padding, y_start-padding, x_end-x_start+padding, y_end-y_start+padding]

def mask_to_bbox(mask):
    split_mask = split_label(mask)
    bbox_list = []
    for labels in split_mask:
        for slice in labels:
            edge, _ = count_connected_components_2D(slice)
            cnt = len(edge) if edge != [] else 0
            sub_bbox_list = [slice_to_xywh(edge[i][0], edge[i][1]) for i in range(0, cnt)] if cnt != 0 else []
            bbox_list.append(sub_bbox_list)
    return bbox_list

def bbox_merge(bbox_list):
    return None

def mask_to_coco(mask):
    split_mask = split_label(mask)
    mask_list = []
    for labels in split_mask:
        if split_mask.index(labels) == 0:
            labels[mask>0]=1
        for slice in labels:
            contour = measure.find_contours(slice)
            poly = Polygon(contour)
            poly = poly.simplify(1.0, preserve_topology=False)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            mask_list.append(segmentation)
    return mask_list

    
def min_max_normalization(data, eps):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range

    return data_normalized

def zero_mean_unit_variance_normalization(data, per_channel=True, epsilon=1e-8):
    data_normalized = np.zeros(data.shape, dtype=data.dtype)
    for b in range(data.shape[0]):
        if per_channel:
            for c in range(data.shape[1]):
                mean = data[b, c].mean()
                std = data[b, c].std() + epsilon
                data_normalized[b, c] = (data[b, c] - mean) / std
        else:
            mean = data[b].mean()
            std = data[b].std() + epsilon
            data_normalized[b] = (data[b] - mean) / std
    return data_normalized

def cut_and_normalization(data, percentile_lower=0.5, percentile_upper=99.5, ):
    '''
    这个函数结合了cut_off_outliers和mean_std_normalization两个函数，并且实现了自动计算mean和std，使用的是per_channel=False的版本
    '''
    data_normalized = np.zeros(data.shape, dtype=np.float32)
    for b in range(len(data)):
        cut_off_lower = np.percentile(data[b], percentile_lower)
        cut_off_upper = np.percentile(data[b], percentile_upper)
        data[b][data[b] < cut_off_lower] = cut_off_lower
        data[b][data[b] > cut_off_upper] = cut_off_upper
        mean = np.mean(data[b])
        std = np.std(data[b])
        data_normalized[b]  = (data[b] - mean) / std
    return data_normalized

def maybe_make_dir(path: str) -> None:
    '''
    判断是否存在文件夹，不存在就新建一个。
    '''
    os.makedirs(path, exist_ok=True)