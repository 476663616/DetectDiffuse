# -*- coding:utf-8 -*-
###
# File: h:\D\python_exercises\code\nii2coco.py
# Project: h:\D\python_exercises\code
# Created Date: Thursday, June 20th 2024, 3:52:13 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-10-02 22:55:53
# Modified By: Xinyu Li
# -----
# Copyright (c) 2024 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
import os
import SimpleITK as sitk
import numpy as np
import json
import cv2

label_dict = {}
label_dict[1] = 'all_type'
dataset = dict()
dataset['images'] = []
dataset['type'] = 'instances'
dataset['annotations'] = []
dataset['categories'] = []
dataset['info'] = None
dataset['licenses'] = None

label_dict = {}
label_dict[1] = 'all_type'

for category_id, category_name in label_dict.items():
        category_item = dict()
        category_item['supercategory'] = category_name
        category_item['id'] = category_id
        category_item['name'] = category_name
        dataset['categories'].append(category_item)


def masks_to_bounding_boxes(mask):
    # 查找所有的连通分量（孤立标注）
    num_labels, labels_im = cv2.connectedComponents(mask)
    
    bounding_boxes = []
    for label in range(1, num_labels):  # 从1开始，0是背景
        component_mask = np.uint8(labels_im == label)
        points = cv2.findNonZero(component_mask)
        x, y, w, h = cv2.boundingRect(points)
        bounding_boxes.append((x, y, x + w, y + h))
    
    return bounding_boxes

def seg2coco(seg_arr, slices, img_id, image_cnt, bbox_cnt):
    for ss in slices:
        for s in ss:
            bboxes = masks_to_bounding_boxes(seg_arr[s,:,:].astype(np.int8))
            for bbox in bboxes:
                image = dict()
                image['id'] = image_cnt
                image['file_name'] = img_id.split('.')[0] + '_' + str(s)+'.png'
                image['width'] = seg_arr[s,:,:].shape[0]
                image['height'] = seg_arr[s,:,:].shape[1]
                annotation_item = dict()
                x1, y1, x2, y2 = bbox
                x = x1 
                y = y1 
                w = int(x2-x1) 
                h = int(y2-y1)
                annotation_item['segmentation'] = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                annotation_item['image_id'] = image_cnt
                annotation_item['iscrowd'] = 0
                annotation_item['bbox'] = [x, y, w, h]
                annotation_item['area'] = w * h
                annotation_item['id'] = bbox_cnt
                annotation_item['category_id'] = 1
                bbox_cnt = bbox_cnt+1
                dataset['annotations'].append(annotation_item)
                dataset['images'].append(image)
            if bboxes != []:
                image_cnt = image_cnt + 1
    return image_cnt, bbox_cnt

    

def separate_discontinuous_numbers(numbers):
    if not numbers:
        return []

    # Sort the numbers first to handle unordered input
    numbers.sort()
    
    # Initialize the result list and the first group
    result = []
    current_group = [numbers[0]]

    # Iterate over the numbers and group them
    for i in range(1, len(numbers)):
        if numbers[i] == numbers[i - 1] + 1:
            current_group.append(numbers[i])
        else:
            result.append(current_group)
            current_group = [numbers[i]]

    # Append the last group
    result.append(current_group)

    return result

def expand_slice(slice, tgt_len=3):
    expend_idx = tgt_len - len(slice)
    for i in range(expend_idx):
        slice.append(slice[-1]+1) if i%2!=0 else slice.insert(0, slice[0]-1)
    return slice


def nii2png(img_arr, slices, save_path, img_id):
    for ss in slices:
        if len(ss)<7:
            ss = expand_slice(ss)
        for s in ss:
            s_arr = img_arr[s, :, :]
            # import matplotlib.pyplot as plt
            # plt.imshow(s_arr, cmap='gray')
            # plt.show()
            s_arr = s_arr.astype(np.int16)
            # import imageio
            # imageio.imwrite(save_path+img_id.split('.')[0]+'_'+str(s)+'.png', s_arr)
            # slice_min = s_arr.min()
            # slice_max = s_arr.max()
            # # normalized_slice = 255 * (s_arr - slice_min) / (slice_max - slice_min)
            # normalized_slice = s_arr.astype(np.uint16)
            cv2.imwrite(save_path+img_id.split('.')[0]+'_'+str(s)+'.png', s_arr)


if __name__ == '__main__':
    img_path = 'V:/lesions/LiTS/ori_data/ct/'
    seg_path = 'V:/lesions/LiTS/ori_data/seg/'
    png_save_path = './images/'
    os.makedirs(png_save_path, exist_ok=True)
    coco_save_path = './'
    image_cnt = 0
    bbox_cnt = 0
    for image in os.listdir(img_path):
        if 'volume' in image:
            patient_path = img_path
            img_id = image.split('_')[0]
            img = sitk.ReadImage(patient_path + image)
            seg = sitk.ReadImage(seg_path + image.replace('volume','segmentation'))
            img_arr = sitk.GetArrayFromImage(img)
            seg_arr = sitk.GetArrayFromImage(seg)
            # seg_arr[seg_arr<2]=0
            slices = list(set(np.where(seg_arr == 2)[0].tolist()))
            slices = separate_discontinuous_numbers(slices)
            nii2png(img_arr, slices, png_save_path, img_id)        
            image_cnt, bbox_cnt = seg2coco(seg_arr, slices, img_id, image_cnt, bbox_cnt)
    json.dump(dataset, open(os.path.join(coco_save_path , 'annotation.json'), 'w'))