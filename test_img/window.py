# -*- coding:utf-8 -*-
###
# File: e:\G\图像\DeepLesion1.0\test\window.py
# Project: e:\G\图像\DeepLesion1.0\test
# Created Date: Thursday, December 21st 2023, 11:27:32 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2023-12-28 17:00:54
# Modified By: Xinyu Li
# -----
# Copyright (c) 2023 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
import cv2
import numpy as np

'''
head and neck
brain W:80 L:40
subdural W:130-300 L:50-100
stroke W:8 L:32 or W:40 L:40 3
temporal bones W:2800 L:600 or W:4000 L:700
soft tissues: W:350–400 L:20–60 4
chest
lungs W:1500 L:-600
mediastinum W:350 L:50
abdomen
soft tissues W:400 L:50
liver W:150 L:30
spine
soft tissues W:250 L:50
bone W:1800 L:400
'''

def adjust_window(img, w_width, w_center):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = img.copy()
    data_adjusted[img < val_min] = val_min
    data_adjusted[img > val_max] = val_max

    return data_adjusted

def cut_window(img, min, max):
    thr_img = np.zeros_like(img)
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            thr_img[i, j] = np.min([255, np.max([0, (img[i, j]-min)/(max-min)*255])])
    return thr_img

def convert_wl_to_minmax(width, center):
    return center-width/2, center+width/2
if __name__ == '__main__':
    img_path = './093.png'
    img = cv2.imread(img_path, -1)
    img = img.astype(np.int32)
    img = img - 32768
    w_width, w_center = 4000, 700
    # img = adjust_window(img, w_width, w_center)
    min, max = convert_wl_to_minmax(w_width, w_center)
    img = cut_window(img, min, max)

    # min = 25
    # max = 160
    # img = cut_window(img, min, max)
    # for i in range(196, 246):
    #     img[500, i] = 255
    #     img[529, i] = 255
    # for j in range(img.shape[1]):
            
    cv2.imwrite('./test_%s_%s.png'%(w_width,w_center), img)