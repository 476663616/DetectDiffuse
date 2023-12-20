# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/image_normal.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet
# Created Date: Thursday, November 9th 2023, 10:28:52 am
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2023-11-09 20:34:48
# Modified By: Xinyu Li
# -----
# Copyright (c) 2023 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
import os
import cv2
from utils import *

if __name__ == '__main__':
    obj_dir = './datasets/'   
    # for dataset_kind in os.listdir(obj_dir):
    img_dir = obj_dir + 'testset' + '/image/'
    tar_dir = './datasets/8bit/'+'testset'+'/image/'
    maybe_make_dir(tar_dir)
    for img in os.listdir(img_dir):
        img_arr = cv2.imread(img_dir+img, -1)
        img_arr = (img_arr-img_arr.min())/(img_arr.max()-img_arr.min())
        img_arr *= 255
        img_arr = img_arr.astype(np.uint8)
        cv2.imwrite(tar_dir+img, img_arr)