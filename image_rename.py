# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/image_rename.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet
# Created Date: Wednesday, November 8th 2023, 11:48:11 am
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2023-11-08 23:14:01
# Modified By: Xinyu Li
# -----
# Copyright (c) 2023 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
import os

if __name__ == '__main__':
    obj_dir = './datasets/'
    for dataset_kind in os.listdir(obj_dir):
        img_dir = obj_dir + dataset_kind + '/image/'
        for img in os.listdir(img_dir):
            if len(img.split('_')[2]) !=2:
                patient_id = img[:-9]
                img_name = img[-9:].split('_')[0]+img[-9:].split('_')[1]
                new_name = patient_id + '_' + img_name
                os.rename(img_dir+img,img_dir+new_name)