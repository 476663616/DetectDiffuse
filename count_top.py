# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/count_top.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet
# Created Date: Tuesday, March 26th 2024, 12:06:18 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-03-26 12:08:47
# Modified By: Xinyu Li
# -----
# Copyright (c) 2024 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###

import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
model = torch.load('output/diffusionDet_8bit_ela/model_0203999.pth')['model']

dummy_input = torch.randn(1, 3, 512, 512)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))