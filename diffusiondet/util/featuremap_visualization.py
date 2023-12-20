'''
File: /home/xinyul/python_exercises/DiffusionDet/diffusiondet/util/featuremap_visualization.py
Project: /home/xinyul/python_exercises/DiffusionDet/diffusiondet/util
Created Date: Wednesday, July 12th 2023, 9:13:22 am
Author: pangpang li
-----
Last Modified: 2023-07-12 11:02:37
Modified By: pangpang li
-----
Copyright (c) 2023 Beijing Institude of Technology.
------------------------------------
Have a nice day!
'''
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
# from random import random
import random
import numpy as np

#定义函数，随机从0-end的一个序列中抽取size个不同的数
def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls
    


path = "test.jpg"
transformss = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#注意如果有中文路径需要先解码，最好不要用中文
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#转换维度
img = transformss(img).unsqueeze(0)

model = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
new_model = torchvision.models._utils.IntermediateLayerGetter(model, {'layer1': '1', 'layer2': '2',"layer3":"3"})
out = new_model(img)

tensor_ls=[(k,v) for  k,v in out.items()]

#这里选取layer2的输出画特征图
v=tensor_ls[1][1]

"""
如果要选layer3的输出特征图只需把第一个索引值改为2，即：
v=tensor_ls[2][1]
只需把第一个索引更换为需要输出的特征层对应的位置索引即可
"""
#取消Tensor的梯度并转成三维tensor，否则无法绘图
v=v.data.squeeze(0)

print(v.shape)  # torch.Size([512, 28, 28])


#随机选取25个通道的特征图
channel_num = random_num(25,v.shape[0])
plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index+1,)
    plt.imshow(v[channel, :, :])
plt.savefig("feature.jpg",dpi=300)