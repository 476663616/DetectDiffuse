# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/draw_net.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet
# Created Date: Monday, March 4th 2024, 8:56:51 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-03-04 21:07:19
# Modified By: Xinyu Li
# -----
# Copyright (c) 2024 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
import graph_tool.all as gt
from graph_tool.draw import *
 
# 创建无向图对象
g = gt.Graph()
 
# 添加节点
v0 = g.add_vertex()
v1 = g.add_vertex()
v2 = g.add_vertex()
 
# 添加边
e0 = g.add_edge(v0, v1)
e1 = g.add_edge(v1, v2)
 
# 设置节点属性
g.vertex_properties["name"] = ["A", "B", "C"]
 
# 设置边属性
g.edge_properties["weight"] = [1]
 
# 绘制网络图
gt.graph_draw(g, vertex_text="name", output_size=(400, 400), output="knowledge_map.png")