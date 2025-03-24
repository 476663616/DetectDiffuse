# -*- coding:utf-8 -*-
###
# File: /home/xinyul/python_exercises/3D_diffusuionDet/test_cpu.py
# Project: /home/xinyul/python_exercises/3D_diffusuionDet
# Created Date: Saturday, April 27th 2024, 9:05:51 pm
# Author: Xinyu Li
# Email: 3120235098@bit.edu.cn
# -----
# Last Modified: 2024-04-27 21:16:20
# Modified By: Xinyu Li
# -----
# Copyright (c) 2024 Beijing Institude of Technology.
# ------------------------------------
# 请你获得幸福！！！
###
# import numpy as np 
# import threading # 占用内存 
# def allocate_memory(): 
#     while True: 
#         array = np.zeros((10000, 10000)) # 生成一个较大的零矩阵 
#         # 占用 CPU 
# def consume_cpu(): 
#     while True: 
#         result = 0 
#         for _ in range(1000000): 
#             result += np.random.random() # 执行大量随机数计算 
# # 主程序 
# if __name__ == "__main__": # 创建多个线程来同时执行占用内存和 CPU 的操作 
#     num_threads = 5 # 定义线程数量 
#     memory_threads = [threading.Thread(target=allocate_memory) for _ in range(num_threads)] 
#     cpu_threads = [threading.Thread(target=consume_cpu) for _ in range(num_threads)] # 启动线程 
#     for thread in memory_threads + cpu_threads: thread.start() 
#     # 等待线程结束 
#     for thread in memory_threads + cpu_threads: thread.join()

import sys  
  
def consume_memory():  
    # 获取系统总内存（以字节为单位）  
    list = []
    while True:
        list.append(' ' * 1000000000)
        print()

      
    # print(f"占用内存: {current_memory / (1024 ** 3)} GB")  
  
# 调用函数  
consume_memory()