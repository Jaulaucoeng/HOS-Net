#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：RemoteHOS 
@File    ：train_hos_net.py
@Author  ：yauloucoeng
@Date    ：2024/4/28 21:36 
'''

import os

# step 1 train sle
cmd = "python train_sle.py --dataset sysu --gpu 0"
os.system(cmd)

# step 2 train sle+hsl
cmd = "python train_sle_hsl.py --dataset sysu --gpu 0"
os.system(cmd)

# step 3 train sle+hsl+cfl (final model, hos)
cmd = "python train_sle_hsl_cfl.py --dataset sysu --gpu 0"
os.system(cmd)

# step 4 test sle+hsl+cfl
cmd = "python test_sle_hsl_cfl.py --dataset sysu --gpu 0"
os.system(cmd)

