#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# References : https://github.com/ohhhyeahhh/SiamGAT
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch

class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel, meanw):
        super(Graph_Attention_Union, self).__init__()
        self.meanw = meanw

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )


        self.init_weights()

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, zf, xf):
        # linear transformation

        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        # xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)
        if self.meanw != 0.0:
            mean_ = torch.mean(similar, dim=[2], keepdim=True)

            similar = torch.where(similar > self.meanw*mean_, similar, 0)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])


        return embedding






