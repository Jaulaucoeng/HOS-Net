#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# References :https://github.com/GouravWadhwa/Hypergraphs-Image-Inpainting
import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    def __init__(
            self,
            in_features=1024,
            out_features=1024,
            features_height=18,
            features_width=9,
            edges=256,
            filters=128,
            apply_bias=True,
            theta1 = 0.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.features_height = features_height
        self.features_width = features_width
        self.vertices = self.features_height * self.features_width
        self.edges = edges
        self.apply_bias = apply_bias
        self.filters = filters
        self.theta1 = theta1

        self.phi_conv = nn.Conv2d(self.in_features, self.filters, kernel_size=1, stride=1, padding=0)
        self.A_conv = nn.Conv2d(self.in_features, self.filters, kernel_size=1, stride=1, padding=0)
        self.M_conv = nn.Conv2d(self.in_features, self.edges, kernel_size=7, stride=1, padding=3)

        self.weight_2 = nn.Parameter(torch.empty(self.in_features, self.out_features))
        nn.init.xavier_normal_(self.weight_2)

        if apply_bias:
            self.bias_2 = nn.Parameter(torch.empty(1, self.out_features))
            nn.init.xavier_normal_(self.bias_2)

    def forward(self, x):
        phi = self.phi_conv(x)
        phi = torch.permute(phi, (0, 2, 3, 1)).contiguous()
        phi = phi.view(-1, self.vertices, self.filters)

        A = F.avg_pool2d(x, kernel_size=(self.features_height, self.features_width))
        A = self.A_conv(A)
        A = torch.permute(A, (0, 2, 3, 1)).contiguous()

        A = torch.diag_embed(A.squeeze())  # checked

        M = self.M_conv(x)
        M = torch.permute(M, (0, 2, 3, 1)).contiguous()
        M = M.view(-1, self.vertices, self.edges)


        H = torch.matmul(phi, torch.matmul(A, torch.matmul(phi.transpose(1, 2), M)))
        H = torch.abs(H)

        if self.theta1 != 0.0:
            mean_H = self.theta1*torch.mean(H,dim=[1,2],keepdim=True)
            H = torch.where(H < mean_H, 0.0, H)
        D = H.sum(dim=2)
        D_H = torch.mul(torch.unsqueeze(torch.pow(D + 1e-10, -0.5), dim=-1), H)
        B = H.sum(dim=1)
        B = torch.diag_embed(torch.pow(B + 1e-10, -1))
        x_ = torch.permute(x, (0, 2, 3, 1)).contiguous()
        features = x_.view(-1, self.vertices, self.in_features)

        out = features - torch.matmul(D_H, torch.matmul(B, torch.matmul(D_H.transpose(1, 2), features)))
        out = torch.matmul(out, self.weight_2)

        if self.apply_bias:
            out = out + self.bias_2
        out = torch.permute(out, (0, 2, 1)).contiguous()
        out = out.view(-1, self.out_features, self.features_height, self.features_width)

        return out