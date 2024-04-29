#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50
from Transformer import ViT
from HyperGraphs import HypergraphConv
import whitening

class whitening_scale_shift(nn.Module):
    def __init__(self, planes, group_size, affine=True):
        super(whitening_scale_shift, self).__init__()
        self.planes = planes
        self.group_size = group_size
        self.affine = affine

        self.wh = whitening.WTransform2d(self.planes,
                                         self.group_size)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.planes, 1, 1))
            self.beta = nn.Parameter(torch.zeros(self.planes, 1, 1))


    def forward(self, x):
        return self.wh(x) * self.gamma + self.beta + x

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Bottleneck12(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck12, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50', dataset="sysu", plearn=0, stage=23, depth=-1, head=-1, graphw=-1, theta1=0.0):
        super(embed_net, self).__init__()
        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

        self.num_stripes = 6
        local_conv_out_channels = 256

        self.local_conv_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
            conv.apply(weights_init_kaiming)
            self.local_conv_list.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)
            ))

        self.fc_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            fc = nn.Linear(local_conv_out_channels, class_num)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_list.append(fc)
        if plearn == 1: # better or worse
            self.p = nn.Parameter(torch.ones(1) * 3.0)
            if dataset == 'sysu':
                self.p1 = nn.Parameter(torch.ones(1) * 3.0)
            else:
                self.p1 = nn.Parameter(torch.ones(1) * 10.0)
        elif plearn == 0:
            self.p = 3.0
            if dataset == 'sysu':
                self.p1 = 3.0
            else:
                self.p1 = 10.0
        self.stage = stage
        self.depth = depth
        self.head = head
        self.graphw = graphw
        self.theta1 = theta1
        self.cnn23 = Bottleneck12(inplanes=1024, planes=1024)
        if self.stage == 23:
            self.vit = ViT(img_size=18 * 9, embed_dim=1024, depth=self.depth, num_heads=self.head)
            self.hypergraph = HypergraphConv(theta1=self.theta1)

        self.whiten_o = whitening_scale_shift(1024, 1)


    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            out_23=x
            x = self.cnn23(x)
            if self.stage == 23 and self.training:
                out_23_shape = out_23.shape[0]//3
                temp = torch.cat((out_23[0:out_23_shape],out_23[2*out_23_shape:3*out_23_shape]), dim=0)
                x_vit = self.vit(temp)

                x = torch.cat((x, x_vit), dim=0)

            x = x + self.graphw * self.hypergraph(self.whiten_o(x))

            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            pass

        feat = x
        assert feat.size(2) % self.num_stripes == 0
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            # shape [N, C, 1, 1]

            # average pool
            # local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
            if self.gm_pool == 'on':
                # gm pool
                local_feat = feat[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                b, c, h, w = local_feat.shape
                local_feat = local_feat.view(b, c, -1)
                local_feat = (torch.mean(local_feat ** self.p1, dim=-1) + 1e-12) ** (1 / self.p1)
            else:
                # average pool
                # local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                                          (stripe_h, feat.size(-1)))

            # shape [N, c, 1, 1]
            local_feat = self.local_conv_list[i](local_feat.view(feat.size(0), feat.size(1), 1, 1))

            # shape [N, c]
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)

            if hasattr(self, 'fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        feat_all = [lf for lf in local_feat_list]
        feat_all = torch.cat(feat_all, dim=1)

        ## golable
        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            x_pool = (torch.mean(x ** self.p, dim=-1) + 1e-12) ** (1 / self.p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat = self.bottleneck(x_pool)
        if self.training:
            return x_pool, self.classifier(feat), local_feat_list, logits_list, feat_all
        else:
            x_pool_1 = torch.cat((x_pool, feat_all), dim=1)
            feat_1 = torch.cat((feat, feat_all), dim=1)
            return self.l2norm(x_pool_1), self.l2norm(feat_1)
