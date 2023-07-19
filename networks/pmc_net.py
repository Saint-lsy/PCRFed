from networks.unet2d import ConvD, ConvU
import math
import numpy as np
from networks.layers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


class PMCNet(nn.Module):
    def __init__(self, c=3, n=16, norm='in', num_classes=2, embed_way=1):
        super(PMCNet, self).__init__()

        self.convd1 = ConvD(c, n, norm, first=True)
        self.convd2 = ConvD(n, 2 * n, norm)
        self.convd3 = ConvD(2 * n, 4 * n, norm)
        self.convd4 = ConvD(4 * n, 8 * n, norm)
        self.convd5 = ConvD(8 * n, 16 * n, norm)

        self.convu4 = ConvU(16 * n, norm, first=True)
        self.convu3 = ConvU(8 * n, norm)
        self.convu2 = ConvU(4 * n, norm)
        self.convu1 = ConvU(2 * n, norm)

        self.seg1 = nn.Conv2d(2 * n, num_classes, 1)

        self.embed_way = embed_way
        # self.bridge_layer = nn.Sequential(
        #     DWConv(16 * n, 1),
        #     nn.BatchNorm2d(1),
        #     nn.ReLU(inplace=True),
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        
        ##### method 1
        if self.embed_way == 1:
            embed = x5.view(x5.size(0), -1)
        elif self.embed_way == 2:
            embed = self.avgpool(x5).view(x5.size(0), -1)
        elif self.embed_way == 3:
            embed = self.maxpool(x5).view(x5.size(0), -1)
        elif self.embed_way == 4:
            # embed = self.bridge_layer(x5).view(x5.size(0), -1)
            embed = torch.cat((self.avgpool(x5).view(x5.size(0), -1), self.maxpool(x5).view(x5.size(0), -1)), 1)

        ##### method 2   emb2
        # embed = self.avgpool(x5).view(x5.size(0), -1)

        # ####or  emb3

        # embed = self.maxpool(x5).view(x5.size(0), -1)

        # ##### method 3  emb4
        # embed = torch.cat((self.avgpool(x5).view(x5.size(0), -1), self.maxpool(x5).view(x5.size(0), -1)), 1)


        y4 = self.convu4(x5, x4)
        y3 = self.convu3(y4, x3)
        y2 = self.convu2(y3, x2)
        y1 = self.convu1(y2, x1)

        y1_pred = conv2d(y1,
                         self.seg1.weight,
                         self.seg1.bias,
                         kernel_size=None,
                         stride=1,
                         padding=0)  #+ upsample(y2)

        predictions = torch.sigmoid(input=y1_pred)

        return embed, predictions




class DWConv(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=in_plane,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_plane)
        self.point_conv = nn.Conv2d(in_channels=in_plane,
                                    out_channels=out_plane,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x



'''
ASPP : Atrous Spatial Pyramid Pooling
'''
class ASPP(nn.Module):
    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(
                in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()