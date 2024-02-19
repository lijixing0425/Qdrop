# -*- coding: utf-8 -*-
# @Time    : 2023/3/6 \u4e0b\u534811:22
# @Author  : lijixing
# @FileName: shufflenetv2.py
# @Software: PyCharm
# @Github £º
import torch
import torch.nn as nn
from data.imagenet import build_imagenet_data
import numpy as np

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class shufflenetv2_block(nn.Module):
    def __init__(self, inp, oup, stride):
        super(shufflenetv2_block, self).__init__()
        self.stride = stride
        branch_features = oup // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=3, stride=self.stride, padding=1, bias=False, groups=inp),
                nn.BatchNorm2d(inp, eps=1e-05, momentum=0.1),
                nn.Conv2d(in_channels=inp, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features, eps=1e-05, momentum=0.1),
                nn.ReLU(inplace=True)
                )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=inp if (self.stride > 1) else branch_features,
                      out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=branch_features, out_channels=branch_features, kernel_size=3,
                      stride=self.stride, padding=1, groups=branch_features, bias=False, ),
            nn.BatchNorm2d(branch_features, eps=1e-05, momentum=0.1),
            nn.Conv2d(in_channels=branch_features, out_channels=branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):

        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class ShuffleNetV2(nn.Module):
    def __init__(self):
        super(ShuffleNetV2,self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(24, eps=1e-05, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )

        self.basic_block0 = shufflenetv2_block(24, 116, 2)
        self.basic_block1 = shufflenetv2_block(24, 116, 1)
        self.basic_block2 = shufflenetv2_block(24, 116, 1)
        self.basic_block3 = shufflenetv2_block(24, 116, 1)

        self.basic_block4 = shufflenetv2_block(116, 232, 2)
        self.basic_block5 = shufflenetv2_block(116, 232, 1)
        self.basic_block6 = shufflenetv2_block(116, 232, 1)
        self.basic_block7 = shufflenetv2_block(116, 232, 1)
        self.basic_block8 = shufflenetv2_block(116, 232, 1)
        self.basic_block9 = shufflenetv2_block(116, 232, 1)
        self.basic_block10 = shufflenetv2_block(116, 232, 1)
        self.basic_block11 = shufflenetv2_block(116, 232, 1)

        self.basic_block12 = shufflenetv2_block(232, 464, 2)
        self.basic_block13 = shufflenetv2_block(232, 464, 1)
        self.basic_block14 = shufflenetv2_block(232, 464, 1)
        self.basic_block15 = shufflenetv2_block(232, 464, 1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=464, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.drop_out = nn.Dropout(0.5, inplace=True)
        self.fc = nn.Linear(1024, 1000)


    def forward(self, x):

        x = self.conv0(x)
        x = self.basic_block0(x)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.basic_block3(x)
        x = self.basic_block4(x)
        x = self.basic_block5(x)
        x = self.basic_block6(x)
        x = self.basic_block7(x)
        x = self.basic_block8(x)
        x = self.basic_block9(x)
        x = self.basic_block10(x)
        x = self.basic_block11(x)
        x = self.basic_block12(x)
        x = self.basic_block13(x)
        x = self.basic_block14(x)
        x = self.basic_block15(x)

        f = self.conv1(x)
        x = f.mean([2, 3])
        x = self.drop_out(x)
        x = self.fc(x)

        return x