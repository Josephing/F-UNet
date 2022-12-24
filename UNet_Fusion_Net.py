import torch
import torch.nn as nn
from utils import *
from ASA_attention import ASAattention
from fengyutingmetrics import *
import torch.nn.functional as F


# Likeness UNet++ BLOCK1
class UNetBlock1(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels, middle_channels, 3)
        self.conv2 = Conv2d(middle_channels, out_channels, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


# Likeness UNet++ BLOCK2
class UNetBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2d(in_channels, out_channels, 3)

    def forward(self, x):
        out = self.conv(x)

        return out


# 第一步特征提取及融合
class Siamese_rgb(nn.Module):
    def __init__(self):
        super(Siamese_rgb, self).__init__()

        self.SL_1 = Conv2d(3, 32, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SL_2 = Conv2d(32, 64, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SL_3 = Conv2d(64, 128, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SL_4 = Conv2d(128, 256, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SL_5 = Conv2d(256, 512, 4, stride=2, bn=True, activation='relu', dropout=False)

        self.SR_1 = Deconv2d(3, 32, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SR_2 = Conv2d(32, 64, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SR_3 = Conv2d(64, 128, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SR_4 = Conv2d(128, 256, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.SR_5 = Conv2d(256, 512, 4, stride=2, bn=True, activation='relu', dropout=False)

        self.cat1 = Deconv2d(1024, 512, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.cat2 = Deconv2d(1024, 256, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.cat3 = Deconv2d(512, 128, 4, stride=2, bn=True, activation='relu', dropout=False)
        self.cat4 = Deconv2d(256, 64, 4, stride=2, bn=True, activation='relu', dropout=False)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_dim = Conv2d(3, 32, 1, stride=1, bn=True, activation='relu', dropout=False)

        self.ASA1 = ASAattention(channel=32)
        self.ASA2 = ASAattention(channel=128)
        self.ASA3 = ASAattention(channel=32)
        self.ASA4 = ASAattention(channel=128)

        self.ASA5 = ASAattention(channel=1024)
        self.ASA6 = ASAattention(channel=256)

    def forward(self, Pan_data, Ms_data):  # 输入[x, 3, 256, 256] [x, 3, 64, 64] x=4
        e1 = self.SL_1(Pan_data)
        e1 = self.ASA1(e1)
        e2 = self.SL_2(e1)
        e3 = self.SL_3(e2)
        e3 = self.ASA2(e3)
        e4 = self.SL_4(e3)
        e5 = self.SL_5(e4)

        e1_1 = self.SR_1(Ms_data)
        e1_1 = self.ASA3(e1_1)
        e2_1 = self.SR_2(e1_1)
        e3_1 = self.SR_3(e2_1)
        e3_1 = self.ASA4(e3_1)
        e4_1 = self.SR_4(e3_1)
        e5_1 = self.SR_5(e4_1)

        c1 = torch.cat((e1, e1_1), 1)
        c2 = torch.cat((e2, e2_1), 1)
        c3 = torch.cat((e3, e3_1), 1)
        c4 = torch.cat((e4, e4_1), 1)
        c5 = torch.cat((e5, e5_1), 1)

        d1 = self.cat1(c5)
        d1 = torch.cat((d1, c4), dim=1)
        d1 = self.ASA5(d1)

        d2 = self.cat2(d1)
        d2 = torch.cat((d2, c3), dim=1)

        d3 = self.cat3(d2)
        d3 = torch.cat((d3, c2), dim=1)
        d3 = self.ASA6(d3)

        d4 = self.cat4(d3)
        d4 = torch.cat((d4, c1), dim=1)

        up_Ms = self.up(e1_1)
        up_dim = self.up_dim(Pan_data)
        d5 = torch.cat((up_dim, up_Ms), dim=1)

        return d1, d2, d3, d4, d5  # # ([x, 1024, 16, 16]) ([x, 512, 32, 32]) ([x, 256, 64, 64]) ([x, 128, 128, 128]) ([x, 64, 256, 256])


# Likeness_UNet++
class NestedUNet_rgb(nn.Module):
    def __init__(self, num_classes=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [64, 128, 256, 512, 1024]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_0 = UNetBlock2(nb_filter[0], nb_filter[1])
        self.conv2_0 = UNetBlock2(nb_filter[1], nb_filter[2])
        self.conv3_0 = UNetBlock2(nb_filter[2], nb_filter[3])
        self.conv4_0 = UNetBlock2(nb_filter[3], nb_filter[4])

        self.conv0_1 = UNetBlock2(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = UNetBlock2(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = UNetBlock2(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = UNetBlock2(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = UNetBlock2(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = UNetBlock2(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = UNetBlock2(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = UNetBlock2(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = UNetBlock2(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = UNetBlock2(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        self.ASA1 = ASAattention(channel=64)
        self.ASA2 = ASAattention(channel=64)
        self.ASA3 = ASAattention(channel=64)
        self.ASA4 = ASAattention(channel=64)
        self.ASA5 = ASAattention(channel=64)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, d1, d2, d3, d4, d5):
        x0_0 = d5
        x0_0 = self.ASA1(x0_0)
        x1_0 = d4
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_1 = self.ASA2(x0_1)

        x2_0 = d3
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_2 = self.ASA3(x0_2)

        x3_0 = d2
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_3 = self.ASA4(x0_3)

        x4_0 = d1
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        x0_4 = self.ASA5(x0_4)

        output = self.final(x0_4)

        return output


class model_rgb(nn.Module):
    def __init__(self):
        super(model_rgb, self).__init__()
        self.sia = Siamese_rgb()
        self.UNet = NestedUNet_rgb()

    def forward(self, pan, ms):
        data_sia = self.sia(pan, ms)
        Unet = self.UNet(data_sia[0], data_sia[1], data_sia[2], data_sia[3], data_sia[4])

        return Unet

# class model_rgb(nn.Module):  # 获取参数
#     def __init__(self):
#         super(model_rgb, self).__init__()
#         # self.sia = Siamese_rgb()
#         self.UNet = NestedUNet_rgb()  # UNTE++对比试验
#         # self.UNet = self_UNet_rgb()  # 修改后的UNET对比试验
#
#     def forward(self, data_sia, data_sia1, data_sia2, data_sia3, data_sia4):
#         # data_sia = self.sia(pan, ms)
#         Unet = self.UNet(data_sia, data_sia1, data_sia2, data_sia3, data_sia4)
#
#         return Unet
