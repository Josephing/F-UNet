import torch
from torch import nn
from collections import OrderedDict
from torch.nn.parameter import Parameter


class ASAattention(nn.Module):

    def __init__(self, channel=0, channel_out=0, hw=0, kernels=[1, 3, 5, 7]):
        super(ASAattention, self).__init__()
        self.hw = hw

        # Adaptive
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel_out, kernel_size=k, padding=k // 2)),  # H_out=H_in
                    ('bn', nn.BatchNorm2d(channel_out)),
                    ('relu', nn.ReLU())
                ]))
            )

        # SA
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel_out // 2, 1, 1))  # (1, 4, 1, 1)
        self.cbias = Parameter(torch.ones(1, channel_out // 2, 1, 1))  # (1, 4, 1, 1)
        self.sweight = Parameter(torch.zeros(1, channel_out // 2, 1, 1))
        self.sbias = Parameter(torch.ones(1, channel_out // 2, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel_out // 2, channel_out // 2)

        # utils
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        fuss = []
        # split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        # SA
        for x in feats:
            b, c, h, w = x.shape

            x = x.reshape(b, -1, h, w)
            assert (x.shape[1] > 1) and (x.shape[1] % 2 == 0), "分割后的通道维度必须大于1且可以被2整除"
            x_0, x_1 = x.chunk(2, dim=1)  # 按照维度进行张量分割

            # channel attention
            xn = self.avg_pool(x_0)  # (64, 4, 1, 1)
            xn = self.cweight * xn + self.cbias
            xn = x_0 * self.sigmoid(xn)

            # spatial attention
            xs = self.gn(x_1)
            xs = self.sweight * xs + self.sbias
            xs = x_1 * self.sigmoid(xs)

            # concatenate along channel axis
            out = torch.cat([xn, xs], dim=1)
            out = out.reshape(b, -1, h, w)

            out = self.channel_shuffle(out, 2)
            fuss.append(out)

        fus = torch.stack(fuss, 0)  # k,bs,channel,h,w
        out = fus.sum(0)

        if self.hw == 0:
            return out

        if out.shape[-1] > self.hw:
            assert (out.shape[-1] is not self.hw) and (out.shape[-1] % self.hw == 0) and (self.hw > 0), "新的长宽尺度必须被原来的整除且大于0"
            m = out.shape[-1] // self.hw
            out = nn.MaxPool2d(m, m)(out)
        elif out.shape[-1] < self.hw:
            assert (out.shape[-1] is not self.hw) and (self.hw % out.shape[-1] == 0) and (self.hw > 0), "新的长宽尺度必须被原来的整除且大于0"
            m = self.hw // out.shape[-1]
            out = nn.Upsample(scale_factor=m, mode='bilinear', align_corners=True)(out)

        return out


# if __name__ == '__main__':
    # Input = torch.randn(1, 4, 256, 256)
    # asa = ASAattention(channel=4, channel_out=2, hw=256)
    # # summary(asa.cuda(), input_size=(64, 32, 32))
    # output = asa(Input)
    # print(output.shape)
