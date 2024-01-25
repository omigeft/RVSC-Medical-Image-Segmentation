# from .unet_parts import *
#
#
# class RSU(nn.Module):
#     """Residual U-Block (RSU)"""
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(RSU, self).__init__()
#         self.height = 6
#
#         self.inc = DoubleConv(in_channels, out_channels)
#         self.down0 = Down(out_channels, mid_channels)
#         self.down_mid = Down(mid_channels, mid_channels)
#         self.up_mid = Up(mid_channels, mid_channels)
#         self.up0 = Up(mid_channels, out_channels)
#         self.outc = OutConv(out_channels, out_channels)
#
#     def forward(self, x):
#         y = []
#         y.append(self.inc(x))       # y[0]
#         y.append(self.down0(y[0]))  # y[1]
#         for i in range(1, self.height + 1): # (1, 7)
#             y.append(self.down_mid(y[i]))   # y[2] ~ y[7]
#         for i in range(self.height - 1, 0, -1): # (6, 0, -1)
#             y.append(self.up_mid(y[self.height * 2 - i], y[i])) # y[8] ~ y[13]
#         y.append(self.up0(y[-1], y[0]))
#         y.append(self.outc(y[-1]))
#         return y[-1]
#
# class RSU4F(nn.Module):
#     """Residual U-Block (RSU)"""
#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(RSU4F, self).__init__()
#         self.height = 4
#
#         self.inc = DoubleConv(in_channels, out_channels)
#         self.down0 = Down(out_channels, mid_channels)
#         self.down_mid = Down(mid_channels, mid_channels)
#         self.up_mid = Up(mid_channels, mid_channels)
#         self.up0 = Up(mid_channels, out_channels)
#         self.outc = OutConv(out_channels, out_channels)
#
#     def forward(self, x):
#         y = []
#         y.append(self.inc(x))
#         y.append(self.down0(y[0]))
#         for i in range(1, self.height + 1):
#             y.append(self.down_mid(y[i]))
#         for i in range(self.height - 1, 0, -1):
#             y.append(self.up_mid(y[i + 1], y[i]))
#         y.append(self.up0(y[-1], y[0]))
#         y.append(self.outc(y[-1]))
#         return y[-1]
#
#
# class U2Net(nn.Module):
#     def __init__(self, n_channels=1, n_classes=1, bilinear=False):
#         super(U2Net, self).__init__()
#         self.model_name = 'u2net'
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.stage1 = RSU(n_channels, 32, 64)
#         self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#         self.stage2 = RSU(64, 32, 128)
#         self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
#
#         self.out_conv = OutConv(64, n_classes)
#
#     def forward(self, x):
#         hx1 = self.stage1(x)
#         hx = self.pool1(hx1)
#
#         hx2 = self.stage2(hx)
#         hx = self.pool2(hx2)
#
#         hx = self.up2(hx2) + hx1
#         hx = self.up1(hx)
#
#         out = self.out_conv(hx)
#         return out


from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation  # 保持图像大小不变
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            # 因为后面有BN，bias不起作用
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class DownConvBNReLu(ConvBNReLU):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, flag=True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)

        self.down_flag = flag

    def forward(self, x):
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.conv(x)


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, flag=True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)

        self.up_flag = flag

    def forward(self, x1, x2):  # x1为下面传入的， x2为左边传入的
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class RSU(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch):
        super().__init__()
        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)  # 这个是不算在height上的

        encode_list = [DownConvBNReLu(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]

        for i in range(height - 2):  # 含有上下采样的模块
            encode_list.append(DownConvBNReLu(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))  # 这里最后的decode的输出是out_ch

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x):
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()  # 这是移除list最后的一个数据，并且将该数据赋值给x，这里的x是含有空洞卷积的输出
        for m in self.decode_modules:
            x2 = encode_outputs.pop()  # 这里是倒数第二深的输出，x表示下面的，x2表示左边的
            x = m(x, x2)  # 将下面的，和左边的一起传入到上卷积中
        return x + x_in  # 这里是最上面一层进行相加


class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x):
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class U2Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super().__init__()
        self.model_name = 'u2net'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        cfg = {
            # height, in_ch, mid_ch, out_ch, RSU4F, side
            "encode": [[7, n_channels, 32, 64, False, False],  # En1
                       [6, 64, 32, 128, False, False],  # En2
                       [5, 128, 64, 256, False, False],  # En3
                       [4, 256, 128, 512, False, False],  # En4
                       [4, 512, 256, 512, True, False],  # En5
                       [4, 512, 256, 512, True, True]],  # En6
            # height, in_ch, mid_ch, out_ch, RSU4F, side
            "decode": [[4, 1024, 256, 512, True, True],  # De5
                       [4, 1024, 128, 256, False, True],  # De4
                       [5, 512, 64, 128, False, True],  # De3
                       [6, 256, 32, 64, False, True],  # De2
                       [7, 128, 16, 64, False, True]]  # De1
        }

        assert "encode" in cfg
        assert "decode" in cfg

        self.encode_num = len(cfg["encode"])

        encode_list = []
        side_list = []

        for c in cfg["encode"]:
            # [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))  # 这里的*是将列表解开为单独的数值，这样才能传入到函数中

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], n_classes, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        for c in cfg["decode"]:
            assert len(c) == 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))

            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], n_classes, kernel_size=3, padding=1))
        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.out_conv = nn.Conv2d(self.encode_num * n_classes, n_classes, kernel_size=1)  # 这里是针对cat后的结果进行卷积，得到最后的out_ch=1

        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, h, w = x.shape

        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:  # 除了最后一个encode_block不用下采样，其余每一个block都需要下采样
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
            x = m(torch.cat([x, x2], dim=1))
            decode_outputs.insert(0, x)  # 这里是保证了从上到下的decode层的输出，在列表中的遍历是从0到5

        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode="bilinear", align_corners=False)
            side_outputs.insert(0, x)
        x = self.out_conv(torch.cat(side_outputs, dim=1))


        # if self.training:  # 在训练的时候，需要将6个输出都拿出来进行loss计算，
        #     return [x] + side_outputs
        # else:  # 非训练时，直接sigmoid后的数据
        #     return torch.sigmoid(x)
        return self.ReLU(x)
        # return torch.sigmoid(x)