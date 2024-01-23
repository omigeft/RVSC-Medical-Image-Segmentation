import torch
import torch.nn as nn

from .unet_parts import DoubleConv, Down, OutConv


class Up(nn.Module):
    """Upscaling then double conv with skip connections from all previous layers"""

    def __init__(self, in_channels, out_channels, skip_channels, bilinear=True):
        super().__init__()
        self.skip_channels = skip_channels
        self.total_in_channels = in_channels + sum(skip_channels)

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(self.total_in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(self.total_in_channels, out_channels)

    def forward(self, x, skip_connections):
        x = self.up(x)
        x = torch.cat([x] + skip_connections, dim=1)
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Contracting Path (Encoder)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Expansive Path (Decoder)
        self.up1 = Up(1024, 512 // factor, [512], bilinear)
        self.up2 = Up(512, 256 // factor, [512, 256], bilinear)
        self.up3 = Up(256, 128 // factor, [512, 256, 128], bilinear)
        self.up4 = Up(128, 64, [512, 256, 128, 64], bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, [x4])
        x = self.up2(x, [x4, x3])
        x = self.up3(x, [x4, x3, x2])
        x = self.up4(x, [x4, x3, x2, x1])

        logits = self.outc(x)
        return logits

# 示例使用
# model = UNetPlusPlus(n_channels=1, n_classes=2, bilinear=True)
