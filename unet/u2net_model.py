from .unet_parts import *


class RSU(nn.Module):
    """Residual U-Block (RSU)"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super(RSU, self).__init__()
        self.down_conv = nn.Sequential(
            DoubleConv(in_ch, mid_ch),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        self.middle_conv = DoubleConv(mid_ch, mid_ch)
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(mid_ch, mid_ch, kernel_size=2, stride=2),
            DoubleConv(mid_ch, out_ch)
        )

    def forward(self, x):
        x = self.down_conv(x)
        x = self.middle_conv(x)
        x = self.up_conv(x)
        return x


class U2Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(U2Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.stage1 = RSU(n_channels, 32, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU(64, 32, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # ... 可以添加更多的 stage 和 pool ...

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)
        # ... 可以添加更多的 forward steps ...

        hx = self.up2(hx2) + hx1
        hx = self.up1(hx)

        out = self.out_conv(hx)
        return out

# 示例使用
# model = U2Net(in_ch=1, out_ch=2)
