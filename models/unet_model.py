""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.ReLU(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.ReLU(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel, c_attention=True, s_attention=True):
        super(CBAM, self).__init__()
        self.c_attention = c_attention
        self.s_attention = s_attention
        if self.c_attention:
            self.channel_attention = ChannelAttention(channel)
        if self.s_attention:
            self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = x
        if self.c_attention:
            out = self.channel_attention(out) * out
        if self.s_attention:
            out = self.spatial_attention(out) * out
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, c_attention=False, s_attention=False):
        super(UNet, self).__init__()
        if c_attention:
            if s_attention:
                self.model_name = 'unet_cs'
            else:
                self.model_name = 'unet_c'
        elif s_attention:
            self.model_name = 'unet_s'
        else:
            self.model_name = 'unet'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.attention = c_attention or s_attention

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        if self.attention:
            self.cbam1 = CBAM(64, c_attention, s_attention)
            self.cbam2 = CBAM(128, c_attention, s_attention)
            self.cbam3 = CBAM(256, c_attention, s_attention)
            self.cbam4 = CBAM(512, c_attention, s_attention)

    def forward(self, x):
        x1 = self.inc(x)
        if self.attention:
            x1 = self.cbam1(x1) + x1

        x2 = self.down1(x1)
        if self.attention:
            x2 = self.cbam2(x2) + x2

        x3 = self.down2(x2)
        if self.attention:
            x3 = self.cbam3(x3) + x3

        x4 = self.down3(x3)
        if self.attention:
            x4 = self.cbam4(x4) + x4

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)