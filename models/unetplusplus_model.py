from .unet_parts import *


class UNetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetPlusPlus, self).__init__()
        self.model_name = 'unet++'
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 初始化U-Net++所需的层
        self.inc = DoubleConv(n_channels, 64)
        self.down0 = Down(64, 128)
        self.down1 = Down(128, 256)
        self.down2 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down3 = Down(512, 1024 // factor)

        self.up0_0 = Up(128, 64, bilinear)
        self.up1_0 = Up(256, 128, bilinear)
        self.up2_0 = Up(512, 256, bilinear)
        self.up3_0 = Up(1024, 512 // factor, bilinear)

        self.up0_1 = Up(128, 64, 128 + 64, bilinear)
        self.up1_1 = Up(256, 128, 256 + 128, bilinear)
        self.up2_1 = Up(512, 256, 512 + 256, bilinear)

        self.up0_2 = Up(128, 64, 128 + 64 + 64, bilinear)
        self.up1_2 = Up(256, 128, 256 + 128 + 128, bilinear)

        self.up0_3 = Up(128, 64, 128 + 64 + 64 + 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x0_0 = self.inc(x)
        x1_0 = self.down0(x0_0)
        x2_0 = self.down1(x1_0)
        x3_0 = self.down2(x2_0)
        x4_0 = self.down3(x3_0)

        x0_1 = self.up0_0(x1_0, x0_0)
        x1_1 = self.up1_0(x2_0, x1_0)
        x2_1 = self.up2_0(x3_0, x2_0)
        x3_1 = self.up3_0(x4_0, x3_0)

        x0_2 = self.up0_1(x1_1, x0_0, x0_1)
        x1_2 = self.up1_1(x2_1, x1_0, x1_1)
        x2_2 = self.up2_1(x3_1, x2_0, x2_1)

        x0_3 = self.up0_2(x1_2, x0_0, x0_1, x0_2)
        x1_3 = self.up1_2(x2_2, x1_0, x1_1, x1_2)

        x0_4 = self.up0_3(x1_3, x0_0, x0_1, x0_2, x0_3)

        # TODO: deep supervision

        output = self.outc(x0_4)
        return output

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down0 = torch.utils.checkpoint(self.down0)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up0_0 = torch.utils.checkpoint(self.up0_0)
        self.up1_0 = torch.utils.checkpoint(self.up1_0)
        self.up2_0 = torch.utils.checkpoint(self.up2_0)
        self.up3_0 = torch.utils.checkpoint(self.up3_0)
        self.up0_1 = torch.utils.checkpoint(self.up0_1)
        self.up1_1 = torch.utils.checkpoint(self.up1_1)
        self.up2_1 = torch.utils.checkpoint(self.up2_1)
        self.up0_2 = torch.utils.checkpoint(self.up0_2)
        self.up1_2 = torch.utils.checkpoint(self.up1_2)
        self.up0_3 = torch.utils.checkpoint(self.up0_3)
        self.outc = torch.utils.checkpoint(self.outc)