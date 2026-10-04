import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, stride=1)
        self.down1 = ConvBNReLU(32, 64, stride=2)
        self.down2 = ConvBNReLU(64, 128, stride=2)
        self.down3 = ConvBNReLU(128, 256, stride=2)
        self.down4 = ConvBNReLU(256, 512, stride=2)

    def forward(self, x):
        c1 = self.stem(x)    # 224 x 400
        c2 = self.down1(c1)  # 112 x 200
        c3 = self.down2(c2)  # 56 x 100
        c4 = self.down3(c3)  # 28 x 50
        c5 = self.down4(c4)  # 14 x 25
        return c1, c2, c3, c4, c5


class DualCamBEVFPN(nn.Module):
    def __init__(self, num_classes=5, bev_h=240, bev_w=240):
        super().__init__()

        self.bev_h = bev_h
        self.bev_w = bev_w

        self.encoder = Encoder()

        self.lat5 = nn.Conv2d(1024, 256, 1)
        self.lat4 = nn.Conv2d(512, 256, 1)
        self.lat3 = nn.Conv2d(256, 256, 1)
        self.lat2 = nn.Conv2d(128, 256, 1)

        self.smooth4 = ConvBNReLU(256, 256)
        self.smooth3 = ConvBNReLU(256, 256)
        self.smooth2 = ConvBNReLU(256, 256)

        self.bev_decoder = nn.Sequential(
            ConvBNReLU(256, 256),
            nn.Upsample(size=(bev_h, bev_w), mode="bilinear", align_corners=False),
            ConvBNReLU(256, 128),
            ConvBNReLU(128, 64),
            nn.Conv2d(64, num_classes, 1)
        )

    def fuse_level(self, f_front, f_rear):
        f_rear = torch.flip(f_rear, dims=[2])
        return torch.cat([f_front, f_rear], dim=1)

    def forward(self, imgs):
        """
        imgs: [B, 2, 3, H, W]
        """
        front = imgs[:, 0]
        rear = imgs[:, 1]

        f1, f2, f3, f4, f5 = self.encoder(front)
        r1, r2, r3, r4, r5 = self.encoder(rear)

        p5 = self.lat5(self.fuse_level(f5, r5))
        p4 = self.lat4(self.fuse_level(f4, r4)) + F.interpolate(
            p5, size=f4.shape[-2:], mode="nearest"
        )
        p4 = self.smooth4(p4)

        p3 = self.lat3(self.fuse_level(f3, r3)) + F.interpolate(
            p4, size=f3.shape[-2:], mode="nearest"
        )
        p3 = self.smooth3(p3)

        p2 = self.lat2(self.fuse_level(f2, r2)) + F.interpolate(
            p3, size=f2.shape[-2:], mode="nearest"
        )
        p2 = self.smooth2(p2)

        out = self.bev_decoder(p2)
        return out
