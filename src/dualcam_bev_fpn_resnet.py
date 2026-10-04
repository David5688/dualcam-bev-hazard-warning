import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResNetEncoder(nn.Module):
    """Shared ImageNet-pretrained ResNet-18 feature extractor."""
    def __init__(self, pretrained=True):
        super().__init__()
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=w)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # 64,  stride /4
        self.layer2 = net.layer2   # 128, stride /8
        self.layer3 = net.layer3   # 256, stride /16
        self.layer4 = net.layer4   # 512, stride /32

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class DualCamBEVFPNResNet(nn.Module):
    """DualCam FPN with a pretrained ResNet-18 backbone. Self-normalizes [0,1] RGB."""
    def __init__(self, num_classes=6, bev_h=240, bev_w=240, pretrained=True):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.encoder = ResNetEncoder(pretrained)

        # ImageNet normalization (input is [0,1] RGB from the dataset)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # fused channels = 2x per level (front + rear concat)
        self.lat4 = nn.Conv2d(1024, 256, 1)
        self.lat3 = nn.Conv2d(512, 256, 1)
        self.lat2 = nn.Conv2d(256, 256, 1)
        self.lat1 = nn.Conv2d(128, 256, 1)

        self.smooth3 = ConvBNReLU(256, 256)
        self.smooth2 = ConvBNReLU(256, 256)
        self.smooth1 = ConvBNReLU(256, 256)

        self.bev_decoder = nn.Sequential(
            ConvBNReLU(256, 256),
            nn.Upsample(size=(bev_h, bev_w), mode="bilinear", align_corners=False),
            ConvBNReLU(256, 128),
            ConvBNReLU(128, 64),
            nn.Conv2d(64, num_classes, 1),
        )

    def fuse(self, f_front, f_rear):
        f_rear = torch.flip(f_rear, dims=[2])
        return torch.cat([f_front, f_rear], dim=1)

    def _enc(self, x):
        x = (x - self.mean) / self.std
        return self.encoder(x)

    def forward(self, imgs):
        """imgs: [B, 2, 3, H, W] in [0,1] RGB."""
        f1, f2, f3, f4 = self._enc(imgs[:, 0])
        r1, r2, r3, r4 = self._enc(imgs[:, 1])

        p4 = self.lat4(self.fuse(f4, r4))
        p3 = self.smooth3(self.lat3(self.fuse(f3, r3)) +
                          F.interpolate(p4, size=f3.shape[-2:], mode="nearest"))
        p2 = self.smooth2(self.lat2(self.fuse(f2, r2)) +
                          F.interpolate(p3, size=f2.shape[-2:], mode="nearest"))
        p1 = self.smooth1(self.lat1(self.fuse(f1, r1)) +
                          F.interpolate(p2, size=f1.shape[-2:], mode="nearest"))

        return self.bev_decoder(p1)
