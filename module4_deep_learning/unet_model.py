"""
2D U-Net 网络定义 — 经典医学图像分割架构
轻量化设计：~1M 参数，CPU可训练
架构: 编码器-解码器 + 跳跃连接
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块: Conv → BN → ReLU → Conv → BN → ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """下采样块: MaxPool → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """上采样块: Upsample → Concat → DoubleConv"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸不匹配
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    经典 2D U-Net 轻量版
    输入: (B, 1, H, W) 灰度图
    输出: (B, num_classes, H, W) 逐像素类别概率

    结构:
    Encoder: C1(64) → C2(128) → C3(256) → C4(512)
    Bottleneck: C5(1024)
    Decoder: C4(512) → C3(256) → C2(128) → C1(64)
    Skip connections: C1↔C1, C2↔C2, C3↔C3, C4↔C4
    """

    def __init__(self, in_channels=1, num_classes=10, base_filters=64):
        """
        :param in_channels: 输入通道数 (灰度=1)
        :param num_classes: 输出类别数 (含背景)
        :param base_filters: 基础滤波器数 (轻量=32, 标准=64)
        """
        super().__init__()
        f = base_filters

        # Encoder
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)

        # Bottleneck
        self.bottleneck = Down(f * 8, f * 16)

        # Decoder — in_ch = skip_channels + upsampled_channels (after torch.cat)
        self.up1 = Up(f * 16 + f * 8, f * 8)   # x5(1024)+x4(512)=1536 → 512
        self.up2 = Up(f * 8 + f * 4, f * 4)    # 512+256=768 → 256
        self.up3 = Up(f * 4 + f * 2, f * 2)    # 256+128=384 → 128
        self.up4 = Up(f * 2 + f, f)            # 128+64=192 → 64

        # Output
        self.outc = nn.Conv2d(f, num_classes, 1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # (B, 64,  H,   W)
        x2 = self.down1(x1)    # (B, 128, H/2, W/2)
        x3 = self.down2(x2)    # (B, 256, H/4, W/4)
        x4 = self.down3(x3)    # (B, 512, H/8, W/8)

        # Bottleneck
        x5 = self.bottleneck(x4)  # (B, 1024, H/16, W/16)

        # Decoder with skip connections
        x = self.up1(x5, x4)   # (B, 512, H/8, W/8)
        x = self.up2(x, x3)    # (B, 256, H/4, W/4)
        x = self.up3(x, x2)    # (B, 128, H/2, W/2)
        x = self.up4(x, x1)    # (B, 64,  H,   W)

        # Output
        logits = self.outc(x)  # (B, num_classes, H, W)
        return logits

    def predict(self, x):
        """推理模式: 返回类别概率和标签"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1)
        return probs, labels
