# scripts/unet_nearest.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class UpNearest(nn.Module):
    """Nearest upsample + 1x1 conv (chan reduce) + concat + double conv"""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.reduce = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.reduce(x)
        # パディングで空間サイズを合わせる（偶奇の差を吸収）
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x = F.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, 1, bias=True)
    def forward(self, x): return self.conv(x)

class UNetNearest(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base=64):
        super().__init__()
        self.inc   = DoubleConv(n_channels, base)
        self.down1 = Down(base,   base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*16)
        self.up1   = UpNearest(in_ch=base*16, skip_ch=base*8,  out_ch=base*8)
        self.up2   = UpNearest(in_ch=base*8,  skip_ch=base*4,  out_ch=base*4)
        self.up3   = UpNearest(in_ch=base*4,  skip_ch=base*2,  out_ch=base*2)
        self.up4   = UpNearest(in_ch=base*2,  skip_ch=base,    out_ch=base)
        self.outc  = OutConv(base, n_classes)
    def forward(self, x):
        x1 = self.inc(x);  x2 = self.down1(x1); x3 = self.down2(x2)
        x4 = self.down3(x3); x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3)
        x  = self.up3(x, x2);  x = self.up4(x, x1)
        return self.outc(x)  # ロジット

