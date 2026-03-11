import torch
import torch.nn as nn
from src.arch.unet import DoubleConvBlock


class AttentionGate(nn.Module):
    """Standard soft attention gate (Oktay et al. 2018)."""
    def __init__(self, x_channels: int, g_channels: int, inter_channels: int):
        super().__init__()
        self.W_x = nn.Conv3d(x_channels, inter_channels, kernel_size=1)
        self.W_g = nn.Conv3d(g_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, x, g):
        # x: skip connection, g: gating signal (same spatial size after upsample)
        att = self.relu(self.W_x(x) + self.W_g(g))
        att = self.psi(att)
        return x * att


class DeepFLAIRAttentionNet(nn.Module):
    """Attention U-Net: same backbone as DeepFLAIRNet but with attention gates
    on all skip connections instead of plain concatenation."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        b = base_channels

        # Encoder (identical to DeepFLAIRNet)
        self.enc1 = DoubleConvBlock(in_channels, b)
        self.pool1 = nn.AvgPool3d(kernel_size=2)

        self.enc2 = DoubleConvBlock(b, b * 2)
        self.pool2 = nn.AvgPool3d(kernel_size=2)

        self.enc3 = DoubleConvBlock(b * 2, b * 4)
        self.pool3 = nn.AvgPool3d(kernel_size=2)

        self.enc4 = DoubleConvBlock(b * 4, b * 8)
        self.pool4 = nn.AvgPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConvBlock(b * 8, b * 16)

        # Decoder upsamples
        self.up4 = nn.ConvTranspose3d(b * 16, b * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(b * 8,  b * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(b * 4,  b * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(b * 2,  b,     kernel_size=2, stride=2)

        # Attention gates
        self.att4 = AttentionGate(x_channels=b * 8, g_channels=b * 8, inter_channels=b * 4)
        self.att3 = AttentionGate(x_channels=b * 4, g_channels=b * 4, inter_channels=b * 2)
        self.att2 = AttentionGate(x_channels=b * 2, g_channels=b * 2, inter_channels=b)
        self.att1 = AttentionGate(x_channels=b,     g_channels=b,     inter_channels=b // 2)

        # Decoder blocks (same input width as plain UNet — attention doesn't change channel count)
        self.dec4 = DoubleConvBlock(b * 16, b * 8)
        self.dec3 = DoubleConvBlock(b * 8,  b * 4)
        self.dec2 = DoubleConvBlock(b * 4,  b * 2)
        self.dec1 = DoubleConvBlock(b * 2,  b)

        self.final_conv = nn.Sequential(
            nn.Conv3d(b, out_channels, kernel_size=1),
            nn.ReLU(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x);  p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)
        e3 = self.enc3(p2); p3 = self.pool3(e3)
        e4 = self.enc4(p3); p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, self.att4(e4, u4)], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, self.att3(e3, u3)], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, self.att2(e2, u2)], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, self.att1(e1, u1)], dim=1))

        return self.final_conv(d1)
