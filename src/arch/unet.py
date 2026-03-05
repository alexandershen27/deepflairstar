import torch
import torch.nn as nn
from typing import Sequence, Optional, Union

class ResidualBlock(nn.Module):
    """
    Standard block B from paper: 2x (Conv3D, BatchNorm3D, LeakyReLU)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeepFLAIRNet(nn.Module):
    """
    3D U-Net as described in the DeepFLAIR* paper.
    Architecture:
    Input -> B(16) -> A -> B(32) -> A -> B(64) -> A -> B(128) -> A -> B(256) (Bottleneck)
    ...and back up using Skip Connections and ConvTranspose.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        
        # Encoder
        self.enc1 = ResidualBlock(in_channels, base_channels)
        self.pool1 = nn.AvgPool3d(kernel_size=2)
        
        self.enc2 = ResidualBlock(base_channels, base_channels * 2)
        self.pool2 = nn.AvgPool3d(kernel_size=2)
        
        self.enc3 = ResidualBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.AvgPool3d(kernel_size=2)
        
        self.enc4 = ResidualBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.AvgPool3d(kernel_size=2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(base_channels * 16, base_channels * 8) # concat(up4, enc4)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels)
        
        # Final projection: Additional block B + 1x1x1 Conv + ReLU
        self.final_conv = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.Conv3d(base_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        
        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.final_conv(d1)
