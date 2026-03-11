"""
UNETR++ with Voxel-Focused Attention (VFA-UNETR++)

Shaker et al., "UNETR++ with Voxel-Focused Attention: Efficient 3D Medical
Image Segmentation with Linear-Complexity Transformers."

Key ideas:
  - VFA: each voxel attends only to its local 3x3x3 neighborhood → O(27N)
  - EPA: paired VFA + channel attention with shared Q,K, separate V
  - 3-stage hierarchical encoder on 64^3 patches (patch_size=4)
  - CNN decoder with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_windows(x):
    """
    For each voxel, gather its 3x3x3 neighborhood.
    x: [B, C, D, H, W]
    Returns: [B, C, 27, D, H, W]
    """
    D, H, W = x.shape[2:]
    x_pad = F.pad(x, (1, 1, 1, 1, 1, 1))
    windows = [
        x_pad[:, :, dz:dz+D, dy:dy+H, dx:dx+W]
        for dz in range(3)
        for dy in range(3)
        for dx in range(3)
    ]
    return torch.stack(windows, dim=2)  # [B, C, 27, D, H, W]


class EPABlock(nn.Module):
    """
    Efficient Paired Attention block.

    Spatial branch: Voxel-Focused Attention (local 3x3x3 window, linear complexity).
    Channel branch: Global channel attention (C x C attention matrix).
    Shared Q, K between branches; separate V for each.
    Both branch outputs are summed, then projected.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Shared Q, K
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

        # Separate V per branch
        self.v_s = nn.Linear(dim, dim, bias=False)
        self.v_c = nn.Linear(dim, dim, bias=False)

        self.proj = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def _vfa(self, q, k, v, spatial_shape):
        """Local 3x3x3 spatial attention."""
        B, N, C = q.shape
        D, H, W = spatial_shape
        h, d = self.num_heads, self.head_dim

        # Reshape K, V to spatial and gather 3x3x3 windows
        k_sp = k.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)   # [B, C, D, H, W]
        v_sp = v.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # [B, C, 27, D, H, W] → [B, N, 27, h, d]
        k_win = _gather_windows(k_sp).permute(0, 3, 4, 5, 2, 1).reshape(B, N, 27, h, d)
        v_win = _gather_windows(v_sp).permute(0, 3, 4, 5, 2, 1).reshape(B, N, 27, h, d)

        # Multi-head attention: each query attends to 27 neighbors
        q_h = q.reshape(B, N, h, d)   # [B, N, h, d]
        # attn[b,n,hd,w] = sum_d q[b,n,hd,d] * k_win[b,n,w,hd,d]
        attn = torch.einsum('bnhd, bnwhd -> bnhw', q_h, k_win) * self.scale  # [B, N, h, 27]
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bnhw, bnwhd -> bnhd', attn, v_win)                # [B, N, h, d]
        return out.reshape(B, N, C)

    def _channel_attn(self, q, k, v):
        """Global channel attention — channels as tokens, spatial positions as features."""
        B, N, C = q.shape
        h, d = self.num_heads, self.head_dim

        # [B, N, h, d] → [B, h, d, N]  (each head_dim channel is a "token" with N features)
        q_c = q.reshape(B, N, h, d).permute(0, 2, 3, 1)   # [B, h, d, N]
        k_c = k.reshape(B, N, h, d).permute(0, 2, 3, 1)
        v_c = v.reshape(B, N, h, d).permute(0, 2, 3, 1)

        attn = (q_c @ k_c.transpose(-1, -2)) * self.scale   # [B, h, d, d]
        attn = F.softmax(attn, dim=-1)
        out = attn @ v_c                                      # [B, h, d, N]
        return out.permute(0, 3, 1, 2).reshape(B, N, C)

    def forward(self, x, spatial_shape):
        shortcut = x
        x = self.norm1(x)

        q, k = self.q(x), self.k(x)
        out = self._vfa(q, k, self.v_s(x), spatial_shape) + self._channel_attn(q, k, self.v_c(x))

        x = shortcut + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                            # [B, C, D, H, W]
        shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)            # [B, N, C]
        return self.norm(x), shape


class PatchMerging3D(nn.Module):
    """Strided conv downsampling: halves spatial dims, doubles channels."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, spatial_shape):
        B, N, C = x.shape
        D, H, W = spatial_shape
        x = x.transpose(1, 2).reshape(B, C, D, H, W)
        x = self.conv(x)
        new_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), new_shape


def _to_spatial(x, shape):
    """[B, N, C] → [B, C, D, H, W]"""
    B, N, C = x.shape
    D, H, W = shape
    return x.transpose(1, 2).reshape(B, C, D, H, W)


def _dec_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
    )


class DeepFLAIRUNETRPP(nn.Module):
    """
    UNETR++ with VFA for 3D medical image synthesis.

    3-stage encoder on 64^3 patches:
      patch_size=4 → 16^3 tokens (stage 1)
                   → 8^3  tokens (stage 2)
                   → 4^3  tokens (bottleneck)
    CNN decoder with skip connections, ReLU final activation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: tuple = (64, 64, 64),
        base_dim: int = 32,
        num_heads: int = 4,
        depths: tuple = (2, 2, 2),
        patch_size: int = 4,
    ):
        super().__init__()
        d1, d2, d3 = base_dim, base_dim * 2, base_dim * 4

        # Encoder
        self.patch_embed = PatchEmbed3D(in_channels, d1, patch_size)
        self.enc1 = nn.ModuleList([EPABlock(d1, num_heads) for _ in range(depths[0])])
        self.merge1 = PatchMerging3D(d1, d2)
        self.enc2 = nn.ModuleList([EPABlock(d2, num_heads) for _ in range(depths[1])])
        self.merge2 = PatchMerging3D(d2, d3)
        self.enc3 = nn.ModuleList([EPABlock(d3, num_heads) for _ in range(depths[2])])

        # Decoder
        self.up2 = nn.ConvTranspose3d(d3, d2, kernel_size=2, stride=2)
        self.dec2 = _dec_block(d2 * 2, d2)

        self.up1 = nn.ConvTranspose3d(d2, d1, kernel_size=2, stride=2)
        self.dec1 = _dec_block(d1 * 2, d1)

        # Upsample back to original resolution (patch_size factor)
        self.up0 = nn.ConvTranspose3d(d1, d1, kernel_size=patch_size, stride=patch_size)

        self.final_conv = nn.Sequential(
            nn.Conv3d(d1, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def _run_stage(self, x, shape, blocks):
        for blk in blocks:
            x = blk(x, shape)
        return x

    def forward(self, x):
        # Encoder
        x, s1 = self.patch_embed(x)           # [B, 16^3, d1], s1=(16,16,16)
        x = self._run_stage(x, s1, self.enc1)
        skip1 = _to_spatial(x, s1)            # [B, d1, 16, 16, 16]

        x, s2 = self.merge1(x, s1)            # [B, 8^3, d2], s2=(8,8,8)
        x = self._run_stage(x, s2, self.enc2)
        skip2 = _to_spatial(x, s2)            # [B, d2, 8, 8, 8]

        x, s3 = self.merge2(x, s2)            # [B, 4^3, d3], s3=(4,4,4)
        x = self._run_stage(x, s3, self.enc3)
        x = _to_spatial(x, s3)                # [B, d3, 4, 4, 4]

        # Decoder
        x = self.dec2(torch.cat([self.up2(x), skip2], dim=1))   # [B, d2, 8, 8, 8]
        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))   # [B, d1, 16, 16, 16]
        x = self.final_conv(self.up0(x))                         # [B, out, 64, 64, 64]
        return x
