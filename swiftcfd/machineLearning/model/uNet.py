"""
Whole-field U-Net for the pressure warm start (Approach #3 / gap G3).

Unlike the MLP/RNN (which predict one cell from a local stencil and therefore
CANNOT represent the global/elliptic coupling of the pressure Poisson equation),
this model ingests the ENTIRE field and outputs the entire field.  The
down/up-sampling path is a learned multigrid: coarse levels carry global +
boundary information, fine levels carry local detail.  That is exactly what a
low-RESIDUAL pressure guess needs.

Tensor convention: NCHW, where H,W are the interior grid (nc_x, nc_y).
Input channels (set by the trainer), e.g. [u^n, v^n, p^n, rhs(div u*)].
Output: 1 channel = predicted pressure field (or the correction delta).

This file is a plain nn.Module (NOT a ModelBase subclass) because ModelBase /
the CSV DataManager assume per-cell stencils; the U-Net uses the separate
field-snapshot pipeline (`gen_field_data.py` + `train_unet.py`).
"""

import torch
from torch import nn


def _conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout), nn.GELU(),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base=32, levels=3):
        """
        base   : channel width at the finest level (doubles each level down)
        levels : number of down-sampling steps (3 is plenty for 64x64)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels

        # encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_channels
        ch = base
        enc_channels = []
        for _ in range(levels):
            self.downs.append(_conv_block(c, ch))
            self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(ch)
            c, ch = ch, ch * 2

        # bottleneck
        self.bottleneck = _conv_block(c, ch)

        # decoder (mirror)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for skip_ch in reversed(enc_channels):
            self.ups.append(nn.ConvTranspose2d(ch, skip_ch, 2, stride=2))
            self.up_convs.append(_conv_block(skip_ch * 2, skip_ch))
            ch = skip_ch

        self.head = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        # pad H,W up to a multiple of 2**levels so pooling/upsampling line up,
        # then crop back -- lets the net run on any grid size (e.g. 64 or 65).
        _, _, H, W = x.shape
        m = 2 ** self.levels
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")

        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, up_conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            x = up_conv(torch.cat([x, skip], dim=1))
        x = self.head(x)

        if ph or pw:
            x = x[:, :, :H, :W]
        return x

    def __str__(self):
        return "unet"
