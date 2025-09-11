import torch
import torch.nn as nn
import torch.nn.functional as F

"""
EventSegFast: Efficient human segmentation model for event-camera voxel inputs.
Design goals:
- Fast inference (edge friendly): depthwise separable convs, lightweight temporal fusion.
- Robust to still (static) humans: maintains a temporal feature memory with exponential decay so regions without recent events retain representation.
- Handles sparse & bursty event structure: adaptive normalization per voxel density & timestamp decay channel.
- Voxels: expected input shape [B, C, H, W] where C = time_bins * polarity (+ optional extra channels). For streaming sequences, call forward on each voxel slice; internal memory accumulates temporal context.

Key Components:
1. VoxelEmbedding: depthwise temporal mixing + pointwise projection.
2. TemporalAggregator: causal gated temporal conv + optional GRU (configurable) operating on a compact latent token (global + pooled quadrants) for long-range context.
3. SpatialEncoder: Hierarchical inverted residual (MobileNetV3-like) blocks.
4. CrossScaleFusion: Lightweight FPN for decoder.
5. TemporalMemory: EMA memory of multi-scale features; allows inference when subject stops generating events.
6. SegmentationHead: produces per-pixel probability map.

Usage:
model = EventSegFast(voxel_channels=10, height=260, width=346)
mask = model(voxel)  # voxel: [B, C, H, W]
For streaming: call model.step(voxel_t) per time slice if you pre-batch per-time voxels.
"""


def _depthwise_separable(in_ch, out_ch, k=3, s=1, act=True):
    padding = k // 2
    layers = [
        nn.Conv2d(in_ch, in_ch, k, s, padding, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_ch)
    ]
    if act:
        layers.append(nn.SiLU(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel=3, se=True):
        super().__init__()
        mid = in_ch * expansion
        self.use_res = in_ch == out_ch
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, kernel, padding=kernel//2, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True)
        )
        self.se = SqueezeExcite(mid) if se else nn.Identity()
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        out = self.pw1(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.pw2(out)
        if self.use_res:
            out = out + x
        return out


class SqueezeExcite(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)


class VoxelEmbedding(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.temporal_mix = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        # Density & decay normalization parameters
        self.gamma = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

    def forward(self, x, density=None):
        # x: [B, C, H, W], density: optional per-pixel event counts
        h = self.temporal_mix(x)
        if density is not None:
            d = torch.log1p(density).unsqueeze(1)  # [B,1,H,W]
            d = (d - d.mean(dim=[2,3], keepdim=True)) / (d.std(dim=[2,3], keepdim=True) + 1e-5)
            h = h * (1 + 0.1 * d)  # mild modulation
        return h * self.gamma + self.beta


class TemporalAggregator(nn.Module):
    """Maintains a compact latent token with causal updates for long-range context."""
    def __init__(self, feat_dim, hidden_dim=128, use_gru=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj_in = nn.Linear(feat_dim, hidden_dim)
        self.use_gru = use_gru
        if use_gru:
            self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Sigmoid()
        )
        self.proj_out = nn.Linear(hidden_dim, feat_dim)
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, feat):
        # feat: [B, C, H, W]
        token = self.pool(feat).flatten(1)  # [B, C]
        token = self.proj_in(token)
        if self.use_gru:
            if self.state is None:
                self.state = torch.zeros_like(token)
            self.state = self.rnn(token, self.state)
            core = self.state
        else:
            core = token if self.state is None else 0.9 * self.state + 0.1 * token
            self.state = core
        gate = self.gate(core).unsqueeze(-1).unsqueeze(-1)
        infused = feat + gate * self.proj_out(core).unsqueeze(-1).unsqueeze(-1)
        return infused


class TemporalMemory(nn.Module):
    def __init__(self, channels, decay=0.9):
        super().__init__()
        self.decay = decay
        self.register_buffer('mem', None, persistent=False)
        self.norm = nn.BatchNorm2d(channels)

    def reset(self):
        self.mem = None

    def forward(self, x):
        if self.mem is None:
            self.mem = x.detach()
        else:
            self.mem = self.decay * self.mem + (1 - self.decay) * x.detach()
        fused = 0.5 * x + 0.5 * self.mem
        return self.norm(fused)


class CrossScaleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, 96, 1) for c in channels])
        self.out_conv = nn.Sequential(
            nn.Conv2d(96 * len(channels), 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True)
        )
    def forward(self, feats):
        # feats list high->low resolution order expected
        up_feats = []
        target_size = feats[0].shape[-2:]
        for f, lat in zip(feats, self.lats):
            x = lat(f)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            up_feats.append(x)
        x = torch.cat(up_feats, dim=1)
        return self.out_conv(x)


class SegmentationHead(nn.Module):
    def __init__(self, in_ch, mid=96):
        super().__init__()
        self.block = nn.Sequential(
            _depthwise_separable(in_ch, mid),
            _depthwise_separable(mid, mid),
            nn.Conv2d(mid, 1, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.block(x))


class EventSegFast(nn.Module):
    def __init__(self, voxel_channels, height=260, width=346, base_ch=48, use_gru=True):
        super().__init__()
        self.height = height
        self.width = width

        self.embed = VoxelEmbedding(voxel_channels, base_ch)
        self.temporal_agg = TemporalAggregator(base_ch, hidden_dim=128, use_gru=use_gru)
        self.temporal_mem = TemporalMemory(base_ch)

        # Encoder hierarchy
        self.enc1 = InvertedResidual(base_ch, base_ch, expansion=4)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 3, 2, 1)
        self.enc2 = InvertedResidual(base_ch*2, base_ch*2, expansion=4)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 3, 2, 1)
        self.enc3 = InvertedResidual(base_ch*4, base_ch*4, expansion=4)

        self.fusion = CrossScaleFusion([base_ch, base_ch*2, base_ch*4])
        self.head = SegmentationHead(128)

    def reset_state(self):
        self.temporal_agg.reset()
        self.temporal_mem.reset()

    @torch.inference_mode()
    def step(self, voxel, density=None):
        """Streaming single-step inference.
        voxel: [B, C, H, W]
        density: optional [B, H, W]
        """
        return self.forward(voxel, density)

    def forward(self, voxel, density=None):
        # voxel: [B, C, H, W]
        x = self.embed(voxel, density)
        x = self.temporal_agg(x)
        x = self.temporal_mem(x)

        f1 = self.enc1(x)               # [B, C, H, W]
        f2 = self.enc2(self.down1(f1))  # [B, 2C, H/2, W/2]
        f3 = self.enc3(self.down2(f2))  # [B, 4C, H/4, W/4]

        fused = self.fusion([f1, f2, f3])
        out = self.head(fused)          # [B,1,H,W]
        return out


if __name__ == '__main__':
    B, C, H, W = 2, 10, 260, 346
    model = EventSegFast(C, H, W)
    x = torch.randn(B, C, H, W)
    y = model(x)
    print('Output shape:', y.shape)
