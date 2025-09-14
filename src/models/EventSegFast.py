import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import eventstovoxel
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
        # Aggressive downsampling right at the start - 4x smaller (2x + 2x)
        self.downsample_conv = nn.Sequential(
            # First 2x downsample
            nn.Conv2d(in_ch, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            # Second 2x downsample  
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True)
        )
        # Density & decay normalization parameters
        self.gamma = nn.Parameter(torch.ones(1, embed_dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

    def forward(self, x, density=None):
        # x: [B, C, H, W] -> [B, embed_dim, H/4, W/4] 
        h = self.downsample_conv(x)
        if density is not None:
            # Downsample density to match
            d = F.avg_pool2d(density.unsqueeze(1), 4, 4)  # [B,1,H/4,W/4]
            d = torch.log1p(d)
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
    def detach(self):
        if self.state is not None:
            self.state = self.state.detach()
    def forward(self, feat):
        # feat: [B, C, H, W]
        token = self.pool(feat).flatten(1)  # [B, C]
        token = self.proj_in(token)
        if self.use_gru:
            if self.state is None:
                self.state = torch.zeros_like(token)
            # Detach state to prevent gradient accumulation
            self.state = self.rnn(token, self.state)
            core = self.state
        else:
            if self.state is None:
                core = token
            else:
                core = 0.9 * self.state + 0.1 * token
            self.state = core
        gate = self.gate(core).unsqueeze(-1).unsqueeze(-1)
        infused = feat + gate * self.proj_out(core).unsqueeze(-1).unsqueeze(-1)
        return infused


class TemporalMemory(nn.Module):
    def __init__(self, channels, hidden_dim=16):
        super().__init__()
        self.register_buffer('mem', None, persistent=False)
        self.norm = nn.BatchNorm2d(channels)
        self.gate_net = nn.Sequential(
            # Input: current features + memory + difference
            nn.Conv2d(channels * 2, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid()
        )
    def reset(self):
        self.mem = None
        
    def detach(self):
        if self.mem is not None:
            self.mem = self.mem.detach()
            
    def forward(self, x):
        if self.mem is None:
            self.mem = x.clone()  # Clone to avoid sharing storage
        else:
            # Detach to prevent gradient accumulation across time steps
            gate_input = torch.cat([x, self.mem], dim=1)  # [B, 2*C, H, W]
            update_rate = self.gate_net(gate_input)  # [B, 1, H, W]
            self.mem = update_rate * self.mem + (1 - update_rate) * x
        return self.norm(self.mem)


class CrossScaleFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.lats = nn.ModuleList([nn.Conv2d(c, 48, 1) for c in channels])  # Reduced from 96 to 48
        self.out_conv = nn.Sequential(
            nn.Conv2d(48 * len(channels), 64, 1, bias=False),  # Reduced from 96*3=288->128 to 48*3=144->64
            nn.BatchNorm2d(64),
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
    def __init__(self, in_ch, mid=96, target_height=260, target_width=346):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width
        
        # Process at low resolution first
        self.block = nn.Sequential(
            _depthwise_separable(in_ch, mid),
            _depthwise_separable(mid, mid),
            nn.Conv2d(mid, 32, 1)  # Reduce channels before upsampling
        )
        
        # Learnable 4x upsampling with transposed convolutions
        self.upsample = nn.Sequential(
            # First 2x upsample
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            # Second 2x upsample  
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            # Final classification
            nn.Conv2d(8, 1, 1)
        )
        
    def forward(self, x):
        x = self.block(x)  # Process at low resolution
        ## interpolate to target 1/4 size if needed
        x = self.upsample(x)  # Learnable 4x upsampling
        x = F.interpolate(x, size=(self.target_height, self.target_width), mode='bilinear', align_corners=False)

        return torch.sigmoid(x)


class EventSegFast(nn.Module):
    def __init__(self, model_type, voxel_channels, height=260, width=346, base_ch=24, use_gru=True):  # Reduced from 48 to 24
        super().__init__()
        self.height = height
        self.width = width
        self.model_type = model_type
        self.embed = VoxelEmbedding(voxel_channels, base_ch)

        self.temporal_agg = TemporalAggregator(base_ch, hidden_dim=64, use_gru=use_gru)  # Reduced from 128 to 64
        self.temporal_mem = TemporalMemory(base_ch)

        # Encoder hierarchy
        self.enc1 = InvertedResidual(base_ch, base_ch, expansion=4)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 3, 2, 1)
        self.enc2 = InvertedResidual(base_ch*2, base_ch*2, expansion=4)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 3, 2, 1)
        self.enc3 = InvertedResidual(base_ch*4, base_ch*4, expansion=4)

        self.fusion = CrossScaleFusion([base_ch, base_ch*2, base_ch*4])
        self.head = SegmentationHead(64, target_height=height, target_width=width)  # Pass target size

    def reset_states(self):
        self.temporal_agg.reset()
        self.temporal_mem.reset()
    def detach_states(self):
        self.temporal_agg.detach()
        self.temporal_mem.detach()
    @torch.inference_mode()
    def step(self, event_sequence, density=None):
        """Streaming single-step inference.
        voxel: [B, C, H, W]
        density: optional [B, H, W]
        """
        return self.forward(event_sequence, training=False, hotpixel=False, density=density)

    def forward(self, event_sequence, training=False, hotpixel=False, density=None):
        # Memory-efficient approach: don't store all outputs, process one by one
        
        # Training mode: store outputs for backprop
        outputs = []
        seq_events = []
       
        for i, events in enumerate(event_sequence):
            with torch.no_grad():
                min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
                max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
                denom = (max_t - min_t)
                denom[denom < 1e-8] = 1.0
                events = events.clone()
                events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                events[:,:, 1] = events[:, :, 1].clamp(0, self.width-1)
                events[:,:, 2] = events[:, :, 2].clamp(0, self.height-1)

                hist_events = eventstovoxel(events, self.height, self.width, training=training, hotpixel=hotpixel).float()
                seq_events.append(hist_events.detach())
            with torch.set_grad_enabled(training):
                out = self._forward_single(hist_events, density)
                outputs.append(out)
            
                
        outputs = torch.stack(outputs, dim=1).squeeze(2)  # [B, T, H, W]
        return outputs, None, seq_events

    

    def _forward_single(self, hist_events, density=None):
        """Forward pass for single timestep"""
        x = self.embed(hist_events, density)
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
