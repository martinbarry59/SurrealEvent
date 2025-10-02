import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.functions import eventstovoxel

"""
EventSegUltraLight: Ultra-lightweight human segmentation for event cameras.
Design principles:
- MINIMAL memory footprint: aggressive downsampling, tiny channels
- Fast inference: depthwise separable only, no complex blocks
- Streaming friendly: minimal state, frequent detaching
- Target: <50MB memory per timestep for batch=10

Key differences from EventSegFast:
- Much smaller channels (8-16 instead of 24-96)
- Early aggressive downsampling (4x immediately)
- No multi-scale fusion - single scale processing
- Minimal temporal state
- Direct upsampling at end
"""

class TinyDepthwise(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True) if stride == 1 else nn.Identity()
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride) if in_ch != out_ch or stride != 1 else nn.Identity()
        
    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class MinimalTemporalState(nn.Module):
    """Ultra-minimal temporal state - just EMA, no RNN"""
    def __init__(self, channels, decay=0.8):
        super().__init__()
        self.decay = decay
        self.register_buffer('state', None, persistent=False)
        
    def reset(self):
        self.state = None
        
    def detach(self):
        if self.state is not None:
            self.state = self.state.detach()
            
    def forward(self, x):
        if self.state is None:
            self.state = x.detach()
        else:
            # Simple EMA update
            self.state = self.decay * self.state.detach() + (1 - self.decay) * x.detach()
        
        # Minimal mixing - just add scaled memory
        return x + 0.2 * self.state.detach()

class EventSegUltraLight(nn.Module):
    def __init__(self, voxel_channels=5, height=260, width=346):
        super().__init__()
        self.height = height
        self.width = width
        
        # Immediate aggressive downsampling to save memory
        self.stem = nn.Sequential(
            nn.Conv2d(voxel_channels, 8, 7, 4, 3, bias=False),  # 4x downsample immediately
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True)
        )
        
        # Minimal temporal state at low resolution
        self.temporal = MinimalTemporalState(8)
        
        # Tiny encoder - only 3 blocks, small channels
        self.enc1 = TinyDepthwise(8, 12, stride=1)    # 65x86 -> 65x86
        self.enc2 = TinyDepthwise(12, 16, stride=2)   # 65x86 -> 32x43  
        self.enc3 = TinyDepthwise(16, 20, stride=2)   # 32x43 -> 16x21
        
        # Simple decoder - direct upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(20, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            nn.Conv2d(8, 1, 1)
        )
        
    def reset_states(self):
        self.temporal.reset()
        
    def detach_states(self):
        self.temporal.detach()
        
    def forward(self, event_sequence, training=False, hotpixel=False, density=None):
        if not training:
            return self._inference_mode(event_sequence, hotpixel)
            
        # Training mode
        outputs = []
        seq_events = []
        
        for i, events in enumerate(event_sequence):
            # Process events to voxel
            with torch.no_grad():
                min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
                max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
                denom = torch.clamp(max_t - min_t, min=1e-8)
                
                events = events.clone()
                events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                events[:, :, 1] = torch.clamp(events[:, :, 1], 0, self.width-1)
                events[:, :, 2] = torch.clamp(events[:, :, 2], 0, self.height-1)
                
                voxel = eventstovoxel(events, self.height, self.width, training=training, hotpixel=hotpixel).float()
                seq_events.append(voxel.detach())
            
            # Forward pass
            out = self._forward_single(voxel)
            outputs.append(out)
            
           
        outputs = torch.stack(outputs, dim=1).squeeze(2)
        return outputs, None, seq_events
    
    def _inference_mode(self, event_sequence, hotpixel=False):
        """Memory-efficient inference"""
        outputs = []
        seq_events = []
        
        for i, events in enumerate(event_sequence):
            with torch.no_grad():
                # Process events
                min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
                max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
                denom = torch.clamp(max_t - min_t, min=1e-8)
                
                events = events.clone()
                events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                events[:, :, 1] = torch.clamp(events[:, :, 1], 0, self.width-1)
                events[:, :, 2] = torch.clamp(events[:, :, 2], 0, self.height-1)
                
                voxel = eventstovoxel(events, self.height, self.width, training=False, hotpixel=hotpixel).float()
                seq_events.append(voxel)
                
                # Forward and immediately move to CPU
                out = self._forward_single(voxel)
                outputs.append(out.cpu())
                
                # Aggressive memory cleanup
                del voxel, events
                if i % 1 == 0:  # Every step
                    torch.cuda.empty_cache()
                    self.detach_states()
                    
        # Move back to GPU only at the end
        outputs = torch.stack([o.cuda() for o in outputs], dim=1).squeeze(2)
        return outputs, None, seq_events
    
    def _forward_single(self, voxel):
        """Single timestep forward pass"""
        # Downsample immediately to save memory
        x = self.stem(voxel)  # [B, 8, H/4, W/4]
        
        # Add temporal context
        x = self.temporal(x)
        
        # Encode at low resolution
        x = self.enc1(x)  # [B, 12, H/4, W/4]
        x = self.enc2(x)  # [B, 16, H/8, W/8]  
        x = self.enc3(x)  # [B, 20, H/16, W/16]
        
        # Decode
        x = self.decoder(x)  # [B, 1, H/16, W/16]
        
        # Upsample to original resolution
        x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(x)

# Memory analysis
def analyze_ultralight_memory(batch_size=10, height=260, width=346):
    """Estimate memory usage"""
    voxel_size = batch_size * 5 * height * width * 4  # Input voxels
    
    # After stem: 4x downsampled
    stem_size = batch_size * 8 * (height//4) * (width//4) * 4
    
    # Encoder stages
    enc1_size = batch_size * 12 * (height//4) * (width//4) * 4
    enc2_size = batch_size * 16 * (height//8) * (width//8) * 4  
    enc3_size = batch_size * 20 * (height//16) * (width//16) * 4
    
    # Output
    output_size = batch_size * 1 * height * width * 4
    
    total = voxel_size + stem_size + enc1_size + enc2_size + enc3_size + output_size
    
    print(f"UltraLight Memory Analysis (batch={batch_size}):")
    print(f"  Voxel input: {voxel_size / (1024**3):.3f} GB")
    print(f"  Stem (8ch, H/4): {stem_size / (1024**3):.3f} GB")
    print(f"  Enc1 (12ch, H/4): {enc1_size / (1024**3):.3f} GB")
    print(f"  Enc2 (16ch, H/8): {enc2_size / (1024**3):.3f} GB")
    print(f"  Enc3 (20ch, H/16): {enc3_size / (1024**3):.3f} GB")
    print(f"  Output: {output_size / (1024**3):.3f} GB")
    print(f"  Total per timestep: {total / (1024**3):.3f} GB")
    print(f"  5 timesteps: {5 * total / (1024**3):.3f} GB")

if __name__ == '__main__':
    model = EventSegUltraLight()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")
    print()
    
    analyze_ultralight_memory()
