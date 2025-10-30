import torch
import torch.nn as nn
import torch.nn.functional as F
from .EventSurrealLayers import ConvLSTMCell
from utils.functions import eventstovoxel

class ResidualBlock(nn.Module):
    """Simple residual block with optional downsampling"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # Residual connection
        out = F.relu(out)
        
        return out

class FastEncoder(nn.Module):
    """Fast encoder with residual connections and skip features"""
    def __init__(self, input_channels=5):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks with downsampling
        self.layer1 = ResidualBlock(32, 64, stride=2)    # 173x130 -> 87x65
        self.layer2 = ResidualBlock(64, 128, stride=2)   # 87x65 -> 44x33
        self.layer3 = ResidualBlock(128, 256, stride=2)  # 44x33 -> 22x17
        
        # Additional residual blocks at same resolution for depth
        self.layer4 = ResidualBlock(256, 256, stride=1)  # 22x17 -> 22x17
        
    def forward(self, x):
        # Collect skip features at different resolutions like original ConvLSTM
        feats = []
        
        x = F.relu(self.bn1(self.conv1(x)))  # 346x260 -> 173x130
        feats.append(x)  # Skip feature 1: [B, 32, 173, 130]
        
        x = self.layer1(x)  # -> 87x65
        feats.append(x)  # Skip feature 2: [B, 64, 87, 65]
        
        x = self.layer2(x)  # -> 44x33
        feats.append(x)  # Skip feature 3: [B, 128, 44, 33]
        
        x = self.layer3(x)  # -> 22x17
        feats.append(x)  # Skip feature 4: [B, 256, 22, 17]
        
        x = self.layer4(x)  # -> 22x17 (same size, more depth)
        
        return x, feats  # Return both bottleneck and skip features

class FastDecoder(nn.Module):
    """Fast decoder with skip connections like the original ConvLSTM"""
    def __init__(self, input_channels=128, encoder_channels=[32, 64, 128, 256], method="add"):
        super().__init__()
        self.method = method
        
        # Upsampling layers that will use skip connections
        self.up_layers = nn.ModuleList()
        
        # From 22x17 to 44x33 (will skip with 128 channels)
        self.up_layers.append(self._make_up_layer(input_channels, 128))
        
        # From 44x33 to 87x65 (will skip with 64 channels) 
        if method == "concatenate":
            self.up_layers.append(self._make_up_layer(128 + 128, 64))  # 128 from up + 128 from skip
        else:
            self.up_layers.append(self._make_up_layer(128, 64))
        
        # From 87x65 to 173x130 (will skip with 32 channels)
        if method == "concatenate":
            self.up_layers.append(self._make_up_layer(64 + 64, 32))  # 64 from up + 64 from skip
        else:
            self.up_layers.append(self._make_up_layer(64, 32))
        
        # Final upsampling to full resolution
        if method == "concatenate":
            final_in_ch = 32 + 32  # 32 from up + 32 from skip
        else:
            final_in_ch = 32
            
        self.final_up = nn.ConvTranspose2d(final_in_ch, 16, 4, stride=2, padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )
        
    def _make_up_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip_feats):
        # skip_feats order: [32ch@173x130, 64ch@87x65, 128ch@44x33, 256ch@22x17]
        # We need them in reverse order for upsampling (skip the last one which is same as bottleneck)
        skip_feats_for_decoder = skip_feats[::-1][1:]  # [128ch@44x33, 64ch@87x65, 32ch@173x130]
        
        # Upsample with skip connections
        for i, (up_layer, skip_feat) in enumerate(zip(self.up_layers, skip_feats_for_decoder)):
            x = up_layer[:3](x)  # ConvTranspose + BN + ReLU
            
            # Adjust size to match skip feature
            if x.shape[-2:] != skip_feat.shape[-2:]:
                x = F.interpolate(x, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            # Apply skip connection
            if self.method == "concatenate":
                x = torch.cat([x, skip_feat], dim=1)
            else:
                x = x + skip_feat  # Element-wise addition (residual-style)
                
            x = up_layer[3:](x)  # Final conv + BN + ReLU
        
        # Final upsampling to target resolution
        x = self.final_up(x)  # 173x130 -> 346x260
        x = self.final_conv(x)
        
        # Ensure exact output size
        return F.interpolate(x, size=(260, 346), mode='bilinear', align_corners=False)

class EfficientConvLSTM(nn.Module):
    """Fast model with skip connections like original ConvLSTM"""
    def __init__(self, model_type="FASTLSTM", width=346, height=260, skip_connections=True, method="add"):
        super().__init__()
        self.width = width
        self.height = height
        self.model_type = model_type
        self.skip_connections = skip_connections
        self.method = method
        
        # Encoder with skip features
        self.encoder = FastEncoder(5)
        
        # Single LSTM at bottleneck (22x17 resolution)
        self.lstm_cell = ConvLSTMCell(256, 128, kernel_size=3)
        self.hidden_state = None
        
        # Decoder with skip connections
        self.decoder = FastDecoder(128, encoder_channels=[32, 64, 128, 256], method=method)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def reset_states(self):
        self.hidden_state = None
        
    def detach_states(self):
        if self.hidden_state is not None:
            h, c = self.hidden_state
            self.hidden_state = (h.detach(), c.detach())

    def forward(self, event_sequence, training=False, hotpixel=False):
        B = event_sequence[0].shape[0]
        outputs = []
        all_skip_feats = []  # Store skip features for each timestep
        
        for events in event_sequence:
            # Simple time normalization (vectorized)
            with torch.no_grad():
                if events.shape[-1] == 4:
                    # Fast normalization
                    min_t = events[:, :, 0].min(dim=1, keepdim=True)[0]
                    max_t = events[:, :, 0].max(dim=1, keepdim=True)[0]
                    denom = max_t - min_t + 1e-8
                    events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                    events[:, :, 1] = events[:, :, 1].clamp(0, self.width-1)
                    events[:, :, 2] = events[:, :, 2].clamp(0, self.height-1)
                    
                    # Convert to voxels
                    hist_events = eventstovoxel(events, self.height, self.width).float()
                else:
                    hist_events = events
            
            # Encode with skip features
            features, skip_feats = self.encoder(hist_events)  # [B, 256, 22, 17], skip_feats
            all_skip_feats.append(skip_feats)
            
            # Initialize hidden state if needed
            if self.hidden_state is None:
                _, _, H, W = features.shape
                self.hidden_state = (
                    torch.zeros(B, 128, H, W, device=features.device),
                    torch.zeros(B, 128, H, W, device=features.device)
                )
            
            # Apply LSTM
            h_new, c_new = self.lstm_cell(features, self.hidden_state)
            self.hidden_state = (h_new, c_new)
            
            # Decode with skip connections
            if self.skip_connections:
                output = self.decoder(h_new, skip_feats)  # [B, 1, 260, 346]
            else:
                output = self.decoder(h_new, [])  # No skip connections
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.cat(outputs, dim=1)  # [B, T, 260, 346]
        
        # Return dummy encodings for compatibility
        dummy_encodings = h_new.unsqueeze(1).repeat(1, len(event_sequence), 1, 1, 1)
        
        return outputs, dummy_encodings.detach(), event_sequence
