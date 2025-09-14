import torch
import torch.nn as nn
import torch.nn.functional as F
from .EventSurrealLayers import ConvLSTM
from torch.nn import LSTM
from utils.functions import eventstovoxel

class EventEncoder(nn.Module):
    """Lightweight encoder specifically designed for event data"""
    def __init__(self, input_channels=5):
        super().__init__()
        
        # Much smaller channel progression for sparse event data
        self.encoder_channels = [16, 32, 64, 128, 256]
        
        # Depthwise separable convolutions for efficiency
        self.conv1 = self._make_depthwise_block(input_channels, 16, stride=1)
        self.conv2 = self._make_depthwise_block(16, 32, stride=2)
        self.conv3 = self._make_depthwise_block(32, 64, stride=2)
        self.conv4 = self._make_depthwise_block(64, 128, stride=2)
        self.conv5 = self._make_depthwise_block(128, 256, stride=2)
        
        # Lightweight attention for event features
        self.channel_attention = nn.ModuleList([
            ChannelAttention(ch) for ch in self.encoder_channels
        ])
        
    def _make_depthwise_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.GELU(),  # Better for sparse data than ReLU
            # Pointwise convolution
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    
    def forward(self, x):
        features = []
        
        x1 = self.conv1(x)
        x1 = self.channel_attention[0](x1)
        features.append(x1)
        
        x2 = self.conv2(x1)
        x2 = self.channel_attention[1](x2)
        features.append(x2)
        
        x3 = self.conv3(x2)
        x3 = self.channel_attention[2](x3)
        features.append(x3)
        
        x4 = self.conv4(x3)
        x4 = self.channel_attention[3](x4)
        features.append(x4)
        
        x5 = self.conv5(x4)
        x5 = self.channel_attention[4](x5)
        
        return x5, features

class ChannelAttention(nn.Module):
    """Lightweight channel attention for sparse event features"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.GELU(),
            nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w

class EventNormalization(nn.Module):
    """Specialized normalization for event data"""
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, x):
        # Compute sparsity mask
        mask = (x.abs().sum(dim=1, keepdim=True) > self.epsilon).float()
        
        # Normalize only non-zero regions
        mean = x.sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[2, 3], keepdim=True) + self.epsilon)
        var = ((x - mean) ** 2 * mask).sum(dim=[2, 3], keepdim=True) / (mask.sum(dim=[2, 3], keepdim=True) + self.epsilon)
        
        return (x - mean) / (var.sqrt() + self.epsilon) * mask

class EfficientDecoder(nn.Module):
    """Efficient decoder for event data"""
    def __init__(self, encoder_channels, method="add"):
        super().__init__()
        self.method = method
        
        # Reverse the channel list for upsampling
        channels = encoder_channels[::-1]
        
        self.decoder_layers = nn.ModuleList()
        
        for i in range(len(channels)-1):
            in_ch = channels[i]
            skip_ch = channels[i+1]
            
            if method == "concatenate":
                # Account for concatenation
                conv_in_ch = in_ch + skip_ch
            else:
                conv_in_ch = in_ch
                
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=2, stride=2),
                    nn.BatchNorm2d(skip_ch),
                    nn.GELU(),
                    nn.Conv2d(conv_in_ch if method == "concatenate" else skip_ch, 
                             skip_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(skip_ch),
                    nn.GELU()
                )
            )
        
    def forward(self, x, features):
        # features are in forward order, we need them in reverse
        skip_features = features[::-1][1:]  # Exclude the bottleneck
        
        for i, (layer, skip_feat) in enumerate(zip(self.decoder_layers, skip_features)):
            # Upsample
            x = layer[0](x)  # ConvTranspose2d
            x = layer[1](x)  # BatchNorm2d
            x = layer[2](x)  # GELU
            
            # Adjust size to match skip connection
            if x.shape[-2:] != skip_feat.shape[-2:]:
                x = F.interpolate(x, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            # Apply skip connection
            if self.method == "concatenate":
                x = torch.cat([x, skip_feat], dim=1)
            else:
                x = x + skip_feat
                
            # Final conv layers
            x = layer[3](x)  # Conv2d
            x = layer[4](x)  # BatchNorm2d
            x = layer[5](x)  # GELU
            
        return x

class EfficientConvLSTM(nn.Module):
    def __init__(self, model_type="CONVLSTM", width=346, height=260, skip_lstm=True):
        super().__init__()
        self.width = width
        self.height = height
        self.model_type = model_type
        self.skip_lstm = skip_lstm 
        self.method = "add"
        
        # Event-specific preprocessing
        self.event_norm = EventNormalization()
        
        # Lightweight event encoder
        self.encoder = EventEncoder(5)
        self.encoder_channels = [16, 32, 64, 128, 256]
        
        # Adjust bottleneck accordingly
        self.mheight = 9
        self.mwidth = 11
        self.bottleneck_channels = 64  # Reduced from 128
        
        # Add gradient monitoring
        self.gradient_clip_val = 1.0
        
        if "LSTM" in self.model_type:
            if self.skip_lstm:
                self.skip_convlstms = nn.ModuleList([
                    ConvLSTM(
                        input_dim=ch,
                        hidden_dims=[ch//2],  # Reduce hidden dims to prevent overfitting
                        kernel_size=3,
                        num_layers=1
                    ) for ch in self.encoder_channels[:-1]
                ])
                
            if "DENSELSTM" in self.model_type:
                self.dense_lstm = LSTM(
                    input_size=self.encoder_channels[-1]*self.mheight*self.mwidth,
                    hidden_size=64,  # Reduced from 128
                    num_layers=2,
                    batch_first=True,
                    dropout=0.1
                )
                self.fc_to_map = nn.Linear(64, self.bottleneck_channels * self.mheight * self.mwidth)
                self.dense_state = None
                
            elif "CONVLSTM" in self.model_type:
                self.convlstm = ConvLSTM(
                    input_dim=self.encoder_channels[-1],
                    hidden_dims=[64, 64],  # Reduced from [128, 128]
                    kernel_size=3,
                    num_layers=2
                )
                
            # Update encoder channels for skip connections
            if self.skip_lstm:
                self.encoder_channels = [ch//2 for ch in self.encoder_channels[:-1]] + [64]
            else:
                self.encoder_channels = self.encoder_channels[:-1] + [64]
        else: 
            self.estimated_depth = None

        # Efficient decoder
        self.decoder = EfficientDecoder(self.encoder_channels, self.method)
        
        # Final output layers with better initialization
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels[0], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to avoid gradient collapse"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Special initialization for final conv to prevent sigmoid saturation
        final_conv = self.final_conv[-2]  # The last Conv2d before Sigmoid
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -1.0)  # Slight negative bias
            
    def reset_states(self):
        if "DENSELSTM" in self.model_type:
            self.dense_state = None
        elif "CONVLSTM" in self.model_type:
            self.convlstm.reset_hidden()
        if "LSTM" in self.model_type and self.skip_lstm:
            for skip_lstm in self.skip_convlstms:
                skip_lstm.reset_hidden()
        else:
            self.estimated_depth = None
            
    def detach_states(self):
        if "LSTM" in self.model_type:
            if "DENSELSTM" in self.model_type:
                if self.dense_state is not None:
                    h, c = self.dense_state
                    self.dense_state = (h.detach(), c.detach())
            elif "CONVLSTM" in self.model_type:
                self.convlstm.detach_hidden()
            if self.skip_lstm:
                for skip_lstm in self.skip_convlstms:
                    skip_lstm.detach_hidden()
        else:
            if self.estimated_depth is not None:
                self.estimated_depth = self.estimated_depth.detach()

    def forward(self, event_sequence, training=False, hotpixel=False):
        lstm_inputs = []
        timed_features = [[] for _ in range(len(self.encoder_channels))]
        seq_events = []
        
        for events in event_sequence:
            # Normalize event coordinates
            with torch.no_grad():
                if events.shape[-1] == 4:
                    min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
                    max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
                    denom = (max_t - min_t)
                    denom[denom < 1e-8] = 1.0
                    events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                    events[:, :, 1] = events[:, :, 1].clamp(0, self.width-1)
                    events[:, :, 2] = events[:, :, 2].clamp(0, self.height-1)

                    hist_events = eventstovoxel(events, self.height, self.width, 
                                              training=training, hotpixel=hotpixel).float()
                    seq_events.append(hist_events)
                else:
                    hist_events = events

            # Apply event normalization
            hist_events = self.event_norm(hist_events)
            
            # Encode with lightweight encoder
            CNN_encoder, feats = self.encoder(hist_events)
            for i, f in enumerate(feats):
                timed_features[i].append(f)
            
            # Prepare for LSTM
            interpolated = F.interpolate(CNN_encoder, size=(self.mheight, self.mwidth), 
                                       mode='bilinear', align_corners=False)
            lstm_inputs.append(interpolated)
            
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        
        # Apply skip LSTMs if enabled
        skip_outputs = []
        if self.skip_lstm:
            for i, skip_lstm in enumerate(self.skip_convlstms):
                skip_feats = torch.stack(timed_features[i], dim=1)
                skip_out = skip_lstm(skip_feats)
                skip_outputs.append(skip_out)
         
        B, T, Cb, Hb, Wb = lstm_inputs.shape
        
        # # Monitor for gradient issues
        # if self.training and torch.rand(1) < 0.1:
        #     print(f"LSTM input stats - Mean: {lstm_inputs.mean():.6f}, Std: {lstm_inputs.std():.6f}")
        #     print(f"LSTM input range - Min: {lstm_inputs.min():.6f}, Max: {lstm_inputs.max():.6f}")
        
        # Apply main LSTM
        if "DENSELSTM" in self.model_type:
            x_seq = lstm_inputs.flatten(2)
            if self.dense_state is None:
                enc_seq, self.dense_state = self.dense_lstm(x_seq)
            else:
                enc_seq, self.dense_state = self.dense_lstm(x_seq, self.dense_state)
            
            enc_flat = self.fc_to_map(enc_seq.reshape(B * T, -1))
            encodings = enc_flat.view(B, T, self.bottleneck_channels, self.mheight, self.mwidth)
            
        elif "CONVLSTM" in self.model_type:
            encodings = self.convlstm(lstm_inputs)
        
        # Monitor encoding collapse
        # if self.training and torch.rand(1) < 0.1:
        #     print(f"Encoding stats - Mean: {encodings.mean():.6f}, Std: {encodings.std():.6f}")
        #     print(f"Encoding range - Min: {encodings.min():.6f}, Max: {encodings.max():.6f}")
        #     zero_ratio = (encodings.abs() < 1e-6).float().mean()
        #     print(f"Zero ratio: {zero_ratio:.3f}")
        #     if zero_ratio > 0.9:
        #         print("WARNING: Encodings are collapsing to zero!")
        
        # Decode to full resolution
        outputs = []
        for t in range(T):
            if self.skip_lstm:
                skip_feats_t = [skip_outputs[i][:, t] for i in range(len(skip_outputs))]
            else:
                skip_feats_t = [timed_features[i][t] for i in range(len(timed_features))]
                
            x = self.decoder(encodings[:, t], skip_feats_t)
            outputs.append(self.final_conv(x))

        outputs = torch.cat(outputs, dim=1)
        return outputs, encodings.detach(), seq_events
