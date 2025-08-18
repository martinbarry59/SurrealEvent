import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstovoxel
class EConvlstm(nn.Module):
    def __init__(self, input_channels = 2, model_type = "CONVLSTM", width=346, height=260, skip_lstm=True):

        super().__init__()
        self.bins = 5
        self.width = width
        self.height = height
        self.model_type = model_type
        self.skip_lstm = skip_lstm 
        self.method = "add"
        # Embed each event (t, x, y, p)
        self.encoder = Encoder(2 * self.bins)
        self.encoder_channels = [32, 24, 32, 64, 1280]
        
        self.mheight = 9
        self.mwidth = 11
        self.voxel_bn = torch.nn.GroupNorm(num_groups=1, num_channels=2 * self.bins)


        if "LSTM" in self.model_type:
            if self.skip_lstm:
                self.skip_convlstms = nn.ModuleList([
                ConvLSTM(
                    input_dim=ch,
                    hidden_dims=[ch],
                    kernel_size=3,
                    num_layers=1
                ) for ch in self.encoder_channels[:-1]
            ])
            self.convlstm = ConvLSTM(
                input_dim=self.encoder_channels[4],
                hidden_dims=[128, 128],
                kernel_size=3,
                num_layers=2
            )
            self.encoder_channels = self.encoder_channels[:-1] + [128]
        else: 
            self.estimated_depth = None


        self.decoder = Decoder(self.encoder_channels, self.method)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        ) 

        # self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to avoid sigmoid saturation"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for better gradient flow
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  
        final_conv = self.final_conv[-2]  # The last Conv2d before Sigmoid
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)  # Small weights
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -2.0)  # Negative bias pushes sigmoid away from 0.5      
    def reset_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.reset_hidden()
            if self.skip_lstm:
                for skip_lstm in self.skip_convlstms:
                    skip_lstm.reset_hidden()
        else:
            self.estimated_depth = None
    def detach_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.detach_hidden()
            if self.skip_lstm:
                for skip_lstm in self.skip_convlstms:
                    skip_lstm.detach_hidden()
        else:
            self.estimated_depth = self.estimated_depth.detach()
   
   
    def forward(self, event_sequence, training=False, hotpixel=False):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        
        lstm_inputs = []
        timed_features = []  # For skip connections
        seq_events = []
        for events in event_sequence:
            # normalise t per batch
            with torch.no_grad():
                if events.shape[-1] == 4:
                    
                    min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
                    max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
                    denom = (max_t - min_t)
                    # Avoid division by zero, but only where denom is zero
                    # print(f"Target time range: [{events[:, :, 0].min():.3f}, {events[:, :, 0].max():.3f}]")
                    # print(f"Target x range: [{events[:, :, 1].min():.3f}, {events[:, :, 1].max():.3f}]")
                    # print(f"Target y range: [{events[:, :, 2].min():.3f}, {events[:, :, 2].max():.3f}]")
                    # print(f"Target polarity range: [{events[:, :, 3].min():.3f}, {events[:, :, 3].max():.3f}]")
                    denom[denom < 1e-8] = 1.0  # If all times are the same, set denom to 1 to avoid NaN
                    events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                    events[:,:, 1] = events[:, :, 1].clamp(0, self.width-1)
                    events[:,:, 2] = events[:, :, 2].clamp(0, self.height-1)
                    hist_events = eventstovoxel(events, self.height, self.width, training = training, hotpixel=hotpixel).float()
                    seq_events.append(hist_events)
                else:
                    hist_events = events
        events = torch.stack(seq_events, dim=0).permute(1, 0, 2, 3, 4).contiguous()  # [B, T, C, H, W]
        B, T, C, H, W = events.shape
        
        batch_flatten = events.view(B*T, C, H, W)  # Flatten batch and time
        hist_events = self.voxel_bn(batch_flatten)
        
        CNN_encoder, feats = self.encoder(hist_events)
       
        timed_features = [feat.view(B, T, *feat.shape[1:]).permute(1,0,2, 3, 4) for feat in feats]  # [T, B, C, H, W]
        
    # Concatenate the outputs from the transformer and CNN
        interpolated = F.interpolate(CNN_encoder, size=(self.mheight, self.mwidth), mode='bilinear', align_corners=False)
        interpolated = interpolated.view(B, T, *interpolated.shape[1:])  # Reshape back to [B, T, C, H, W]
        
        lstm_inputs = interpolated.permute(1,0,2, 3, 4)  # [B, C, T, H, W]
        skip_outputs = []
        if self.skip_lstm:
            for i, skip_lstm in enumerate(self.skip_convlstms):
                # Stack features for this skip level: [B, T, C, H, W]
                skip_out = skip_lstm(timed_features[i])  # Output: [B, T, C, H, W]
                skip_outputs.append(skip_out.clone())
        encodings = self.convlstm(lstm_inputs)
        flatten_encodings = encodings.view(B*T, *encodings.shape[2:])  # [B, T, C, H, W]
        # Decode to full self.resolution depth map
        outputs = []
        
        flatten_features = [f.view(B*T, *f.shape[2:]) for f in skip_outputs]  # Flatten time dimension
        
        x = self.decoder(flatten_encodings, flatten_features)
        
        outputs = self.final_conv(x)  # [T*B, C, H, W]
        outputs = outputs.view(B,T,H,W)
        return outputs, encodings.detach(), seq_events
