from asyncio import events
import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstovoxel, normalize_event_times_vectorized
class EConvlstm(nn.Module):
    def __init__(self,model_type = "CONVLSTM", width=346, height=260, skip_lstm=True):

        super().__init__()
        self.width = width
        self.height = height
        self.model_type = model_type
        self.skip_lstm = skip_lstm 
        self.method = "add"
        # Embed each event (t, x, y, p)
        self.encoder = Encoder(5)
        self.encoder_channels = [32, 24, 32, 64, 1280]
        
        self.mheight = 9
        self.mwidth = 11

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
   
    def print_statistics(self, hist_events, events):
        """ Print complete set of different statistics for debugging """
        print("Histogram Events:")
        for i, hist in enumerate(hist_events):
            print(f"  Frame {i}: {hist.shape}")
        print("Original Events:")
        print(f"  Shape: {events.shape}")
        print(f"  Min t: {events[:, :, 0].min().item()}, Max t: {events[:, :, 0].max().item()}, Mean t: {events[:, :, 0].mean().item()}, Std t: {events[:, :, 0].std().item()}")
        print(f"  Min x: {events[:, :, 1].min().item()}, Max x: {events[:, :, 1].max().item()}, Mean x: {events[:, :, 1].mean().item()}, Std x: {events[:, :, 1].std().item()}")
        print(f"  Min y: {events[:, :, 2].min().item()}, Max y: {events[:, :, 2].max().item()}, Mean y: {events[:, :, 2].mean().item()}, Std y: {events[:, :, 2].std().item()}")
        print(f"  Min p: {events[:, :, 3].min().item()}, Max p: {events[:, :, 3].max().item()}, Mean p: {events[:, :, 3].mean().item()}, Std p: {events[:, :, 3].std().item()}")
        for hist in hist_events:
            print(f"  Histogram shape: {hist.shape}, Min: {hist.min().item()}, Max: {hist.max().item()}")
            print(f"  Histogram mean: {hist.mean().item()}, std: {hist.std().item()}")

    def robust_normalize(self, x, percentile=95):
        """Camera-agnostic robust normalization"""
        B = x.size(0)
        x_flat = x.view(B, -1)
        
        x_min = torch.quantile(x_flat, 0.05, dim=1, keepdim=True).view(B, 1, 1, 1)
        x_max = torch.quantile(x_flat, percentile/100, dim=1, keepdim=True).view(B, 1, 1, 1)

        return 2 * torch.clamp((x - x_min) / (x_max - x_min + 1e-8), 0, 1) - 1
    def forward(self, event_sequence, training=False, hotpixel=False):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        
        # Vectorized time normalization across all timesteps
        with torch.no_grad():
            events = torch.stack(event_sequence, dim=1)  # [B, T, N, 4]
            B, T, N, _ = events.shape
            if events.shape[-1] == 4:
                # Vectorized time normalization
                events = normalize_event_times_vectorized(events)
                
                # Clamp spatial coordinates
                events[:, :, :, 1] = events[:, :, :, 1].clamp(0, self.width-1)
                events[:, :, :, 2] = events[:, :, :, 2].clamp(0, self.height-1)
                
                # Vectorized voxel conversion: [B, T, N, 4] -> [B, T, C, H, W]
                hist_events = eventstovoxel(events, self.height, self.width, bins=self.bins, training=training, hotpixel=hotpixel).float()
            else:
                # If events are already processed, just use them
                hist_events = events
        events = hist_events  # [B, T, C, H, W]
        B, T, C, H, W = events.shape
        
        batch_flatten = events.view(B*T, C, H, W)  # Flatten batch and time
        hist_events = self.voxel_bn(batch_flatten)
        
        CNN_encoder, feats = self.encoder(hist_events)
               
    # Concatenate the outputs from the transformer and CNN
        interpolated = F.interpolate(CNN_encoder, size=(self.mheight, self.mwidth), mode='bilinear', align_corners=False)
        interpolated = interpolated.view(B, T, *interpolated.shape[1:])  # Reshape back to [B, T, C, H, W]
        

        skip_outputs = []
        if self.skip_lstm:
            for i, skip_lstm in enumerate(self.skip_convlstms):
                # Stack features for this skip level: [B, T, C, H, W]
                skip_out = skip_lstm(feats[i].view(B, T, *feats[i].shape[1:]))  # Output: [B, T, C, H, W]
                skip_outputs.append(skip_out.clone())
        encodings = self.convlstm(interpolated)
        flatten_encodings = encodings.view(B*T, *encodings.shape[2:])  # [B, T, C, H, W]
        # Decode to full self.resolution depth map
        outputs = []
        
        

        outputs.append(self.final_conv(x))
        outputs = torch.cat(outputs, dim=1)
        print(f"Final outputs shape: {outputs.shape}, min: {outputs.min().item()}, max: {outputs.max().item()}, mean: {outputs.mean().item()}, std: {outputs.std().item()}")
        exit()
        return outputs, encodings.detach(), seq_events
