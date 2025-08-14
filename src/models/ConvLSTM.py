import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstovoxel
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

    def forward(self, event_sequence, mask_sequence = None):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        
        lstm_inputs = []
        timed_features = [[] for _ in range(len(self.encoder_channels))]  # For skip connections
        for events in event_sequence:
            # normalise t per batch
            with torch.no_grad():
                min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
                max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
                denom = (max_t - min_t)
                # Avoid division by zero, but only where denom is zero
                denom[denom < 1e-8] = 1.0  # If all times are the same, set denom to 1 to avoid NaN
                events[:, :, 0] = (events[:, :, 0] - min_t) / denom
                hist_events = eventstovoxel(events, self.width, self.height)

            CNN_encoder, feats = self.encoder(hist_events)
            for i, f in enumerate(feats):
                timed_features[i].append(f)
            
        # Concatenate the outputs from the transformer and CNN
            interpolated = F.interpolate(CNN_encoder, size=(self.mheight, self.mwidth), mode='bilinear', align_corners=False)
            lstm_inputs.append(interpolated)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        skip_outputs = []
        if self.skip_lstm:
            for i, skip_lstm in enumerate(self.skip_convlstms):
                # Stack features for this skip level: [B, T, C, H, W]
                skip_feats = torch.stack(timed_features[i], dim=1)
                skip_out = skip_lstm(skip_feats)  # Output: [B, T, C, H, W]
                skip_outputs.append(skip_out.clone())
        encodings = self.convlstm(lstm_inputs)
        # Decode to full self.resolution depth map
        outputs = []
        
        for t in range(encodings.shape[1]):
            if self.skip_lstm:
                skip_feats_t = [skip_outputs[i][:, t] for i in range(len(skip_outputs))]
            else:
                skip_feats_t = [timed_features[i][t] for i in range(len(timed_features))]
            x = self.decoder(encodings[:, t], skip_feats_t)
        
            outputs.append(self.final_conv(x))
        outputs = torch.cat(outputs, dim=1)
        del lstm_inputs, skip_outputs
        return outputs, encodings.detach()
