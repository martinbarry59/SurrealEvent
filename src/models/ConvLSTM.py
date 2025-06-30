import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstovoxel
class EConvlstm(nn.Module):
    def __init__(self, input_channels = 2, model_type = "CONVLSTM", width=346, height=260):
        super().__init__()
        self.width = width
        self.height = height
        self.model_type = model_type
        
        self.method = "add"
        # Embed each event (t, x, y, p)
        self.encoder = Encoder( 5)
        self.encoder_channels = [32, 24, 32, 64, 1280]
        
        self.mheight = 9
        self.mwidth = 11

        if "LSTM" in self.model_type:
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
        else:
            self.estimated_depth = None
    def detach_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.detach_hidden()
        else:
            self.estimated_depth = self.estimated_depth.detach()
   
   
    def forward(self, event_sequence, mask_sequence):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        
        lstm_inputs = []
        timed_features = []

        for events in event_sequence:
            # normalise t per batch
            
            min_t = torch.min(events[:, :, 0], dim=1, keepdim=True)[0]
            max_t = torch.max(events[:, :, 0], dim=1, keepdim=True)[0]
            denom = (max_t - min_t)
            # Avoid division by zero, but only where denom is zero
            denom[denom < 1e-8] = 1.0  # If all times are the same, set denom to 1 to avoid NaN
            events[:, :, 0] = (events[:, :, 0] - min_t) / denom
            
            hist_events = eventstovoxel(events, self.height, self.width)
            CNN_encoder, feats = self.encoder(hist_events)
            timed_features.append(feats)
        # Concatenate the outputs from the transformer and CNN
            interpolated = F.interpolate(CNN_encoder, size=(self.mheight, self.mwidth), mode='bilinear', align_corners=False)

            lstm_inputs.append(interpolated)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        encodings = self.convlstm(lstm_inputs)
        # Decode to full self.resolution depth map
        outputs = []
        
        for t in range(encodings.shape[1]):
            x = self.decoder(encodings[:, t], timed_features[t])
        
            outputs.append(self.final_conv(x))
        outputs = torch.cat(outputs, dim=1)

        return outputs, encodings.detach()
