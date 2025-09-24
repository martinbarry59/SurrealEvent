import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from torch.nn import LSTM
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
        
    
        self.bottleneck_channels = 128

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
            if "DENSELSTM" in self.model_type:
                self.dense_lstm = LSTM(
                    input_size=self.encoder_channels[-1]*self.mheight*self.mwidth,
                    hidden_size=128,
                    num_layers=2,
                    batch_first=True

                )
                self.fc_to_map = nn.Linear(128, self.bottleneck_channels * self.mheight * self.mwidth)
                self.dense_state = None
            elif "CONVLSTM" in self.model_type:
                self.convlstm = ConvLSTM(
                    input_dim=self.encoder_channels[-1],
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
        if "DENSELSTM" in self.model_type:
            self.dense_state  = None
        elif "CONVLSTM" in self.model_type:
            self.convlstm.reset_hidden()
        if "LSTM" in self.model_type:
                
            if self.skip_lstm:
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
        for hist in hist_events[0]:
            print(f"  Histogram shape: {hist.shape}, Min: {hist.min().item()}, Max: {hist.max().item()}")
            print(f"  Histogram mean: {hist.mean().item()}, std: {hist.std().item()}")

    def forward(self, event_sequence, training=False, hotpixel=False):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        
        lstm_inputs = []
        timed_features = [[] for _ in range(len(self.encoder_channels))]  # For skip connections
        seq_events = []
        for events in event_sequence:
            # normalise t per batch
            with torch.no_grad():
                if events.shape[-1] == 4:
                    times = events[:, :, 0]
                    ### only keep times where polarity events[:, :, 3] != 0
                    non_zero_mask = events[:, :, 3] != 0
                    non_zero_events = times[non_zero_mask]
                    if non_zero_events.numel() != 0:
                        min_t = torch.min(non_zero_events)
                        max_t = torch.max(non_zero_events)
                        denom = (max_t - min_t)
                        # Avoid division by zero, but only where denom is zero
                        
                        ## linear interpolation to send x between 0 and 346
                        # x = events[:, :, 1]
                        # x = ((x - x.min()) / (x.max() - x.min()) * 346)
                        # ## cast x to int
                        # events[:, :, 1] = x.to(torch.uint8)
                        # y = events[:, :, 2]
                        # y = ((y - y.min()) / (y.max() - y.min()) * 260)
                        # events[:, :, 2] = y.to(torch.uint8)
                        # print(f"Target time range: [{events[:, :, 0].min():.3f}, {events[:, :, 0].max():.3f}]")
                        # print(f"Target x range: [{events[:, :, 1].min():.3f}, {events[:, :, 1].max():.3f}]")
                        # print(f"Target y range: [{events[:, :, 2].min():.3f}, {events[:, :, 2].max():.3f}]")
                        # print(f"Target polarity range: [{events[:, :, 3].min():.3f}, {events[:, :, 3].max():.3f}]")
                        denom[denom < 1e-8] = 1.0  # If all times are the same, set denom to 1 to avoid NaN
                        events[:, :, 0] = ((events[:, :, 0] - min_t) / denom).clamp(0, 1)
                        # print("After normalization:")
                        non_zero_events = events[:, :, 0][non_zero_mask]
                        # print(f"min: {non_zero_events.min().item()}, max: {non_zero_events.max().item()}, mean: {non_zero_events.mean().item()}, std: {non_zero_events.std().item()}")
                        events[:,:, 1] = events[:, :, 1].clamp(0, self.width-1)
                        events[:,:, 2] = events[:, :, 2].clamp(0, self.height-1)
                        hist_events = eventstovoxel(events, self.height, self.width, training=training, hotpixel=hotpixel).float() * 0
                    else:
                        hist_events = torch.zeros((events.shape[0], 5, self.height, self.width), device=events.device)
                    self.print_statistics(hist_events, events)
                    # exit()
                    seq_events.append(hist_events)
                else:
                    hist_events = events

            CNN_encoder, feats = self.encoder(hist_events)
            for i, f in enumerate(feats):
                timed_features[i].append(f)
            
        # Concatenate the outputs from the transformer and CNN
            lstm_inputs.append(CNN_encoder)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        skip_outputs = []
        
        if self.skip_lstm:
            for i, skip_lstm in enumerate(self.skip_convlstms):
                # Stack features for this skip level: [B, T, C, H, W]
                skip_feats = torch.stack(timed_features[i], dim=1)
                skip_out = skip_lstm(skip_feats)  # Output: [B, T, C, H, W]
                skip_outputs.append(skip_out.clone())
         
        B, T, Cb, Hb, Wb = lstm_inputs.shape
        # print(f"data statistics mean: {lstm_inputs.mean().item()}, std: {lstm_inputs.std()},  min: {lstm_inputs.min().item()}, max: {lstm_inputs.max().item()}")
        if "DENSELSTM" in self.model_type:
        # 
        # Dense LSTM at the lowest layer
            
            x_seq = lstm_inputs.flatten(2)  # [B, T, Cb*Hb*Wb]
            
            if self.dense_state is None:
                enc_seq, self.dense_state = self.dense_lstm(x_seq)                 # enc_seq: [B, T, 128]
            else:
                enc_seq, self.dense_state = self.dense_lstm(x_seq, self.dense_state)

            # Project back to spatial bottleneck map
            enc_flat = self.fc_to_map(enc_seq.reshape(B * T, -1))                  # [B*T, Cb_out*Hb*Wb]
            encodings = enc_flat.view(B, T, self.bottleneck_channels, self.mheight, self.mwidth)
        elif "CONVLSTM" in self.model_type:
            encodings = self.convlstm(lstm_inputs)
        # Decode to full self.resolution depth map
        outputs = []
        for t in range(T):
            if self.skip_lstm:
                skip_feats_t = [skip_outputs[i][:, t] for i in range(len(skip_outputs))]
            else:
                skip_feats_t = [timed_features[i][t] for i in range(len(timed_features))]
            x = self.decoder(encodings[:, t], skip_feats_t)                      # [B, C_dec, H, W]
            outputs.append(self.final_conv(x))                                  # [B, 1, H, W]

        outputs = torch.cat(outputs, dim=1)  # [B, T, H, W]
        return outputs, encodings.detach(), seq_events
