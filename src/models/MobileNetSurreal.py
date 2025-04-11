import torch
import torch.nn as nn
from models.EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstohistogram

class UNetMobileNetSurreal(nn.Module):
    def __init__(self, in_channels, out_channels, use_lstm = False, method = "concatenate"):
        super(UNetMobileNetSurreal, self).__init__()
        assert method in ["concatenate", "add"], "Method must be either 'concatenate' or 'add'"
        
        # Load pretrained MobileNetV2 as encoder backbone
        self.model_type = "FF" if use_lstm is False else "LSTM"
        self.method = method
        self.encoder = Encoder(in_channels)
        encoder_channels = [32, 24, 32, 64, 1280]
        
        if self.model_type == "LSTM":
            self.convlstm = ConvLSTM(
                input_dim=encoder_channels[4],
                hidden_dims=[encoder_channels[4], encoder_channels[4]],
                kernel_size=3,
                num_layers=2
            )
        else: 
            self.estimated_depth = None
        self.decoder = Decoder(encoder_channels, method)

        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0] + 1 * (not use_lstm), 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    def reset_states(self):
        if self.model_type == "LSTM":
            self.convlstm.reset_hidden()
        else:
            self.estimated_depth = None
    def detach_states(self):
        if self.model_type == "LSTM":
            self.convlstm.detach_hidden()
        else:
            self.estimated_depth = self.estimated_depth.detach()

    def forward(self, events):
        x = eventstohistogram(events)
        if self.model_type == "FF":
            if self.estimated_depth is None:
                self.estimated_depth = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            x = torch.cat([x, self.estimated_depth], dim=1)  # [B, in_channels, H, W]

        embedding, feats = self.encoder(x)

        if self.model_type == "LSTM":
            x = self.convlstm(embedding)
        else:
            x = embedding
        x = self.decoder(x, feats)
        if self.model_type == "FF":
            x = torch.cat([x, self.estimated_depth], dim=1)
        output = self.final_conv(x)
        self.estimated_depth = output if self.model_type == "FF" else None
        return output, embedding
    


