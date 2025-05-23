import torch
import torch.nn as nn
from models.EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstohistogram
import numpy as np
def make_kernel(kernel_size=7):
    center = kernel_size // 2
    first_odds = torch.arange(1, 2 * (center + 1), 2)
    kernel = torch.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            d = np.maximum(np.abs(i - center), np.abs(j - center))
            kernel[i, j] = 1 / first_odds[d] ** 2
    
    return kernel
def generate_conv(kernel_size=7, seq_len=8):
    kernel = make_kernel(kernel_size)
    conv = torch.nn.Conv2d(
        seq_len,
        seq_len,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=False,
    )
    index = torch.arange(0, seq_len)
    conv.weight.data[:, :, :, :] = 0
    conv.weight.data[index, index] = kernel
    return conv
def kernel_transform(data, conv):
    convoluted = conv(data)
    ## normalise the convoluted data
    convoluted = convoluted - convoluted.min() / 0.1
    return convoluted
class UNetMobileNetSurreal(nn.Module):
    def __init__(self, in_channels, out_channels, net_type, method = "concatenate"):
        super(UNetMobileNetSurreal, self).__init__()
        assert method in ["concatenate", "add"], "Method must be either 'concatenate' or 'add'"
        add_channels = 1 if net_type == "FF" else 0

        # Load pretrained MobileNetV2 as encoder backbone
        self.model_type = net_type
        self.gausian_conv = generate_conv(kernel_size=7, seq_len=2)
        self.method = method
        self.encoder = Encoder(in_channels+add_channels)
        encoder_channels = [32, 24, 32, 64, 1280]
        
        if self.model_type == "LSTM":
            self.convlstm = ConvLSTM(
                input_dim=encoder_channels[4],
                hidden_dims=[128, 128],
                kernel_size=3,
                num_layers=2
            )
        else: 
            self.estimated_depth = None
        encoder_channels = encoder_channels[:-1] + [128]
        self.decoder = Decoder(encoder_channels, method)

        self.final_conv = nn.Sequential(
            nn.Conv2d(encoder_channels[0] + add_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
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
        with torch.no_grad():
            if events.shape[-1] == 4:
                events = eventstohistogram(events)
            
            kernelized = kernel_transform(events, self.gausian_conv)
            x = kernelized.detach()
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
        self.estimated_depth = output.detach() if self.model_type == "FF" else None
        return output, embedding, kernelized
    


