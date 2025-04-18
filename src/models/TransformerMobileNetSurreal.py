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
    def __init__(self, in_channels, out_channels, method = "concatenate"):
        super(UNetMobileNetSurreal, self).__init__()
        assert method in ["concatenate", "add"], "Method must be either 'concatenate' or 'add'"
        
        # Load pretrained MobileNetV2 as encoder backbone
        self.gausian_conv = generate_conv(kernel_size=7, seq_len=2)
        self.method = method
        self.encoder = Encoder(in_channels)
        encoder_channels = [32, 24, 32, 64, 1280]
        
        self.temporal_attention = nn.MultiheadAttention(embed_dim=encoder_channels[4], num_heads=4, batch_first=True)

        self.decoder = Decoder(encoder_channels, method)

        
    def reset_states(self):
        self.history = []
    def detach_states(self):
        self.history = [e.detach() for e in self.history]

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

        if len(self.history) < 10:
            self.history.append(embedding)
        else:
            self.history.pop(0)
            self.history.append(embedding)
        history = torch.stack(self.history, dim=1)
        history = history.permute(0, 2, 1, 3, 4).reshape(history.shape[0], history.shape[1], -1)
        history = history.permute(0, 2, 1)
        x = self.temporal_attention(history, history, history)[0]
        x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[1], history.shape[2], history.shape[3])
        print(x.shape)
        x = self.decoder(x, feats)
        if self.model_type == "FF":
            x = torch.cat([x, self.estimated_depth], dim=1)
        output = self.final_conv(x)
        self.estimated_depth = output.detach() if self.model_type == "FF" else None
        return output, embedding, kernelized
    


