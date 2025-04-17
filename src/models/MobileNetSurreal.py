import torch
import torch.nn as nn
from models.EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstohistogram
def make_kernel(kernel_size=7):
    center = kernel_size // 2
    first_odds = torch.arange(1, 2 * (center + 1), 2)
    kernel = torch.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            d = torch.maximum(torch.abs(i - center), torch.abs(j - center))
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

    convoluted = (convoluted - torch.mean(convoluted, dim=(1, 2), keepdim=True)) / (
        torch.std(convoluted, dim=(1, 2), keepdim=True) + 1e-10
    )

    return convoluted
class UNetMobileNetSurreal(nn.Module):
    def __init__(self, in_channels, out_channels, use_lstm = False, method = "concatenate"):
        super(UNetMobileNetSurreal, self).__init__()
        assert method in ["concatenate", "add"], "Method must be either 'concatenate' or 'add'"
        
        # Load pretrained MobileNetV2 as encoder backbone
        self.model_type = "FF" if use_lstm is False else "LSTM"
        self.gausian_conv = generate_conv(kernel_size=7, seq_len=self.num_seq)
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
        with torch.no_grad():
            if events.shape[-1] == 4:
                x = eventstohistogram(events)
            x = events
            kernelized = kernel_transform(events, self.gausian_conv)
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
    


