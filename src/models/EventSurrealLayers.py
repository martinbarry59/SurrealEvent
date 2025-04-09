import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 
            hidden_dim * 4, 
            kernel_size, 
            padding=padding,
            bias=bias
        )

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, bias=True):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.cells.append(
                ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size, bias)
            )
        self.reset_hidden()

    def forward(self, input_seq):
        """
        input_seq: (batch, seq_len, channels, height, width)
        """
        batch_size, _, height, width = input_seq.size()
        if self.h is None or self.c is None:
            self._init_hidden(batch_size, height, width, input_seq.device)

        x = input_seq
        for i, cell in enumerate(self.cells):
            self.h[i], self.c[i] = cell(x, (self.h[i], self.c[i]))
            x = self.h[i]
        

        return self.h[-1]  # return the last layer's output
    def reset_hidden(self):
        self.h = None
        self.c = None
    def _init_hidden(self, batch_size, height, width, device):
        h = []
        c = []
        for hidden_dim in self.hidden_dims:
            h.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
            c.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
        self.h = h
        self.c = c
    def detach_hidden(self):
        for i in range(self.num_layers):
            self.h[i].detach_()
            self.c[i].detach_()
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels =None):
        super(Encoder, self).__init__()
        self.backbone = mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1).features

        # Modify first conv layer to accept custom input channels
        self.backbone[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [0, 2, 4, 7, 18]:
                feats.append(x)
        return feats[-1], feats
    
class Decoder(nn.Module):
    def __init__(self, encoder_channels, method, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.method = method
        c = 2 if self.method == "concatenate" else 1
        encoder_channels[-1] = int(encoder_channels[-1] / c)
        self.decoder_layers = []
        for i in range(len(encoder_channels)-1):
            if i == 0:
                self.decoder_layers.append(self.upsample_block( c * encoder_channels[i], 32))
            self.decoder_layers.append(self.upsample_block(c * encoder_channels[i+1], encoder_channels[i]))
        self.decoder_layers = nn.ModuleList(self.decoder_layers)    
        
        
    def upsample_block(self, in_channels, skip_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, feats):
        for i, layer in reversed(list(enumerate(self.decoder_layers))):
            x = layer(x)
            if i != 0:
                x = F.interpolate(x, size=feats[i-1].shape[-2:], mode='bilinear', align_corners=False)
                if self.method == "concatenate":
                    x = torch.cat([x, feats[i-1]], dim=1)
                else:
                    x = x + feats[i-1]
        
        return x