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
        self.reset_hidden()
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i-1]
            self.cells.append(
                ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size, bias)
            )

    def forward(self, input_seq):
        """
        input_seq: (batch, seq_len, channels, height, width)
        hidden: tuple of (h, c) for all layers, each is a list of tensors
        Returns:
            output_seq: (batch, seq_len, hidden_dim_last, height, width)
        """
        batch_size, seq_len, _, height, width = input_seq.size()
        if self.h is None:
            self._init_hidden(batch_size, height, width, input_seq.device)


        outputs = []
        for t in range(seq_len):
            x = input_seq[:, t]
            for i, cell in enumerate(self.cells):
                self.h[i], self.c[i] = cell(x, (self.h[i], self.c[i]))
                x = self.h[i]
            outputs.append(self.h[-1].unsqueeze(1))  # collect output from last layer
            # self._init_hidden(batch_size, height, width, input_seq.device)  # reset hidden state for next time step
        output_seq = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim_last, height, width)
        return output_seq

    def _init_hidden(self, batch_size, height, width, device):
        self.h = []
        self.c = []
        for hidden_dim in self.hidden_dims:
            self.h.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
            self.c.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
    def reset_hidden(self):
        self.h = None
        self.c = None
    def detach_hidden(self):
        self.h = [h.detach() for h in self.h]
        self.c = [c.detach() for c in self.c]
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels =None):
        super(Encoder, self).__init__()
        # self.backbone = mobilenet_v2(weights = None).features
        self.backbone = mobilenet_v2(weights = None).features
        # Modify first conv layer to accept custom input channels
        self.backbone[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # base = mobilenet_v2(weights = None).features
        # # Modify first conv layer to accept custom input channels
        # old_conv = base[0][0]
        # new_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # with torch.no_grad():
        #     w = old_conv.weight  # [32, 3, 3, 3]
        #     if in_channels == 3:
        #         new_conv.weight.copy_(w)
        #     elif in_channels > 3:
        #         # copy RGB then fill extra channels with mean over RGB
        #         new_conv.weight[:, :3].copy_(w)
        #         mean_w = w.mean(dim=1, keepdim=True)
        #         new_conv.weight[:, 3:].copy_(mean_w.expand(-1, in_channels - 3, -1, -1))
        #     else:
        #         # collapse RGB to fewer channels by averaging
        #         mean_w = w.mean(dim=1, keepdim=True)  # [32,1,3,3]
        #         new_conv.weight.copy_(mean_w.expand(-1, in_channels, -1, -1))
        # base[0][0] = new_conv
        # self.backbone = base

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

        # Apply the bottleneck layer

        for i, layer in reversed(list(enumerate(self.decoder_layers))):
            x = layer(x)
            if i != 0:
                x = F.interpolate(x, size=feats[i-1].shape[-2:], mode='bilinear', align_corners=False)
                if self.method == "concatenate":
                    x = torch.cat([x, feats[i]], dim=1)
                else:
                    x = x + feats[i-1]
        
        return x