import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn

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

class UNetMobileNetLSTM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetMobileNetLSTM, self).__init__()

        # Load pretrained MobileNetV2 as encoder backbone
        self.backbone = mobilenet_v2(pretrained=True).features

        # Modify first conv layer to accept custom input channels
        self.backbone[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Bottleneck output channels for MobileNetV2
        encoder_channels = [32, 24, 32, 64, 1280]  # example layer outputs

        self.convlstm = ConvLSTM(
                input_dim=encoder_channels[4],
                hidden_dims=[encoder_channels[4], encoder_channels[4]],
                kernel_size=3,
                num_layers=1
            )
        # Decoder upsampling blocks
        self.up3 = self.upsample_block(encoder_channels[4], encoder_channels[3])
        self.up2 = self.upsample_block(encoder_channels[3], encoder_channels[2])
        self.up1 = self.upsample_block(encoder_channels[2], encoder_channels[1])
        self.up0 = self.upsample_block(encoder_channels[1], encoder_channels[0])
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def upsample_block(self, in_channels, skip_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def eventstohistogram(self, events, height=260, width=346):
        B, N, _ = events.shape
        x = (events[:, :, 1] * width).long().clamp(0, width - 1)
        y = (events[:, :, 2] * height).long().clamp(0, height - 1)
        p = events[:, :, 3].long().clamp(0, 1)

        hist = torch.zeros(B, 2, height, width, device=events.device)
        batch_idx = torch.arange(B, device=events.device).unsqueeze(1).expand(-1, N)
        hist.index_put_((batch_idx, p, y, x), torch.abs(events[:, :, 3]), accumulate=True)

        return hist
    def init_state(self, batch_size, channels, height, width):
        self.h = torch.zeros(batch_size, channels, height, width, device=device)
        self.c = torch.zeros(batch_size, channels, height, width, device=device)
    def reset_state(self):
        self.h = None
        self.c = None
    def forward(self, events):

        x = self.eventstohistogram(events)
        
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [0, 2, 4, 7, 18]:
                feats.append(x)
        convlstm_out = self.convlstm(feats[4])  # apply ConvLSTM once
        
        x = convlstm_out 
        x = self.up3(x)
        x = F.interpolate(x, size=feats[3].shape[-2:], mode='bilinear', align_corners=False)
        x = x + feats[3]

        x = self.up2(x)
        x = F.interpolate(x, size=feats[2].shape[-2:], mode='bilinear', align_corners=False)
        x = x + feats[2]

        x = self.up1(x)
        x = F.interpolate(x, size=feats[1].shape[-2:], mode='bilinear', align_corners=False)
        x = x + feats[1]

        x = self.up0(x)
        x = F.interpolate(x, size=feats[0].shape[-2:], mode='bilinear', align_corners=False)
        x = x + feats[0]

        x = self.final_up(x)
        x = F.interpolate(x, size=(260, 346), mode='bilinear', align_corners=False)
        out = self.final_conv(x)
        del convlstm_out
        return out
