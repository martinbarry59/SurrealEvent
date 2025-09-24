import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from typing import List, Tuple

__all__ = [
    "ConvLSTMCell",
    "ConvLSTM",
    "Encoder",
    "Decoder",
]


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell.

    Implements a convolutional LSTM recurrence preserving spatial structure.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim * 4,
            kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
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
    """Multi-layer ConvLSTM wrapper maintaining hidden state across calls.

    Call ``reset_hidden()`` between independent sequences. Use ``detach_hidden()``
    to truncate BPTT while preserving numerical values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: int,
        num_layers: int,
        bias: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            self.cells.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size, bias))
        self.reset_hidden()

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Forward sequence.

        Args:
            input_seq: (batch, seq_len, channels, height, width)
        Returns:
            output_seq: (batch, seq_len, hidden_last, height, width)
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
            outputs.append(self.h[-1].unsqueeze(1))
        output_seq = torch.cat(outputs, dim=1)
        return output_seq

    def _init_hidden(self, batch_size: int, height: int, width: int, device: torch.device):
        self.h = []
        self.c = []
        for hidden_dim in self.hidden_dims:
            self.h.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))
            self.c.append(torch.zeros(batch_size, hidden_dim, height, width, device=device))

    def reset_hidden(self):
        self.h = None
        self.c = None

    def detach_hidden(self):
        if self.h is not None:
            self.h = [h.detach() for h in self.h]
        if self.c is not None:
            self.c = [c.detach() for c in self.c]


class Encoder(nn.Module):
    """MobileNetV2 backbone extractor with adaptable first layer.

    Collects intermediate feature maps at predefined layer indices to support
    UNet-like skip connections during decoding.
    """

    def __init__(
        self, in_channels: int, out_channels: int | None = None
    ):  # out_channels kept for backward compat
        super().__init__()
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
        old_conv = base[0][0]
        new_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            w = old_conv.weight  # [32, 3, 3, 3]
            if in_channels == 3:
                new_conv.weight.copy_(w)
            elif in_channels > 3:
                new_conv.weight[:, :3].copy_(w)
                mean_w = w.mean(dim=1, keepdim=True)
                new_conv.weight[:, 3:].copy_(mean_w.expand(-1, in_channels - 3, -1, -1))
            else:
                mean_w = w.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(mean_w.expand(-1, in_channels, -1, -1))
        base[0][0] = new_conv
        self.backbone = base

    def forward(self, x: torch.Tensor):
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [0, 2, 4, 7, 18]:  # chosen layers for multi-scale decoding
                feats.append(x)
        return feats[-1], feats


class Decoder(nn.Module):
    """Lightweight upsampling decoder reconstructing spatial resolution.

    Supports either addition or concatenation for skip fusion ("method").
    """

    def __init__(self, encoder_channels: List[int], method: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        c = 2 if self.method == "concatenate" else 1
        encoder_channels[-1] = int(encoder_channels[-1] / c)
        self.decoder_layers = []
        for i in range(len(encoder_channels) - 1):
            if i == 0:
                self.decoder_layers.append(self.upsample_block(c * encoder_channels[i], 32))
            self.decoder_layers.append(
                self.upsample_block(c * encoder_channels[i + 1], encoder_channels[i])
            )
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

    def upsample_block(self, in_channels: int, skip_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def _match_size(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Align spatial size of x to ref via symmetric pad/crop (no resampling)."""
        H, W = ref.shape[-2:]
        h, w = x.shape[-2:]
        # Pad if smaller
        pad_h = max(0, H - h)
        pad_w = max(0, W - w)
        if pad_h or pad_w:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            x = F.pad(x, (left, right, top, bottom))
            h, w = x.shape[-2:]
        # Center-crop if larger
        if h > H or w > W:
            ys = (h - H) // 2
            xs = (w - W) // 2
            x = x[:, :, ys:ys + H, xs:xs + W]
        return x

    def forward(self, x: torch.Tensor, feats: List[torch.Tensor]):
        for i, layer in reversed(list(enumerate(self.decoder_layers))):
            x = layer(x)
            if i != 0:
                # Align spatial dims without resampling to preserve features
                target = feats[i - 1]
                if x.shape[-2:] != target.shape[-2:]:
                    x = self._match_size(x, target)
                # Consistently fuse with the matching skip level
                if self.method == "concatenate":
                    x = torch.cat([x, target], dim=1)
                else:
                    x = x + target
        return x
