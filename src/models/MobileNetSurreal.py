import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNetMobileNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetMobileNet, self).__init__()

        # Load pretrained MobileNetV2 as encoder backbone
        self.backbone = mobilenet_v2(pretrained=True).features

        # Modify first conv layer to accept custom input channels
        self.backbone[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Bottleneck output channels for MobileNetV2
        encoder_channels = [32, 24, 32, 64, 1280]  # example layer outputs

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
            nn.Conv2d(32 +1, 32, kernel_size=3, padding=1),
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

    def forward(self, events, est_depth):
        x = self.eventstohistogram(events)
        x = torch.cat([x, est_depth], dim=1)  # [B, in_channels, H, W]

        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [0, 2, 4, 7, 18]:
                feats.append(x)

        x = self.up3(feats[4])
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
        x = torch.cat([x, est_depth], dim=1)
        out = self.final_conv(x)
        return out
