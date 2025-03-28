import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Event Transformer Encoder with Batching ----------
class EventTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, depth=4, heads=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        # encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=embed_dim * 2, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, events, mask):
        # events: [B, N, 4], mask: [B, N] where 1 = valid, 0 = pad
        x = self.embedding(events)  # [B, N, D]
        # x = self.encoder(x, src_key_padding_mask=~mask.bool())  # Transformer expects padding mask as True where masked
        return x  # [B, N, D]

# ---------- Batched Splatting to Spatial Grid ----------
def splat_to_grid_batched(events, features, mask, height, width):
    # events: [B, N, 4], features: [B, N, D], mask: [B, N]
    B, N, D = features.shape
    grid = torch.zeros(B, D, height, width, device=features.device)
    counts = torch.zeros(B, 1, height, width, device=features.device)

    x = (events[:, :, 1] * width).long().clamp(0, width - 1)
    y = (events[:, :, 2] * height).long().clamp(0, height - 1)

    for b in range(B):
        for i in range(N):
            if mask[b, i]:
                xi, yi = x[b, i], y[b, i]
                grid[b, :, yi, xi] += features[b, i]
                counts[b, :, yi, xi] += 1

    grid = grid / (counts + 1e-6)
    return grid  # [B, D, H, W]

# ---------- ConvLSTM Cell ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.hidden_dim = hidden_dim

    def forward(self, x, prev):
        h_prev, c_prev = prev
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

# ---------- Full Model ----------
class EventToDepthModel(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=64, height=128, width=128):
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.transformer = EventTransformer(embed_dim=embed_dim)
        self.conv_lstm = ConvLSTMCell(embed_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1)  # Output depth map
        )

    def forward(self, batched_events, mask):
        # batched_event_chunks: List of [B, N_t, 4] for T timesteps
        # batched_masks: List of [B, N_t] masks for each chunk
        B = batched_events.shape[0]
        h, c = self.init_hidden(B, batched_events.device)
        
        feats = self.transformer(batched_events, mask)  # [B, N, D]
        grid = splat_to_grid_batched(batched_events, feats, mask, self.height, self.width)  # [B, D, H, W]
        
        h, c = self.conv_lstm(grid, (h, c))
        depth = self.decoder(h)  # [B, 1, H, W]
        depth = torch.sigmoid(depth)
        return depth.squeeze(1)  # List of [B, 1, H, W] over time

    def init_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, self.height, self.width, device=device)
        return h, c
