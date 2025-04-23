import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class EventTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, depth=6, heads=4, width=346, height=260, num_queries=16):
        super().__init__()
        self.width = width
        self.height = height
        self.num_queries = num_queries
        self.model_type = "Transformer"

        self.method = "Attention"
        # Embed each event (t, x, y, p)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positional encodings for spatial and temporal info
        self.temporal_pe = self._build_sinusoidal_encoding(embed_dim)

        # Learned spatial positional encoding
        self.spatial_pe = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Learnable attention pooling queries
        self.attn_pool_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.attn_proj = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)

        self.resolution = 8 ## the lower the more parameters
        self.channels = 32
        # Projection to spatial latent map from pooled tokens
        self.project_to_grid = nn.Sequential(
            nn.Linear(embed_dim * num_queries, (self.height // self.resolution) * (self.width // self.resolution) * self.channels)
        )

        # Image decoder
        self.decoder = torch.nn.Sequential()

        n_layers = torch.log2(torch.Tensor([self.resolution])).int().item() - 1   # Number of upsampling layers

        for n in range(n_layers):
            if n != n_layers - 1:
                self.decoder.append(nn.ConvTranspose2d(self.channels //(2 ** n), self.channels // (2 ** (n+1)), kernel_size=4, stride=2, padding=1))
                self.decoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.ConvTranspose2d(self.channels // (2 ** n), 1, kernel_size=4, stride=2, padding=1))
                self.decoder.append(nn.Sigmoid())
    def _build_sinusoidal_encoding(self, dim, max_len=10000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, dim]
        return pe
    def forward(self, events, mask):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        B, N, _ = events.shape

        # Token embedding + positional encoding
        x_base = self.embedding(events)  # [B, N, D]

        # Sinusoidal temporal encoding
        t_norm = (events[:, :, 0] * 9999).long().clamp(0, 9999)  # scale time to [0, 9999]
        pos_time = self.temporal_pe[:, t_norm]  # [1, B, N, D] -> [B, N, D]

        pos_xy = self.spatial_pe(events[:, :, 1:3])  # [B, N, D]
        x = x_base + pos_time + pos_xy  # [B, N, D]
        
        # Transformer over tokens
        x = self.transformer(x, src_key_padding_mask=~mask)  # [B, N, D]
        # Attention pooling
        queries = self.attn_pool_queries.expand(B, -1, -1)  # [B, num_queries, D]
        pooled, attn_weights = self.attn_proj(queries, x, x, key_padding_mask=~mask)  # [B, num_queries, D]
        
        # Flatten pooled tokens and project to latent spatial grid
        pooled_flat = pooled.flatten(start_dim=1)  # [B, num_queries * D]
        x_latent = self.project_to_grid(pooled_flat)  # [B, H'*W']
        x_latent = x_latent.view(B, self.channels, self.height // self.resolution, self.width // self.resolution)  # [B, 1, H', W']

        # Decode to full self.resolution depth map
        depth_imgs = self.decoder(x_latent)  # [B, 1, H, W]
        depth_imgs = F.interpolate(depth_imgs, size=(self.height, self.width), mode='bilinear', align_corners=False)
        return depth_imgs, attn_weights

