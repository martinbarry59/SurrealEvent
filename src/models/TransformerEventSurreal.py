import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
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
        self.channels = 128
        # Projection to spatial latent map from pooled tokens
        self.project_to_grid = nn.Sequential(
            nn.Linear(embed_dim * num_queries, (self.height // self.resolution) * (self.width // self.resolution) * self.channels)
        )

        # Image decoder
        self.decoder = torch.nn.Sequential()
        n_layers = torch.log2(torch.Tensor([self.resolution])).int().item() - 1   # Number of upsampling layers

        for n in range(n_layers):
            print(n, self.channels //(2 ** n), self.channels // (2 ** (n+1)))
            if n != n_layers - 1:
                self.decoder.append(nn.ConvTranspose2d(self.channels //(2 ** n), self.channels // (2 ** (n+1)), kernel_size=4, stride=2, padding=1))
                self.decoder.append(nn.ReLU())
            else:
                self.decoder.append(nn.ConvTranspose2d(self.channels // (2 ** n), 1, kernel_size=4, stride=2, padding=1))
                self.decoder.append(nn.Sigmoid())

    def forward(self, events, mask):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)
        B, N, _ = events.shape

        # Token embedding + positional encoding
        x = self.embedding(events) + self.pos_embed(events)  # [B, N, D]
        
        # Transformer over tokens
        x = self.transformer(x, src_key_padding_mask=~mask)  # [B, N, D]
        # Attention pooling
        queries = self.attn_pool_queries.expand(B, -1, -1)  # [B, num_queries, D]
        pooled, _ = self.attn_proj(queries, x, x, key_padding_mask=~mask)  # [B, num_queries, D]
        
        # Flatten pooled tokens and project to latent spatial grid
        pooled_flat = pooled.flatten(start_dim=1)  # [B, num_queries * D]
        x_latent = self.project_to_grid(pooled_flat)  # [B, H'*W']
        x_latent = x_latent.view(B, self.channels, self.height // self.resolution, self.width // self.resolution)  # [B, 1, H', W']

        # Decode to full self.resolution depth map
        depth_imgs = self.decoder(x_latent)  # [B, 1, H, W]
        depth_imgs = F.interpolate(depth_imgs, size=(self.height, self.width), mode='bilinear', align_corners=False)
        # depth_imgs = torch.nn.functional.sigmoid(depth_imgs)  # Normalize to [0, 1]
        return depth_imgs
