import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstohistogram

class BestOfBothWorld(nn.Module):
    def __init__(self, input_dim=4, input_channels = 2, model_type = "BOBWFF", embed_dim=128, depth=6, heads=4, width=346, height=260, num_queries=16):
        super().__init__()
        self.width = width
        self.height = height
        self.num_queries = num_queries
        self.model_type = model_type

        self.method = "add"
        # Embed each event (t, x, y, p)
        add_channels = 1 if "FF" in self.model_type else 0
        self.encoder = Encoder(input_channels+add_channels)
        self.encoder_channels = [32, 24, 32, 64, 1280]
        
        
        self.init_transformer(input_dim, embed_dim, depth, heads)

        if self.model_type == "LSTM":
            self.convlstm = ConvLSTM(
                input_dim=self.encoder_channels[4],
                hidden_dims=[128, 128],
                kernel_size=3,
                num_layers=2
            )
            self.encoder_channels = self.encoder_channels[:-1] + [128]
        else: 
            self.estimated_depth = None


        

        self.decoder = Decoder(self.encoder_channels, self.method)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels[0] + add_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        ) 

        
    def init_transformer(self, input_dim=4, embed_dim=128, depth=6, heads=4):
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
        self.attn_pool_queries = nn.Parameter(torch.randn(1, self.num_queries, embed_dim))
        self.attn_proj = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, batch_first=True)

        self.resolution = 16 ## the lower the more parameters
        self.channels = 16
        # Projection to spatial latent map from pooled tokens
        self.mheight = 9
        self.mwidth = 11
        self.project_to_grid = nn.Sequential(
            nn.Linear(embed_dim * self.num_queries, self.mheight * self.mwidth * self.channels)
        )
        self.encoder_channels[-1] = self.encoder_channels[-1] + self.channels
        
    def reset_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.reset_hidden()
        else:
            self.estimated_depth = None
    def detach_states(self):
        if "LSTM" in self.model_type:
            self.convlstm.detach_hidden()
        else:
            self.estimated_depth = self.estimated_depth.detach()
    def transformer_forward(self, events, mask):
        B, N, _ = events.shape

        # Token embedding + positional encoding
        x = self.embedding(events) + self.pos_embed(events)  # [B, N, D]
        
        # Transformer over tokens
        x = self.transformer(x, src_key_padding_mask=~mask)  # [B, N, D]
        # Attention pooling
        queries = self.attn_pool_queries.expand(B, -1, -1)  # [B, num_queries, D]
        pooled, attn_weights = self.attn_proj(queries, x, x, key_padding_mask=~mask)  # [B, num_queries, D]
        
        # Flatten pooled tokens and project to latent spatial grid
        pooled_flat = pooled.flatten(start_dim=1)  # [B, num_queries * D]
        x_latent = self.project_to_grid(pooled_flat)  # [B, H'*W']
        x_latent = x_latent.view(B, self.channels, self.mheight, self.mwidth)  # [B, 1, H', W']
        return x_latent
    def forward(self, events, mask):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)


        transformer_encoder = self.transformer_forward(events, mask)

        hist_events = eventstohistogram(events)
        if "FF" in self.model_type:
            if self.estimated_depth is None:
                self.estimated_depth = torch.zeros_like(hist_events[:, 0:1, :, :], device=hist_events.device)
            x = torch.cat([hist_events, self.estimated_depth], dim=1)  # [B, in_channels, H, W]
        CNN_encoder, feats = self.encoder(x)
        # Concatenate the outputs from the transformer and CNN
        x = torch.cat([transformer_encoder, CNN_encoder], dim=1)

        if "LSTM" in self.model_type:
            x = self.convlstm(x)
        # Decode to full self.resolution depth map
        depth_imgs = self.decoder(x, feats)
        if "FF" in self.model_type:
            x = torch.cat([depth_imgs, self.estimated_depth], dim=1)
        output = self.final_conv(x)
        self.estimated_depth = output.detach() if "FF" in self.model_type else None
        return output, None

