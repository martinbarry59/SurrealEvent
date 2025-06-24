import torch
import torch.nn as nn
import torch.nn.functional as F
from  .EventSurrealLayers import Encoder, Decoder, ConvLSTM
from utils.functions import eventstohistogram
import math
class BestOfBothWorld(nn.Module):
    def __init__(self, input_dim=4, input_channels = 2, model_type = "BOBWFF", embed_dim=256, depth=12, heads=8, width=346, height=260, num_queries=64):
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

        if "LSTM" in self.model_type:
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
        temporal_pe = self._build_sinusoidal_encoding(embed_dim)
        ## register buffer to avoid reallocation
        self.register_buffer("temporal_pe", temporal_pe, persistent=False)

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
    def _build_sinusoidal_encoding(self, dim, max_len=10000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, dim]
        return pe
    def transformer_forward(self, events, mask):
        B, N, _ = events.shape

        # Token embedding + positional encoding
        # Token embedding + positional encoding
        x_base = self.embedding(events)  # [B, N, D]

        # Sinusoidal temporal encoding
        t_norm = (events[:,:, 0] * 9999).long().clamp(0, 9999)  # scale time to [0, 9999]
        pos_time = self.temporal_pe[:, t_norm].squeeze(0)  # [1, B, N, D] -> [B, N, D]

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
        x_latent = x_latent.view(B, self.channels, self.mheight, self.mwidth)  # [B, 1, H', W']
        return x_latent
    def forward(self, event_sequence, mask_sequence):
        # events: [B, N, 4], mask: [B, N] (True = valid, False = padding)

        lstm_inputs = []
        timed_features = []
        for events, mask in zip(event_sequence, mask_sequence):
            transformer_encoder = self.transformer_forward(events, mask)
            hist_events = eventstohistogram(events, self.height, self.width)
            CNN_encoder, feats = self.encoder(hist_events)
            timed_features.append(feats)
        # Concatenate the outputs from the transformer and CNN
            interpoted = F.interpolate(CNN_encoder, size=(self.mheight, self.mwidth), mode='bilinear', align_corners=False)
            x = torch.cat([transformer_encoder, interpoted], dim=1)
            lstm_inputs.append(x)
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        encodings = self.convlstm(lstm_inputs)
        # Decode to full self.resolution depth map
        outputs = []
        
        for t in range(encodings.shape[1]):
            x = self.decoder(encodings[:,t], timed_features[t])
        
            outputs.append(self.final_conv(x))
        outputs = torch.cat(outputs, dim=1)

        return outputs, encodings.detach()
