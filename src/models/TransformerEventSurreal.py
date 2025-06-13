import torch
import torch.nn as nn
import torch.nn.functional as F

import math
class EventTransformer(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, depth=6, heads=16, num_queries=32, lstm_hidden=256, lstm_layers=1):
        super().__init__()
        self.model_type = "TransLSTM"
        self.embed = nn.Linear(input_dim, embed_dim)
        temporal_pe = self._build_sinusoidal_encoding(embed_dim)
        self.register_buffer("temporal_pe", temporal_pe, persistent=False)
        self.spatial_pe = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.pool_queries = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim, heads, batch_first=True, dropout=0.1)

        self.lstm = nn.LSTM(embed_dim * num_queries, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.projector = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, lstm_hidden)
        )

    def _build_sinusoidal_encoding(self, dim, max_len=10000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, dim]

    def forward(self, event_sequence, mask_sequence):
        # event_sequence: list of length T, each [B, N, 4]
        # mask_sequence: list of length T, each [B, N]
        lstm_inputs = []
        for events, mask in zip(event_sequence, mask_sequence):
            B, N, _ = events.shape
            times = events[:, :, 0]
            times = (times - times.min()) / (times.max() - times.min() + 1e-6)
            ## zeroing all times with probability 0.1
            if torch.rand(1).item() < 0.1:
                times = torch.zeros_like(times)
            events[:, :, 0] = times
            x = self.embed(events)
            pos_t = self.temporal_pe[:, (times * 9999).long().clamp(0, 9999)]
            pos_xy = self.spatial_pe(events[:, :, 1:3])
            x = x + pos_t.squeeze(0) + pos_xy
            x = self.transformer(x, src_key_padding_mask=~mask)

            queries = self.pool_queries.expand(B, -1, -1)
            pooled, _ = self.attn_pool(queries, x, x, key_padding_mask=~mask)
            lstm_inputs.append(pooled.flatten(start_dim=1))

        lstm_inputs = torch.stack(lstm_inputs, dim=1)  # [B, T, D*num_queries]
        lstm_out, _ = self.lstm(lstm_inputs)
        final = self.projector(lstm_out)  # [B, T, D]
        return final, lstm_out # For CPC


