"""Graph-context encoder with interchangeable temporal modules."""

import torch
import torch.nn as nn

from src.models.gnn_regressor import CausalTransformerLayer, RpHGNNSpatialEncoder, _sinusoidal_pe


class GraphSequenceRegressor(nn.Module):
    """Spatial encoder over precomputed context groups + a temporal encoder.

    ``temporal`` selects the sequence model applied to the monthly embeddings:
    ``transformer`` (causal self-attention), ``lstm`` or ``gru``. All three are
    causal, so the output at a window position only depends on earlier months.
    """

    def __init__(self, num_groups, group_size, feat_dim, hidden_dim, *, temporal="transformer",
                 num_layers=2, heads=4, dropout=0.1, mlp_layers=2, conv_filters=2, merge_mode="concat"):
        super().__init__()
        self.temporal = temporal
        self.spatial = RpHGNNSpatialEncoder(
            num_groups=num_groups, group_size=group_size, feat_dim=feat_dim, hidden_dim=hidden_dim,
            mlp_layers=mlp_layers, conv_filters=conv_filters, dropout=dropout, merge_mode=merge_mode,
        )
        if temporal == "transformer":
            self.layers = nn.ModuleList(
                [CausalTransformerLayer(hidden_dim, heads, dropout) for _ in range(num_layers)]
            )
        elif temporal in ("lstm", "gru"):
            rnn = nn.LSTM if temporal == "lstm" else nn.GRU
            self.rnn = rnn(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"Unknown temporal module {temporal!r}")
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.spatial(x)
        if self.temporal == "transformer":
            h = h + _sinusoidal_pe(h.size(1), h.size(2), h.device)
            for layer in self.layers:
                h = layer(h)
        else:
            out, _ = self.rnn(h)
            h = self.norm(out + h)
        return self.head(h)
