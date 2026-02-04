"""Shared autoencoder model definition for training and inference."""

import torch.nn as nn


class Autoencoder(nn.Module):
    """Simple feedforward autoencoder for tabular features."""

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 8,
        hidden_dims: list | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        # Encoder
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, encoding_dim))
        self.encoder = nn.Sequential(*layers)
        # Decoder
        dec_layers = []
        prev = encoding_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
