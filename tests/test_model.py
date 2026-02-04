"""Tests for autoencoder model."""

import tempfile
from pathlib import Path

import pytest
import torch

import model as model_module


def test_forward_shape():
    """Forward pass preserves batch size and output dimension."""
    m = model_module.Autoencoder(input_dim=10, encoding_dim=4, hidden_dims=[8, 6])
    x = torch.randn(5, 10)
    out = m(x)
    assert out.shape == (5, 10)


def test_forward_deterministic():
    """Same input and eval mode give same output."""
    m = model_module.Autoencoder(input_dim=6, encoding_dim=2)
    m.eval()
    x = torch.randn(3, 6)
    with torch.no_grad():
        out1 = m(x)
        out2 = m(x)
    torch.testing.assert_close(out1, out2)


def test_save_load_roundtrip():
    """State dict save and load preserves model output."""
    m1 = model_module.Autoencoder(input_dim=12, encoding_dim=4)
    m1.eval()
    x = torch.randn(2, 12)
    with torch.no_grad():
        out1 = m1(x)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.pt"
        torch.save(m1.state_dict(), path)
        m2 = model_module.Autoencoder(input_dim=12, encoding_dim=4)
        m2.load_state_dict(torch.load(path, map_location="cpu"))
        m2.eval()
        with torch.no_grad():
            out2 = m2(x)
    torch.testing.assert_close(out1, out2)


def test_encoding_dim():
    """Encoder output has encoding_dim size."""
    m = model_module.Autoencoder(input_dim=8, encoding_dim=3)
    x = torch.randn(4, 8)
    z = m.encoder(x)
    assert z.shape == (4, 3)
