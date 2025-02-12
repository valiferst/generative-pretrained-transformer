"""
Dummy implementation of a Transformer model.
"""

import torch
import torch.nn as nn
from model.layers import DummyEmbedding, DummyFeedForward

class DummyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super(DummyTransformer, self).__init__()
        self.embedding = DummyEmbedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [DummyFeedForward(embed_dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        # Dummy forward pass
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    # Dummy test run
    vocab_size = 10000
    embed_dim = 256
    num_layers = 4
    model = DummyTransformer(vocab_size, embed_dim, num_layers)
    dummy_input = torch.randint(0, vocab_size, (1, 10))  # Batch size 1, sequence length 10
    output = model(dummy_input)
    print("Dummy Transformer output shape:", output.shape)
