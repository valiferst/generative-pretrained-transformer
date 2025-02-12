"""
Dummy custom layers used in the Transformer model.
"""

import torch
import torch.nn as nn

class DummyEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(DummyEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)

class DummyFeedForward(nn.Module):
    def __init__(self, embed_dim):
        super(DummyFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))
