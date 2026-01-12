import torch
import torch.nn as nn

class TransformerCL(nn.Module):
    def __init__(self, vocab_size, dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(dim, 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, 4)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.fc(x)
