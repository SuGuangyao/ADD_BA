import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer


class LinearTransformer(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(LinearTransformer, self).__init__()

        self.transformer = Transformer()
        self.encoder = nn.Linear(input_size, 512)
        self.decoder = nn.Linear(512, output_size)

        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        out = self.encoder(x)
        out = self.transformer(out)
        out = self.decoder(out)
        out = out[:, -1, :]
        return out

    def name(self):
        return self.__class__.__name__
