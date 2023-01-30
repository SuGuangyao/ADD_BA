import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Transformer

 
class CNNTransformerHighway(nn.Module):

    def __init__(self, input_size: int, output_size: int, feature_size: int = 512):
        super(CNNTransformerHighway, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=feature_size, kernel_size=(3, input_size), padding=(1, 0), padding_mode='replicate')
        self.transformer = Transformer()
        # self.decoder = nn.Sequential(nn.Linear(feature_size, int(feature_size / 2)),
        #                              nn.Tanh(),
        #                              nn.Linear(int(feature_size / 2), output_size))
        self.decoder = nn.Linear(feature_size, output_size)
        self.highway = nn.Linear(input_size, output_size)

        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor):
        hw = self.highway(x)

        x = x.unsqueeze(1)

        out = self.conv(x)
        out = out.squeeze(-1).permute(0, 2, 1)
        out = self.transformer(out)
        out = (self.decoder(out) + hw)[:, -1, :]

        return out

    def name(self):
        return self.__class__.__name__
