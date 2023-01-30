import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import PositionalEncoding


class MaskedTransformer(nn.Module):
    
    def __init__(self, feature_size=512, num_layers=1, dropout=0.1):
        super(MaskedTransformer, self).__init__()
                       
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=32, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x, do_train: bool):
        if self.src_mask is None or self.src_mask.size(0) != x.shape[1]:
            mask = self.__generate_random_mask(x.shape[1]).to('cuda')
            self.src_mask = mask

        out = self.pos_encoder(x)
        
        if do_train:
            out = self.transformer_encoder(out, self.src_mask)
        else:
            out = self.transformer_encoder(out)

        return out

    def __generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask

    def __generate_random_mask(self, size, prob: float = 0.2):
        mask = torch.bernoulli(torch.zeros(size, size), prob) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask

    def name(self):
        return self.__class__.__name__


class MaskedLinearTransformer(nn.Module):

    def __init__(self, input_size: int, output_size: int):
        super(MaskedLinearTransformer, self).__init__()

        self.transformer = MaskedTransformer()
        self.encoder = nn.Linear(input_size, 512)
        self.decoder = nn.Linear(512, output_size)

        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, do_train: bool):
        out = self.encoder(x)
        out = self.transformer(out, do_train)
        out = self.decoder(out)

        return out

    def name(self):
        return self.__class__.__name__
