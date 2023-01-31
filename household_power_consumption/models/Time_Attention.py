import torch
import torch.nn as nn
import torch.nn as nn
import torch
import torch.nn.functional as F



class Time_Attention(nn.Module):
    def __init__(self, input_size, out_size, device="cuda"):
        super(Time_Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.W = torch.randn((input_size, out_size), device=device) * 0.01
        self.b = torch.randn((1, out_size), device=device) * 0.01
        self.H = torch.randn((1, out_size), device=device) * 0.01

    def forward(self, x):
        S = self.softmax(self.tanh(x) @ self.W + self.b)
        out = self.H * S
        return out

class BLA (nn.Module):

    def __init__(self,
                 input_size: int = 15,
                 output_size: int = 15,
                 hidden_dim: int = 256,
                 lstm_dropout: float = 0,
                 linear_dropout=0.1):
        super(BLA, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.lstm_dropout = lstm_dropout
        self.linear_dropout = linear_dropout

        self.blstm = nn.LSTM(self.input_size,
                             self.hidden_dim,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True,
                             dropout=self.lstm_dropout)

        self.lstm = nn.LSTM(512,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=False,
                            batch_first=True,
                            dropout=self.lstm_dropout)
        self.time_attention = Time_Attention(self.hidden_dim,self.hidden_dim)
        self.dropout = nn.Dropout(self.linear_dropout)
        self.linear = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.blstm(x)
        #
        out , (h_n, c_n)= self.lstm(out)
        out = self.time_attention(out)
        out = torch.sum(out, dim=1)
        out = self.dropout(out)
        out = self.linear(out)

        return out

    def name(self):
        return self.__class__.__name__

