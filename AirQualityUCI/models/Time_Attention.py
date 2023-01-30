import torch
import torch.nn as nn


# class Score(nn.Module):
#     def __init__(self, input_size, out_size,device):
#         super(Score, self).__init__()
#         self.tanh = nn.Tanh()
#         self.softmax = nn.Softmax()
#         self.W = torch.randn((input_size,out_size),device=device)*0.01
#         self.b = torch.randn((input_size),device=device)*0.01
#     def forward(self, Yf):
#         S = self.softmax(self.W @ self.tanh(Yf)+self.b)
#         return S

class Time_Attention(nn.Module):
    def __init__(self, input_size, out_size, device):
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

if __name__ == "__main__":
    input_size = 512
    out_size = 512
    yf = torch.arange(512).reshape(1, 512)
    print(yf)
    score =Time_Attention(input_size=input_size, out_size=out_size, device="cpu")
    out = score(yf)
    print(out)