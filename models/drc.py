import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


# From https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/geister.py
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4*self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def init_hidden(self, input_size, batch_size):
        return tuple([
            torch.zeros(*batch_size, self.hidden_dim, *input_size).to(device),
            torch.zeros(*batch_size, self.hidden_dim, *input_size).to(device)
        ])

    def forward(self, input_tensor, state):
        h, c = state

        # ... x C x H x W
        comb = torch.cat((input_tensor, h), dim=-3)
        conv_output = self.conv(comb)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=-3)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class DRC(nn.Module):
    def __init__(self, D, N, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(DRC, self).__init__()

        self.D = D
        self.N = N

        self.blocks = nn.ModuleList([ConvLSTM(input_dim, hidden_dim, (kernel_size, kernel_size), bias) for i in range(self.D)])

    def init_hidden(self, input_size, batch_size):
        hiddens = [b.init_hidden(input_size, batch_size) for b in self.blocks]
        return hiddens

    def forward(self, x, hidden):
        if hidden is None:
            hidden = self.init_hidden(x.shape[-2:], x.shape[:-3])

        for n in range(self.N):
            for i, block in enumerate(self.blocks):
                hidden[i] = block(hidden[i-1][0] if i > 0 else x, hidden[i])

        return hidden[-1][0], hidden
