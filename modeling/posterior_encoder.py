## Reference: https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py#L212

import torch.nn as nn
from .. import commons
from .. import modules
import torch

'''
For the posterior encoder, we use the non-causal WaveNet residual blocks used in WaveGlow (Prenger et al., 2019)
and Glow-TTS (Kim et al., 2020). A WaveNet residual block consists of layers of dilated convolutions with a gated
activation unit and skip connection. The linear projection layer above the blocks produces the mean and variance of
the normal posterior distribution. For the multi-speaker case, we use global conditioning (Oord et al., 2016) in residual
blocks to add speaker embedding
'''


class PosteriorEncoder(nn.Module):

    def __init__(self, 
                in_channels,
                out_channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers, 
                gin_channels = 0               
                ):


        super(PosteriorEncoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.proj = nn.Conv1d(hidden_channels, out_channels*2, kernel_size = 1)

        ## seperate file for modules needed
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels = gin_channels)



    def forward(self, x, x_lengths, g = None):

        ## seperate file for commons needed
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x)*x_mask
        x = self.enc(x, x_mask, g = g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim = 1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
