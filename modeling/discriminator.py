import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import sys
sys.path.append("..")

import modules
from torch.nn import functional as F
import torch
import commons
from commons import get_padding
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d

# DiscriminatorP
# reference: https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py#L299

class DiscriminatorP(nn.Module):
    def __init__(self,
                 period,
                 kernel_size=5,
                 stride=3,
                 use_spectral_norm=False):

        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding = (get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding = (get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding = (get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding = (get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), (stride, 1), padding = (get_padding(kernel_size, 1), 0))),

        ])
        self.conv_post = norm_f(Conv2d(1024,1, (3,1),1, padding = (1,0)))

    def forward(self, x):

        fmap = []

        ## 1D to 2D
        batch, channel, time_t = x.shape

        if time_t%self.period != 0: ## pad first
            n_pad = self.period - (time_t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            time_t = time_t + n_pad

        ## split into multiple segments
        x = x.view(batch, channel, time_t//self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# DiscriminatorS
# reference: https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py#L336

class DiscriminatorS(nn.Module):
    def __init__(self,
                 use_spectral_norm=False):

        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding = 7)),
            norm_f(Conv1d(16, 64, 41, 4, groups = 4, padding = 20)),
            norm_f(Conv1d(64, 256, 41, 4, groups =16, padding = 20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups = 64, padding = 20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups = 256, padding = 20)),
            norm_f(Conv1d(1024, 1024, 5, 1,padding = 2)),
        ])

        self.conv_post = norm_f(Conv2d(1024,1, (3,1),1, padding = 1)) 

    def forward(self, x):

        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


## Multi Period Discriminator
## reference: https://github.com/jaywalnut310/vits/blob/2e561ba58618d021b5b8323d3765880f7e0ecfdb/models.py#L364

class MultiPeriodDiscriminator(nn.Module):

    def __init__(self, use_spectral_norm = False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm = use_spectral_norm)]
        discs = discs + [DiscriminatorP(period, use_spectral_norm = use_spectral_norm) for period in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



