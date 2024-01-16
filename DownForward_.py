import torch
from torch import nn as nn
import torch.nn.functional as F
import numpy as np
import time
from Down_ import FouriDown, CutDown, BlurDown, unPixelShuffle


class DownSample(nn.Module):
    def __init__(self, in_channels, base_channel, downsampling):
        super(DownSample, self).__init__()

        if downsampling == 'Bilinear':
            self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                     nn.Conv2d(in_channels, base_channel, 1, 1, 0, bias=False))

        elif downsampling == 'Bicubic':
            self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                            nn.Conv2d(in_channels, base_channel, 1, 1, 0, bias=False))

        elif downsampling == 'StrideConv22':
            self.down = nn.Conv2d(in_channels, base_channel, 2, 2, 0, bias=False)
        
        elif downsampling == 'MaxPooling':
            self.down = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(in_channels, base_channel, 1, 1, 0, bias=False))

        elif downsampling == 'AverPooling':
            self.down = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2),
                            nn.Conv2d(in_channels, base_channel, 1, 1, 0, bias=False))

        elif downsampling == 'PixelShuffle':
            self.down = nn.Sequential(unPixelShuffle(downscale_factor=2),
                                      nn.Conv2d(in_channels*4, base_channel, 1, 1, 0, bias=False))

        elif downsampling == 'FouriDown':
                self.down = FouriDown(in_channels, base_channel)

        elif downsampling == 'CutDown':
                self.down = CutDown(in_channels, base_channel)

        elif downsampling == 'BlurDown':
                self.down = BlurDown(in_channels, base_channel)        

    def forward(self, x):

        x = self.down(x)
        return x
