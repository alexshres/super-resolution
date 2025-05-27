# Alex Shrestha
# FILE: denoiser.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Simple denoising class with u-net structure
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, 3, padding=1)


    def forward(self, x, t_embed):
        x = torch.cat([x, t_embed], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02

    return torch.linspace(beta_start, beta_end, timesteps)

