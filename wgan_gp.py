import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ks_2samp

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    """ Generator G takes random noise and transforms it into fake data samples.
    Outputs in Tanh, which matches MinMaxScaler(-1, 1) output range 
    Architecture: latent_dim -> 256 -> 512 -> 256 -> output_dim """
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, input_dim),
            nn.Tanh()
        )

    def forward(self, z):
        """ Forward pass of generator. Take batch of latent noise vectors (z),
         and transform them into synthetic feature vectors that resemble real distribution. """
        return self.net(z)


class Critic(nn.Module):
    """ Scores inputs, high for real and low for fake. 
    Architecture: input_dim -> 256 -> 512 -> 256 -> 1 """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1)           # raw score, no activation
        )

    def forward(self, x):
        """ Forward pass of critic. Take batch of input samples and assign each a real-value score.
         Returns score for each sample (batch_size, 1) where a higher score means more "realistic" """
        return self.net(x)
    
