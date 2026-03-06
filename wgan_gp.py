""" Generative Adverserial Network using Wasserstein distance and gradient penalty
implemented according to research by Gulrajani et al. https://arxiv.org/abs/1704.00028 """
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
         Returns score for each sample (batch_size, 1) where a higher score means more realistic """
        return self.net(x)


def gradient_penalty(critic: nn.Module,
                     real: torch.Tensor,
                     fake: torch.Tensor,
                     device: torch.device,
                     lambda_gp: float = 10.0) -> torch.Tensor:
    """ For computing the gradient penalty. Enforces the Lipschitz constraint
     by penalising the critic if the gradient norm deviates from 1 """
    batch_size = real.size(0)

    # random interpolation weight per sample
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)

    # interpolation between real and fake
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    # give score at interpolated point
    critic_interp = critic(interpolated)

    # gradient score at this input
    gradients = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # penalize deviations from 1
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty

class WGANGP:
    """ WGAN trainer repurposed for tabular data.
     Input in shape of scaled numpy arrays in [-1, 1] 
     Returns generated numpy array within the same space """

    # parameters mostly according to existing research/github repos. may be subject to change

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 128,
                 lr: float = 1e-4,
                 n_critic: int = 5,
                 lambda_gp: float = 10.0):

        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.n_critic   = n_critic
        self.lambda_gp  = lambda_gp
        self.device     = DEVICE

        # create neural networks, move parameters to the GPU

        self.generator = Generator(input_dim, latent_dim).to(self.device)
        self.critic = Critic(input_dim).to(self.device)

        # optimizers: update network weights after backpropagation computes gradients.
        # "how much should I change and in which direction"

        self.opt_generator = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(0.0, 0.9))
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=(0.0, 0.9))

        # loss history, two lists to collect average loss per epoch during training
        # plot these to see whether we have convergence
        # ideal losses:
        # critic loss: starts large, decreases and stabilises around a small negative value
        # generator loss: starts large, decreases and stabilises - oscillation is normal

        self.g_losses: list[float] = []
        self.c_losses: list[float] = []

    def train(self,
            X: np.ndarray,
            epochs: int = 2000,
            batch_size: int = 256,
            log_interval: int = 200) -> None:
        pass
