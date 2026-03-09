# pylint: disable=line-too-long
# pylint: disable=invalid-name
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
            generator_iters: int,
            batch_size: int = 256,
            log_interval: int = 200,
            checkpoint_dir: str = None,
            resume_from: str = None) -> None:
        """ Training WGAN-GP on a single-label scaled numpy array at a time """

        # data loader
        tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)
        loader = DataLoader(
            TensorDataset(tensor_X),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        # infinite iterator to go to the next batch, restarting when no more batches
        def infinite_loader():
            """ Infinite iterator to go to the next batch, restarting when no more batches """
            while True:
                yield from loader

        data_iter = infinite_loader()

        # checkpoint resume point if available
        start_iter = 0
        if resume_from is not None and os.path.exists(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.opt_generator.load_state_dict(checkpoint['opt_generator'])
            self.opt_critic.load_state_dict(checkpoint['opt_critic'])
            self.g_losses = checkpoint.get('g_losses', [])
            self.c_losses = checkpoint.get('c_losses', [])
            start_iter = checkpoint.get('generator_iter', 0)
            print(f"  Resumed from checkpoint: {resume_from}")
            print(f"  Continuing from generator iteration {start_iter}")

        # set networks to training mode
        self.generator.train()
        self.critic.train()

        # if no checkpoint directory exists, create it
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # control prints
        print(f"  Device          : {self.device}")
        print(f"  Samples         : {len(X)} | Features: {X.shape[1]}")
        print(f"  Generator iters : {generator_iters} | Batch size: {batch_size}")

        # training loop begin

        for g_iter in tqdm(range(start_iter, generator_iters), desc=" Training"):

            # critic loss update, update it n_critic times
            c_loss_accum = 0.0

            for _ in range(self.n_critic):
                real_batch = next(data_iter)[0]
                current_batch = real_batch.size(0)

                # generate fake batch (with random noise). detached so gradients don't flow into generator
                z = torch.randn(current_batch, self.latent_dim, device=self.device)
                fake_batch = self.generator(z).detach()

                # real batch from dataset, fake batch from generator
                real_score = self.critic(real_batch).mean()
                fake_score = self.critic(fake_batch).mean()

                gp = gradient_penalty(
                    self.critic, real_batch, fake_batch,
                    self.device, self.lambda_gp
                )

                # critic wants to maximize (real - fake), minimise ((fake - real) + penalty)
                c_loss = fake_score - real_score + gp

                # clear old gradients, compute new gradient, apply update to model parameters
                self.opt_critic.zero_grad()
                c_loss.backward()
                self.opt_critic.step()

                # to keep track of average critic loss over updates
                c_loss_accum += c_loss.item()

            # generate fake batch. do not need to detach in case of the generator
            z = torch.randn(current_batch, self.latent_dim, device=self.device)
            fake_batch = self.generator(z)

            # generator wants to get a high score by the critic (pytorch optimizers minimize losses so we need negative sign)
            g_loss = -self.critic(fake_batch).mean()

            self.opt_generator.zero_grad()
            g_loss.backward()
            self.opt_generator.step()

            # append losses to lists initialized outside of training loop
            self.g_losses.append(g_loss.item())
            self.c_losses.append(c_loss_accum / self.n_critic)

            # logging
            if (g_iter + 1) % log_interval == 0:
                avg_c = sum(self.c_losses[-log_interval:]) / log_interval
                avg_g = sum(self.g_losses[-log_interval:]) / log_interval
                print(f" Iteration {g_iter+1:>6}/{generator_iters}")
                print(f"Critic: {avg_c:+.4f}    Generator: {avg_g:+.4f}")

            # checkpointing
            if checkpoint_dir is not None and (g_iter + 1) % 2000 == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_iter{g_iter+1}.pt"
                )
                torch.save({
                    "generator_iter" : g_iter + 1,
                    "generator" : self.generator.state_dict(),
                    "critic" : self.critic.state_dict(),
                    "opt_generator" : self.opt_generator.state_dict(),
                    "opt_critic" : self.opt_critic.state_dict(),
                    "g_losses" : self.g_losses,
                    "c_losses" : self.c_losses,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        print("Training completed")

    def generate(self, n_samples: int) -> np.ndarray:
        """ Generate n samples of synthetic rows after training is completed.
         Returns float32 numpy array """

        # set to evaluation mode from training. generation can use bigger batch size than training.
        self.generator.eval()
        generated = []
        batch_size = 512

        # do not track gradients (save memory). process in batch_size steps. compute where current batch should stop.
        # create tensor filled with random noise during generation.
        # convert tensor back to numpy array (which lives in cpu, not gpu)
        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                z = torch.randn(end - start, self.latent_dim, device=self.device)
                batch = self.generator(z).cpu().numpy()
                generated.append(batch)

        # reset generator back to training mode. basically restore the method.
        # stack generated batches into one array, convert to float32 and return
        self.generator.train()
        return np.vstack(generated).astype(np.float32)

    def plot_losses(self, label: str, save_dir: str) -> None:
        """ Generate plots to visualize loss curves after training.
         Save image to save_dir """
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(self.c_losses, label="Critic",    color="steelblue",  linewidth=0.8, alpha=0.9)
        ax.plot(self.g_losses, label="Generator", color="darkorange", linewidth=0.8, alpha=0.9)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f"WGAN-GP Training Losses — {label}")
        ax.set_xlabel("Generator Iteration")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()

        safe = label.replace(' ', '_').replace('/', '_')
        path = os.path.join(save_dir, f"{safe}_losses.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Loss plot saved → {path}")
