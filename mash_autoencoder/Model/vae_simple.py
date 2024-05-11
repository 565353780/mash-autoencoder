import torch
from torch import nn
from typing import Tuple

from mash_autoencoder.Model.Layer.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)


class VAE(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
    ):
        super().__init__()
        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2

        encode_dims = [22, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        encode_dims = [22, 22, 22, 22, 22, 22, 22, 22]
        d_latent = encode_dims[-1]

        self.encoder = nn.Sequential()
        for i in range(len(encode_dims) - 2):
            self.encoder.append(nn.Linear(encode_dims[i], encode_dims[i + 1]))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(encode_dims[-2], encode_dims[-1]))

        self.mean_fc = nn.Linear(d_latent, d_latent)
        self.logvar_fc = nn.Linear(d_latent, d_latent)

        self.decoder = nn.Sequential()
        self.decoder.append(nn.Linear(encode_dims[-1], encode_dims[-2]))
        for i in range(len(encode_dims) - 2, 0, -1):
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Linear(encode_dims[i], encode_dims[i - 1]))
        return

    def encode(self, mash_params: torch.Tensor, drop_prob: float = 0.0, deterministic: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if drop_prob > 0.0:
            mask = mash_params.new_empty(*mash_params.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            mash_params = mash_params * mask.unsqueeze(-1).expand_as(mash_params).type(mash_params.dtype)

        x = self.encoder(mash_params)

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()
        return x, kl

    def decode(self, x):
        mash_params = self.decoder(x)
        return mash_params

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool=False):
        mash_params = data['mash_params']

        x, kl = self.encode(mash_params, drop_prob, deterministic)

        output = self.decode(x)

        return {"mash_params": output, "kl": kl}
