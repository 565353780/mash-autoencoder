import sys
sys.path.append('/home/chli/github/POINTS/point-cept/')

import torch
from torch import nn
from typing import Tuple

from pointcept.models.point_transformer_v3 import PointTransformerV3

from mash_autoencoder.Model.Layer.point_embed import PointEmbed
from mash_autoencoder.Model.Layer.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)

class PTV3ShapeVAE(nn.Module):
    def __init__(self,
                 mask_degree: int = 3,
                 sh_degree: int = 2,
                 latent_dim: int = 512,
                 positional_embedding_dim: int = 256) -> None:
        super().__init__()

        mask_dim = 2 * mask_degree + 1
        sh_dim = (sh_degree + 1) ** 2

        ptv3_output_dim = 64

        self.embedding = PointEmbed(hidden_dim=positional_embedding_dim, dim=1)

        self.feature_encoder = PointTransformerV3(self.embedding.embedding_dim)

        self.mean_fc = nn.Linear(ptv3_output_dim, latent_dim)
        self.logvar_fc = nn.Linear(ptv3_output_dim, latent_dim)

        self.proj = nn.Linear(latent_dim, ptv3_output_dim)

        self.rotate_vectors_decoder = nn.Linear(ptv3_output_dim, 3)

        self.mask_params_decoder = nn.Linear(ptv3_output_dim, mask_dim)

        self.sh_params_decoder = nn.Linear(ptv3_output_dim, sh_dim)
        return

    def encode(self, positions: torch.Tensor, drop_prob: float = 0.0, deterministic: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if drop_prob > 0.0:
            mask = positions.new_empty(*positions.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            positions = positions * mask.unsqueeze(-1).expand_as(positions).type(positions.dtype)

        position_embeddings = self.embedding.embed(positions, self.embedding.basis)

        input_dict = {
            'coord': positions.reshape(-1, 3),
            'feat': position_embeddings.reshape(-1, self.embedding.embedding_dim),
            'batch': torch.cat([torch.ones(positions.shape[1], dtype=torch.long, device=positions.device) * i for i in range(positions.shape[0])]),
            'grid_size': 0.1,
        }

        points = self.feature_encoder(input_dict)

        x = points.feat.reshape(positions.shape[0], positions.shape[1], -1)

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()

        return x, kl

    def decode(self, x):
        x = self.proj(x)

        rotate_vectors = self.rotate_vectors_decoder(x)
        mask_params = self.mask_params_decoder(x)
        sh_params = self.sh_params_decoder(x)

        mash_dict = {
            'rotate_vectors': rotate_vectors,
            'mask_params': mask_params,
            'sh_params': sh_params,
        }
        return mash_dict

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool=False):
        positions = data['positions']

        x, kl = self.encode(positions, drop_prob, deterministic)

        output = self.decode(x)

        return {"mash_params_dict": output, "kl": kl}
