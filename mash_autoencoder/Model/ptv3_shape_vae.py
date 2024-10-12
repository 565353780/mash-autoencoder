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
                 latent_dim: int = 64,
                 positional_embedding_dim: int = 48) -> None:
        super().__init__()

        mask_dim = 2 * mask_degree + 1
        sh_dim = (sh_degree + 1) ** 2

        output_dim = 3 + mask_dim + sh_dim

        ptv3_output_dim = 64

        self.embedding = PointEmbed(hidden_dim=positional_embedding_dim, dim=1)

        self.feature_encoder = PointTransformerV3(self.embedding.embedding_dim)

        self.mean_fc = nn.Linear(ptv3_output_dim, latent_dim)
        self.logvar_fc = nn.Linear(ptv3_output_dim, latent_dim)

        self.proj = nn.Linear(latent_dim, ptv3_output_dim)

        self.shape_decoder = nn.Linear(ptv3_output_dim, output_dim)
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
            'batch': torch.cat([torch.ones(positions.shape[1], dtype=torch.long, device='cuda') * i for i in range(positions.shape[0])]),
            'grid_size': 0.01,
        }

        points = self.feature_encoder(input_dict)

        x = points.feat.reshape(positions.shape[0], positions.shape[1], -1)

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()

        return x, kl

    def decode(self, x, positions):
        x = self.proj(x)

        shape_params = self.shape_decoder(x)

        predict_mash_params = torch.cat([positions, shape_params], dim=2)
        return predict_mash_params

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool=False):
        mash_params = data['mash_params']
        positions = mash_params[:, :, :3]

        x, kl = self.encode(positions, drop_prob, deterministic)

        output = self.decode(x, positions)

        return {"mash_params": output, "kl": kl}
