import torch
from typing import Tuple
from torch import nn
from torch.nn import ReLU
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch.nn.utils import weight_norm
from torch_cluster import knn


from mash_autoencoder.Model.Layer.pre_norm import PreNorm
from mash_autoencoder.Model.Layer.feed_forward import FeedForward
from mash_autoencoder.Model.Layer.attention import Attention
from mash_autoencoder.Model.Layer.point_embed import PointEmbed
from mash_autoencoder.Model.Layer.point_conv import PointConv
from mash_autoencoder.Model.Layer.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from mash_autoencoder.Method.io import exists
from mash_autoencoder.Method.cache import cache_fn


class ShapeVAE(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        depth=12,
        dim=512,
        latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()

        mask_dim = 2 * mask_degree + 1
        sh_dim = (sh_degree + 1) ** 2

        output_dim = 3 + mask_dim + sh_dim

        self.depth = depth

        self.point_embed = PointEmbed(dim=dim)

        self.conv = PointConv(
            local_nn=Seq(
                weight_norm(Lin(3 + self.point_embed.embedding_dim, 256)),
                ReLU(True),
                weight_norm(Lin(256, 256)),
            ),
            global_nn=Seq(
                weight_norm(Lin(256, 256)), ReLU(True), weight_norm(Lin(256, dim))
            ),
        )

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        def get_latent_attn():
            return PreNorm(
                dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
            )

        def get_latent_ff():
            return PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            dim,
            Attention(dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(dim, FeedForward(dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(dim, output_dim) if exists(output_dim) else nn.Identity()
        )

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, positions: torch.Tensor, drop_prob: float = 0.0, deterministic: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if drop_prob > 0.0:
            mask = positions.new_empty(*positions.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            positions = positions * mask.unsqueeze(-1).expand_as(positions).type(positions.dtype)

        # pc: B x N x 3
        B, N = positions.shape[:2]

        pos = positions.view(B * N, 3)

        batch = torch.arange(B).to(pos.device)
        batch = torch.repeat_interleave(batch, N)

        row, col = knn(pos, pos, 10, batch, batch)
        edge_index = torch.stack([col, row], dim=0)

        x = self.conv(pos, pos, edge_index, self.point_embed.basis)
        x = x.view(B, -1, x.shape[-1])

        position_embeddings = self.point_embed(positions)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(x, context=position_embeddings, mask=None)
            + x
        )
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()

        return x, kl

    def decode(self, x, positions):
        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        position_embeddings = self.point_embed(positions)
        latents = self.decoder_cross_attn(position_embeddings, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        decoded_mash_params = self.to_outputs(latents)
        return torch.cat([positions, decoded_mash_params], dim=2)

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool=False):
        mash_params = data['mash_params']
        positions = mash_params[:, :, :3]

        x, kl = self.encode(positions, drop_prob, deterministic)

        output = self.decode(x, positions)

        return {"mash_params": output, "kl": kl}
