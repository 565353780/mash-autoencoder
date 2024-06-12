import torch
from torch import nn
from typing import Tuple


from mash_autoencoder.Model.Layer.pre_norm import PreNorm
from mash_autoencoder.Model.Layer.feed_forward import FeedForward
from mash_autoencoder.Model.Layer.attention import Attention
from mash_autoencoder.Model.Layer.point_embed import PointEmbed
from mash_autoencoder.Model.Layer.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)
from mash_autoencoder.Method.io import exists
from mash_autoencoder.Method.cache import cache_fn


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        d_hidden: int = 512,
        d_hidden_embed: int = 48,
        depth=24,
        d_latent=22,
        heads=8,
        dim_head=64,
        weight_tie_layers=True,
        decoder_ff=True,
    ):
        super().__init__()

        self.mask_dim = 2 * mask_degree + 1
        self.sh_dim = (sh_degree + 1) ** 2
        self.mash_dim = 6 + self.mask_dim + self.sh_dim

        assert d_hidden % 4 == 0

        self.rotation_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.position_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.mask_embed = PointEmbed(self.mask_dim, d_hidden_embed, d_hidden // 4)
        self.sh_embed = PointEmbed(self.sh_dim, d_hidden_embed, d_hidden // 4)

        def get_latent_attn():
            return PreNorm(
                d_hidden, Attention(d_hidden, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
            )

        def get_latent_ff():
            return PreNorm(d_hidden, FeedForward(d_hidden, drop_path_rate=0.1))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.encode_layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.encode_layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decode_layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for _ in range(depth):
            self.decode_layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_ff = (
            PreNorm(d_hidden, FeedForward(d_hidden)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(d_hidden, self.mash_dim)
        )

        self.proj = nn.Linear(d_latent, d_hidden)

        self.mean_fc = nn.Linear(d_hidden, d_latent)
        self.logvar_fc = nn.Linear(d_hidden, d_latent)
        return

    def embedMash(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(mash_params[:, :, :3])
        position_embeddings = self.position_embed(mash_params[:, :, 3:6])
        mask_embeddings = self.mask_embed(mash_params[:, :, 6: 6 + self.mask_dim])
        sh_embeddings = self.sh_embed(mash_params[:, :, 6 + self.mask_dim :])

        mash_embeddings = torch.cat([rotation_embeddings, position_embeddings, mask_embeddings, sh_embeddings], dim=2)
        return mash_embeddings

    def encode(self, mash_params: torch.Tensor, drop_prob: float = 0.0, deterministic: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if drop_prob > 0.0:
            mask = mash_params.new_empty(*mash_params.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            mash_params = mash_params * mask.unsqueeze(-1).expand_as(mash_params).type(mash_params.dtype)

        x = self.embedMash(mash_params)

        for self_attn, self_ff in self.encode_layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar, deterministic)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    def decode(self, x):
        x = self.proj(x)

        for self_attn, self_ff in self.decode_layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if exists(self.decoder_ff):
            x = x + self.decoder_ff(x)

        return self.to_outputs(x)

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool=False):
        mash_params = data['mash_params']

        kl, x = self.encode(mash_params, drop_prob, deterministic)

        output = self.decode(x)

        return {"mash_params": output, "kl": kl}
