from typing import Tuple
import torch
from torch import nn
from functools import partial
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

from mash_autoencoder.Model.mamba_block import create_block, init_weights
from mash_autoencoder.Model.Layer.pre_norm import PreNorm
from mash_autoencoder.Model.Layer.feed_forward import FeedForward
from mash_autoencoder.Model.Layer.attention import Attention
from mash_autoencoder.Model.Layer.point_embed import PointEmbed
from mash_autoencoder.Model.Layer.diagonal_gaussian_distribution import (
    DiagonalGaussianDistribution,
)


class MashKLAutoEncoder(nn.Module):
    def __init__(
        self,
        mask_degree: int = 3,
        sh_degree: int = 2,
        d_hidden: int = 400,
        d_hidden_embed: int = 100,
        d_latent=64,
        n_layer: int = 96,
        n_cross: int = 4,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg=None,
        fused_add_norm=True,
        residual_in_fp32=True,
        dtype=torch.float32,
        device="cuda:0",
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.mask_degree = mask_degree
        self.sh_degree = sh_degree

        assert d_hidden % 4 == 0

        self.rotation_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.position_embed = PointEmbed(3, d_hidden_embed, d_hidden // 4)
        self.mask_embed = PointEmbed(self.mask_dim, d_hidden_embed, d_hidden // 4)
        self.sh_embed = PointEmbed(self.sh_dim, d_hidden_embed, d_hidden // 4)

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    d_hidden, Attention(d_hidden, d_hidden, heads=1, dim_head=d_hidden), context_dim=d_hidden
                ),
                PreNorm(d_hidden, FeedForward(d_hidden)),
            ]
        )

        self.mean_fc = nn.Linear(d_hidden, d_latent)
        self.logvar_fc = nn.Linear(d_hidden, d_latent)

        self.proj = nn.Linear(d_latent, d_hidden)

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_hidden,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_hidden, eps=norm_epsilon, **factory_kwargs
        )

        self.decoder_cross_attn = PreNorm(
            d_hidden,
            Attention(d_hidden, d_hidden, heads=n_cross, dim_head=d_hidden),
            context_dim=d_hidden,
        )
        self.decoder_ff = PreNorm(d_hidden, FeedForward(d_hidden))

        self.to_outputs = nn.Linear(d_hidden, 1)

        self.apply(
            partial(
                init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        return

    def embedMash(self, mash_params: torch.Tensor) -> torch.Tensor:
        rotation_embeddings = self.rotation_embed(mash_params[:, :, :3])
        position_embeddings = self.position_embed(mash_params[:, :, 3:6])
        mask_embeddings = self.mask_embed(mash_params[:, :, 6 : 6 + self.mask_dim])
        sh_embeddings = self.sh_embed(mash_params[:, :, 6 + self.mask_dim :])

        mash_embeddings = torch.cat(
            [rotation_embeddings, position_embeddings, mask_embeddings, sh_embeddings],
            dim=2,
        )
        return mash_embeddings

    def encode(self, mash_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mash_embeddings = self.embedMash(mash_params)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(mash_embeddings, context=mash_embeddings, mask=None)
            + mash_embeddings
        )
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()
        return x, kl

    def decode(self, x):
        hidden_states = self.proj(x)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        latents = self.decoder_cross_attn(hidden_states, context=hidden_states)

        latents = latents + self.decoder_ff(latents)

        mash_params = self.to_outputs(latents)
        return mash_params

    def forward(self, mash_params):
        x, kl = self.encode(mash_params)

        output = self.decode(x)

        return {"mash_params": output, "kl": kl}
