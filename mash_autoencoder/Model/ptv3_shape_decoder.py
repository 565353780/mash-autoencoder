import torch
from torch import nn

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)

from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.rotate import toRotateVectorsFromOrthoPoses


class PTV3ShapeDecoder(nn.Module):
    def __init__(
        self,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 100,
        sample_theta_num: int = 100,
    ) -> None:
        super().__init__()

        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num

        mask_dim = 2 * mask_degree_max + 1
        sh_dim = (sh_degree_max + 1) ** 2

        ptv3_output_dim = 64
        self.feature_encoder = PointTransformerV3(
            in_channels=3,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(ptv3_output_dim, 64, 128, 256),
            dec_patch_size=(1024, 1024, 1024, 1024),
        )

        anchor_dim = mask_dim + sh_dim + 9
        self.mash_encoder = nn.Linear(ptv3_output_dim, anchor_dim)

        if False:
            self.ortho_poses_decoder = nn.Sequential(
                nn.Linear(anchor_dim, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, 3),
            )

            self.positions_decoder = nn.Sequential(
                nn.Linear(anchor_dim, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, 3),
            )

            self.mask_params_decoder = nn.Sequential(
                nn.Linear(anchor_dim, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, mask_dim),
            )

            self.sh_params_decoder = nn.Sequential(
                nn.Linear(anchor_dim, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, anchor_dim // 2),
                nn.Linear(anchor_dim // 2, sh_dim),
            )
        else:
            self.ortho_poses_decoder = nn.Linear(anchor_dim, 6)

            self.positions_decoder = nn.Linear(anchor_dim, 3)

            self.mask_params_decoder = nn.Linear(anchor_dim, mask_dim)

            self.sh_params_decoder = nn.Linear(anchor_dim, sh_dim)

        return

    def forward(self, data, drop_prob: float = 0.0, deterministic: bool = False):
        surface_pts = data["surface_pts"]

        batch_size, point_num = surface_pts.shape[:2]

        if drop_prob > 0.0:
            mask = surface_pts.new_empty(*surface_pts.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            surface_pts = surface_pts * mask.unsqueeze(-1).expand_as(surface_pts).type(
                surface_pts.dtype
            )

        flatten_surface_pts = surface_pts.reshape(-1, 3)

        input_dict = {
            "coord": flatten_surface_pts,
            "feat": flatten_surface_pts,
            "batch": torch.cat(
                [
                    torch.ones(
                        point_num,
                        dtype=torch.long,
                        device=surface_pts.device,
                    )
                    * i
                    for i in range(batch_size)
                ]
            ),
            "grid_size": 0.01,
        }

        points = self.feature_encoder(input_dict)

        x = points.feat.reshape(batch_size, point_num, -1)

        x = x.mean(dim=1)

        x = self.mash_encoder(x)

        ortho_poses = self.ortho_poses_decoder(x)
        positions = self.positions_decoder(x)
        mask_params = self.mask_params_decoder(x)
        sh_params = self.sh_params_decoder(x)

        rotate_vectors = toRotateVectorsFromOrthoPoses(ortho_poses)

        mash = SimpleMash(
            anchor_num=batch_size,
            mask_degree_max=self.mask_degree_max,
            sh_degree_max=self.sh_degree_max,
            sample_phi_num=self.sample_phi_num,
            sample_theta_num=self.sample_theta_num,
            use_inv=True,
            idx_dtype=torch.int64,
            dtype=surface_pts.dtype,
            device=surface_pts.device,
        )

        mash.mask_params = mask_params
        mash.sh_params = sh_params
        mash.rotate_vectors = rotate_vectors
        mash.positions = positions

        boundary_pts, inner_pts = mash.toSamplePoints()[:2]

        batched_boundary_pts = boundary_pts.view(batch_size, -1, 3)
        batched_inner_pts = inner_pts.view(batch_size, -1, 3)

        surface_pts = torch.cat([batched_boundary_pts, batched_inner_pts], dim=1)

        result_dict = {
            "surface_pts": surface_pts,
            "boundary_pts": batched_boundary_pts,
            "inner_pts": batched_inner_pts,
        }

        return result_dict
