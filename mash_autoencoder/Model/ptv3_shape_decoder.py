import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

from pointcept.models.point_transformer_v3.point_transformer_v3m2_sonata import (
    PointTransformerV3,
)

from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.rotate import (
    toRotateVectorsFromOrthoPoses,
    compute_rotation_matrix_from_ortho6d,
)


class AttentionPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 1),
        )
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        # x: BxNx64
        scores = self.score_net(x)  # BxNx1
        weights = F.softmax(scores, dim=1)  # BxNx1
        features = torch.sum(weights * x, dim=1)  # Bx64
        return self.mlp(features)  # BxM


class MultiPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pool_dim = in_dim // 2
        self.mlp = nn.Sequential(nn.Linear(in_dim * 2, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x):
        # x: BxNx64
        max_feat = torch.max(x, dim=1)[0]  # Bx64
        mean_feat = torch.mean(x, dim=1)  # Bx64
        global_feat = torch.cat([max_feat, mean_feat], dim=1)  # Bx128
        return self.mlp(global_feat)  # BxM


class HybridPooling(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attention = AttentionPooling(in_dim, in_dim)
        self.multi_pool = MultiPooling(in_dim, in_dim)
        self.final_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        # x: BxNx64
        att_feat = self.attention(x)  # BxM
        pool_feat = self.multi_pool(x)  # BxM
        combined = torch.cat([att_feat, pool_feat], dim=1)  # Bx2M
        return self.final_mlp(combined)  # BxM


class PTV3ShapeDecoder(nn.Module):
    def __init__(
        self,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 40,
        sample_theta_num: int = 40,
    ) -> None:
        super().__init__()

        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num

        mask_dim = 2 * mask_degree_max + 1
        sh_dim = (sh_degree_max + 1) ** 2

        ptv3_encode_dim = 512
        ptv3_decode_dim = 64
        self.rot_encoder = PointTransformerV3(
            in_channels=3,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, ptv3_encode_dim),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(ptv3_decode_dim, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
        )

        self.pos_encoder = PointTransformerV3(
            in_channels=3,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, ptv3_encode_dim),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(ptv3_decode_dim, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
        )

        self.shape_encoder = PointTransformerV3(
            in_channels=3,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, ptv3_encode_dim),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(ptv3_decode_dim, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
        )

        if False:
            self.ortho_poses_decoder = nn.Sequential(
                nn.Linear(ptv3_decode_dim, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, 3),
            )

            self.positions_decoder = nn.Sequential(
                nn.Linear(ptv3_decode_dim, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, 3),
            )

            self.mask_params_decoder = nn.Sequential(
                nn.Linear(ptv3_decode_dim, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, mask_dim),
            )

            self.sh_params_decoder = nn.Sequential(
                nn.Linear(ptv3_decode_dim, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, ptv3_decode_dim // 2),
                nn.Linear(ptv3_decode_dim // 2, sh_dim),
            )
        else:
            self.ortho_poses_decoder = HybridPooling(ptv3_decode_dim, 6)

            self.positions_decoder = HybridPooling(ptv3_decode_dim, 3)

            self.mask_params_decoder = HybridPooling(ptv3_decode_dim, mask_dim)

            self.sh_params_decoder = HybridPooling(ptv3_decode_dim, sh_dim)

        return

    def encodeRotation(self, surface_pts: torch.Tensor) -> torch.Tensor:
        batch_size, point_num = surface_pts.shape[:2]

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

        rot_points = self.rot_encoder(input_dict)

        rot_feature = rot_points.feat.reshape(batch_size, point_num, -1)

        ortho_poses = self.ortho_poses_decoder(rot_feature)

        return ortho_poses

    def encodePosition(self, surface_pts: torch.Tensor) -> torch.Tensor:
        batch_size, point_num = surface_pts.shape[:2]

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

        pos_points = self.pos_encoder(input_dict)

        pos_feature = pos_points.feat.reshape(batch_size, point_num, -1)

        positions = self.positions_decoder(pos_feature)

        return positions

    def encodeShape(
        self, surface_pts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, point_num = surface_pts.shape[:2]

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

        shape_points = self.shape_encoder(input_dict)

        shape_feature = shape_points.feat.reshape(batch_size, point_num, -1)

        mask_params = self.mask_params_decoder(shape_feature)

        sh_params = self.sh_params_decoder(shape_feature)

        return mask_params, sh_params

    def forward(self, data_dict, drop_prob: float = 0.0, deterministic: bool = False):
        surface_pts = data_dict["surface_pts"]

        batch_size = surface_pts.shape[0]

        if drop_prob > 0.0:
            mask = surface_pts.new_empty(*surface_pts.shape[:2])
            mask = mask.bernoulli_(1 - drop_prob)
            surface_pts = surface_pts * mask.unsqueeze(-1).expand_as(surface_pts).type(
                surface_pts.dtype
            )

        ortho_poses = self.encodeRotation(surface_pts)

        with torch.no_grad():
            ortho_poses_cp = ortho_poses.detach().clone()
            rotate_matrices = compute_rotation_matrix_from_ortho6d(ortho_poses_cp)
            invrot_surface_pts = torch.matmul(surface_pts, rotate_matrices)

        positions = self.encodePosition(invrot_surface_pts)

        mask_params, sh_params = self.encodeShape(invrot_surface_pts)

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
            "mask_params": mask_params,
            "sh_params": sh_params,
            "rotate_vectors": rotate_vectors,
            "positions": positions,
        }

        return result_dict
