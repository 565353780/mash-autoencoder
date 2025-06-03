import torch
import numpy as np
from torch.utils.data import Dataset

from ma_sh.Model.simple_mash import SimpleMash


class AnchorDataset(Dataset):
    def __init__(
        self,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        dtype=torch.float64,
    ) -> None:
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.dtype = dtype

        self.mask_dim = 2 * self.mask_degree_max + 1
        self.sh_dim = (self.sh_degree_max + 1) ** 2

        self.data_list = np.zeros([100000])
        return

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        index = index % len(self.data_list)

        mash = SimpleMash(
            anchor_num=1,
            mask_degree_max=self.mask_degree_max,
            sh_degree_max=self.sh_degree_max,
            sample_phi_num=self.sample_phi_num,
            sample_theta_num=self.sample_theta_num,
            use_inv=True,
            idx_dtype=torch.int64,
            dtype=torch.float64,
            device="cpu",
        )

        rand_mask_param = np.random.rand(1, self.mask_dim).astype(np.float64)
        rand_sh_param = np.random.rand(1, self.sh_dim).astype(np.float64)
        rand_rotate_vector = np.random.rand(1, 3).astype(np.float64)
        rand_position = np.random.rand(1, 3).astype(np.float64)

        mash.loadParams(
            mask_params=rand_mask_param,
            sh_params=rand_sh_param,
            rotate_vectors=rand_rotate_vector,
            positions=rand_position,
        )

        boundary_pts, inner_pts = mash.toSamplePoints()[:2]

        surface_pts = torch.vstack([boundary_pts, inner_pts])

        min_xyz = torch.amin(surface_pts, dim=0)
        max_xyz = torch.amax(surface_pts, dim=0)

        center = (min_xyz + max_xyz) * 0.5

        length = torch.max(max_xyz - min_xyz)

        scale = 0.9 / length

        surface_pts = (surface_pts - center) * scale
        boundary_pts = (boundary_pts - center) * scale
        inner_pts = (inner_pts - center) * scale

        # mash.translate(-center)
        # mash.scale(scale)

        data_dict = {
            "surface_pts": surface_pts.float(),
            "boundary_pts": boundary_pts.float(),
            "inner_pts": inner_pts.float(),
        }

        return data_dict
