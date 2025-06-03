import os
import torch
import numpy as np
from tqdm import trange
from typing import Union

from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter

from chamfer_distance.Module.chamfer_distances import ChamferDistances

# from mash_autoencoder.Model.shape_vae import ShapeVAE
# from mash_autoencoder.Model.mash_vae import MashVAE
# from mash_autoencoder.Model.vae_simple import VAE
# from mash_autoencoder.Model.mash_vae_tr import KLAutoEncoder
from mash_autoencoder.Model.ptv3_shape_vae import PTV3ShapeVAE
from mash_autoencoder.Model.ptv3_shape_decoder import PTV3ShapeDecoder


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device

        model_id = 5
        if model_id == 1:
            self.model = MashVAE(dtype=self.dtype, device=self.device).to(self.device)
        elif model_id == 2:
            self.model = VAE().to(self.device)
        elif model_id == 3:
            self.model = KLAutoEncoder().to(self.device)
        elif model_id == 4:
            self.model = PTV3ShapeVAE().to(self.device)
        elif model_id == 5:
            self.model = PTV3ShapeDecoder().to(self.device)

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Detector::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        state_dict = torch.load(model_file_path, map_location="cpu")["model"]

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("[INFO][Detector::loadModel]")
        print("\t load model success!")
        print("\t model_file_path:", model_file_path)
        return True

    @torch.no_grad()
    def detect(self, surface_pts: Union[torch.Tensor, np.ndarray]) -> dict:
        if isinstance(surface_pts, np.ndarray):
            surface_pts = torch.from_numpy(surface_pts)

        surface_pts = surface_pts.to(self.device, dtype=self.dtype)

        min_xyz = torch.amin(surface_pts, dim=1)
        max_xyz = torch.amax(surface_pts, dim=1)

        center = (min_xyz + max_xyz) * 0.5

        length = torch.amax(max_xyz - min_xyz, dim=1)

        scale = 0.9 / length

        normalized_surface_pts = (surface_pts - center[:, None, :]) * scale[
            :, None, None
        ]

        data_dict = {
            "surface_pts": normalized_surface_pts,
        }

        result_dict = self.model(data_dict)

        fit_dist, coverage_dist = ChamferDistances.namedAlgo("cuda")(
            result_dict["surface_pts"], data_dict["surface_pts"]
        )[:2]

        fit_loss = torch.mean(fit_dist)
        coverage_loss = torch.mean(coverage_dist)
        chamfer_loss = fit_loss + coverage_loss

        print(
            "chamfer loss:", fit_loss.item(), coverage_loss.item(), chamfer_loss.item()
        )

        result_dict["surface_pts"] = (
            result_dict["surface_pts"] / scale[:, None, None] + center[:, None, :]
        )
        result_dict["boundary_pts"] = (
            result_dict["boundary_pts"] / scale[:, None, None] + center[:, None, :]
        )
        result_dict["inner_pts"] = (
            result_dict["inner_pts"] / scale[:, None, None] + center[:, None, :]
        )
        result_dict["sh_params"] /= scale[:, None]
        result_dict["positions"] = result_dict["positions"] / scale[:, None] + center

        return result_dict

    @torch.no_grad()
    def detectWithBatch(
        self, surface_pts: Union[torch.Tensor, np.ndarray], batch_size: int = 400
    ) -> dict:
        combined_result = None

        num_batches = (surface_pts.shape[0] + batch_size - 1) // batch_size
        print("[INFO][Detector::detectWithBatch]")
        print("\t start detect with batch, num_batches:", num_batches)
        for i in trange(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, surface_pts.shape[0])
            batch_surface_pts = surface_pts[start_idx:end_idx]

            batch_result = self.detect(batch_surface_pts)

            if combined_result is None:
                combined_result = {k: [] for k in batch_result.keys()}

            for k, v in batch_result.items():
                combined_result[k].append(v)

        final_result = {}
        for k, v_list in combined_result.items():
            if isinstance(v_list[0], torch.Tensor):
                final_result[k] = torch.cat(v_list, dim=0)
            else:
                final_result[k] = v_list

        return final_result

    @torch.no_grad()
    def detectMeshFile(
        self,
        mesh_file_path: str,
        anchor_num: int = 400,
        points_per_submesh: int = 8192,
        batch_size: int = 400,
    ) -> Union[dict, None]:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return None

        mesh_graph_cutter = MeshGraphCutter(mesh_file_path)
        mesh_graph_cutter.cutMesh(anchor_num, points_per_submesh)

        surface_pts = mesh_graph_cutter.sub_mesh_sample_points

        result_dict = self.detectWithBatch(surface_pts, batch_size)

        return result_dict
