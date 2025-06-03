import os
import torch
import numpy as np
from typing import Union

from ma_sh.Model.mash import Mash
from ma_sh.Method.rotate import toRotateVectorsFromOrthoPoses

from mesh_graph_cut.Module.mesh_graph_cutter import MeshGraphCutter

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

        data_dict = {
            "surface_pts": surface_pts,
        }

        result_dict = self.model(data_dict)

        return result_dict

    @torch.no_grad()
    def detectMeshFile(
        self, mesh_file_path: str, anchor_num: int = 4000
    ) -> Union[dict, None]:
        if not os.path.exists(mesh_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mesh file not exist!")
            print("\t mesh_file_path:", mesh_file_path)
            return None

        mesh_graph_cutter = MeshGraphCutter(mesh_file_path)
        mesh_graph_cutter.cutMesh(anchor_num)

        surface_pts = mesh_graph_cutter.sub_mesh_sample_points

        result_dict = self.detect(surface_pts)

        return result_dict
