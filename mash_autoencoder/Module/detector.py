import sys
sys.path.append('../ma-sh/')

import os
import torch
import numpy as np
from typing import Union

from ma_sh.Model.mash import Mash
from ma_sh.Method.rotate import toRotateVectorsFromOrthoPoses

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
        print('[INFO][Detector::loadModel]')
        print('\t load model success!')
        print('\t model_file_path:', model_file_path)
        return True

    @torch.no_grad()
    def encodeFile(self, mash_params_file_path: str) -> Union[dict, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::encodeFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        mash_params = np.load(mash_params_file_path, allow_pickle=True).item()

        surface_points = mash_params["surface_points"]
        surface_points_tensor = torch.from_numpy(surface_points).unsqueeze(0).type(self.dtype).to(self.device)

        x, kl = self.model.encode(surface_points_tensor, 0.0, True)
        results = {'x': x, 'kl': kl}
        return results

    def decodeLatent(self, latent: torch.Tensor) -> dict:
        mash_params_dict = self.model.decode(latent)
        return mash_params_dict

    @torch.no_grad()
    def detectFile(self, mash_params_file_path: str) -> Union[dict, None]:
        if not os.path.exists(mash_params_file_path):
            print("[ERROR][Detector::detectFile]")
            print("\t mash params file not exist!")
            print("\t mash_params_file_path:", mash_params_file_path)
            return None

        mash_params = np.load(mash_params_file_path, allow_pickle=True).item()
        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        mash = Mash(400, 3, 2, 0, 1, 1.0, True, torch.int64, torch.float64, 'cpu')
        mash.loadParams(mask_params, sh_params, rotate_vectors, positions)

        surface_points = mash.toFaceToPoints().unsqueeze(0).type(self.dtype).to(self.device)

        data = {'surface_points': surface_points}

        results = self.model(data)

        return results
