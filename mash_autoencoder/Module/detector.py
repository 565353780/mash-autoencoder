import os
import torch
import numpy as np
from typing import Union

# from mash_autoencoder.Model.shape_vae import ShapeVAE
# from mash_autoencoder.Model.mash_vae import MashVAE
# from mash_autoencoder.Model.vae_simple import VAE
# from mash_autoencoder.Model.mash_vae_tr import KLAutoEncoder
from mash_autoencoder.Model.ptv3_shape_vae import PTV3ShapeVAE


class Detector(object):
    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = device

        model_id = 4
        if model_id == 1:
            self.model = MashVAE(dtype=self.dtype, device=self.device).to(self.device)
        elif model_id == 2:
            self.model = VAE().to(self.device)
        elif model_id == 3:
            self.model = KLAutoEncoder().to(self.device)
        elif model_id == 4:
            self.model = PTV3ShapeVAE().to(self.device)

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

        positions = mash_params["positions"]
        positions_tensor = torch.from_numpy(positions).unsqueeze(0).type(self.dtype).to(self.device)

        x, kl = self.model.encode(positions_tensor, 0.0, True)
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

        positions = mash_params["positions"]
        positions_tensor = torch.from_numpy(positions).unsqueeze(0).type(self.dtype).to(self.device)

        data = {'positions': positions_tensor}

        results = self.model(data)

        results['mash_params_dict']['positions'] = positions_tensor
        return results
