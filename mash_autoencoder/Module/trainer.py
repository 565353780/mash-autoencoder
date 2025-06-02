import torch
from torch import nn
from typing import Union

from base_trainer.Module.base_trainer import BaseTrainer

from mash_autoencoder.Dataset.anchor import AnchorDataset
# from mash_autoencoder.Dataset.mash import MashDataset

# from mash_autoencoder.Model.shape_vae import ShapeVAE
# from mash_autoencoder.Model.mash_vae import MashVAE
from mash_autoencoder.Model.vae_simple import VAE
from mash_autoencoder.Model.mash_vae_tr import KLAutoEncoder
from mash_autoencoder.Model.shape_vae_v2 import ShapeVAE
from mash_autoencoder.Model.ptv3_shape_vae import PTV3ShapeVAE
from mash_autoencoder.Model.ptv3_shape_decoder import PTV3ShapeDecoder


class Trainer(BaseTrainer):
    def __init__(
        self,
        batch_size: int = 5,
        accum_iter: int = 10,
        num_workers: int = 16,
        model_file_path: Union[str, None] = None,
        weights_only: bool = False,
        device: str = "cuda:0",
        dtype=torch.float32,
        warm_step_num: int = 2000,
        finetune_step_num: int = -1,
        lr: float = 2e-4,
        lr_batch_size: int = 256,
        ema_start_step: int = 5000,
        ema_decay_init: float = 0.99,
        ema_decay: float = 0.999,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
        best_model_metric_name: Union[str, None] = None,
        is_metric_lower_better: bool = True,
        sample_results_freq: int = -1,
        use_amp: bool = False,
        quick_test: bool = False,
        drop_prob: float = 0.0,
        deterministic: bool = True,
        kl_weight: float = 0.0,
    ) -> None:
        self.drop_prob = drop_prob
        self.deterministic = deterministic
        self.loss_kl_weight = kl_weight

        self.mask_degree_max = 3
        self.sh_degree_max = 2
        self.sample_phi_num = 100
        self.sample_theta_num = 100
        self.dtype = torch.float64

        self.loss_surface_weight = 1.0
        self.loss_boundary_weight = 1.0
        self.loss_inner_weight = 1.0

        self.loss_fn = nn.L1Loss()

        super().__init__(
            batch_size,
            accum_iter,
            num_workers,
            model_file_path,
            weights_only,
            device,
            dtype,
            warm_step_num,
            finetune_step_num,
            lr,
            lr_batch_size,
            ema_start_step,
            ema_decay_init,
            ema_decay,
            save_result_folder_path,
            save_log_folder_path,
            best_model_metric_name,
            is_metric_lower_better,
            sample_results_freq,
            use_amp,
            quick_test,
        )
        return

    def createDatasets(self) -> bool:
        self.dataloader_dict["anchor"] = {
            "dataset": AnchorDataset(
                self.mask_degree_max,
                self.sh_degree_max,
                self.sample_phi_num,
                self.sample_theta_num,
                self.dtype,
            ),
            "repeat_num": 1,
        }

        self.dataloader_dict["eval"] = {
            "dataset": AnchorDataset(
                self.mask_degree_max,
                self.sh_degree_max,
                self.sample_phi_num,
                self.sample_theta_num,
                self.dtype,
            ),
        }

        # crop data num for faster evaluation
        self.dataloader_dict["eval"]["dataset"].data_list = self.dataloader_dict[
            "eval"
        ]["dataset"].data_list[:64]
        return True

    def createModel(self) -> bool:
        model_id = 6
        if model_id == 1:
            self.model = MashVAE(dtype=self.dtype, device=self.device).to(self.device)
        elif model_id == 2:
            self.model = VAE().to(self.device)
        elif model_id == 3:
            self.model = KLAutoEncoder().to(self.device)
        elif model_id == 4:
            self.model = ShapeVAE().to(self.device)
        elif model_id == 5:
            self.model = PTV3ShapeVAE().to(self.device)
        elif model_id == 6:
            self.model = PTV3ShapeDecoder().to(self.device)
        return True

    def preProcessData(self, data_dict: dict, is_training: bool = False) -> dict:
        if is_training:
            data_dict["drop_prob"] = self.drop_prob
        else:
            data_dict["drop_prob"] = 0.0

        return data_dict

    def getLossDict(self, data_dict: dict, result_dict: dict) -> dict:
        gt_surface_pts = data_dict["surface_pts"]
        gt_boundary_pts = data_dict["boundary_pts"]
        gt_inner_pts = data_dict["inner_pts"]

        surface_pts = result_dict["surface_pts"]
        boundary_pts = result_dict["boundary_pts"]
        inner_pts = result_dict["inner_pts"]

        loss_surface = self.loss_fn(surface_pts, gt_surface_pts)
        loss_boundary = self.loss_fn(boundary_pts, gt_boundary_pts)
        loss_inner = self.loss_fn(inner_pts, gt_inner_pts)

        weighted_loss_surface = self.loss_surface_weight * loss_surface
        weighted_loss_boundary = self.loss_boundary_weight * loss_boundary
        weighted_loss_inner = self.loss_inner_weight * loss_inner

        loss_kl = torch.tensor(0, dtype=self.dtype, device=self.device)
        weighted_loss_kl = 0.0
        if "kl" in result_dict.keys():
            kl = result_dict["kl"]
            loss_kl = torch.sum(kl) / kl.shape[0]
            weighted_loss_kl = self.loss_kl_weight * loss_kl

        loss = (
            weighted_loss_surface
            + weighted_loss_boundary
            + weighted_loss_inner
            + weighted_loss_kl
        )

        loss_dict = {
            "LossSurface": loss_surface,
            "LossBoundary": loss_boundary,
            "LossInner": loss_inner,
            "LossKL": loss_kl,
            "Loss": loss,
        }

        return loss_dict

    @torch.no_grad()
    def sampleModelStep(self, model: torch.nn.Module, model_name: str) -> bool:
        return True
        if self.local_rank != 0:
            return True

        dataset = self.dataloader_dict["anchor"]["dataset"]

        model.eval()

        data = dataset.__getitem__(0)

        # process data here
        pcd = data["pcd"]
        mesh = data["mesh"]

        self.logger.addPointCloud(model_name + "/pcd_0", pcd, self.step)
        self.logger.addMesh(model_name + "/mesh_0", mesh, self.step)

        return True
