import os
import torch
from torch import nn
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from mash_autoencoder.Dataset.mash import MashDataset
from mash_autoencoder.Method.time import getCurrentTime
from mash_autoencoder.Method.path import createFileFolder
from mash_autoencoder.Model.kl_ae import MashKLAutoEncoder
from mash_autoencoder.Module.logger import Logger


class Trainer(object):
    def __init__(
        self,
        dataset_root_folder_path: str,
        batch_size: int = 400,
        accum_iter: int = 1,
        num_workers: int = 4,
        model_file_path: Union[str, None] = None,
        dtype=torch.float32,
        device: str = "cpu",
        warm_epoch_step_num: int = 20,
        warm_epoch_num: int = 10,
        finetune_step_num: int = 400,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        factor: float = 0.9,
        patience: int = 1,
        min_lr: float = 1e-4,
        save_result_folder_path: Union[str, None] = None,
        save_log_folder_path: Union[str, None] = None,
    ) -> None:
        self.loss_kl_weight = 1e-3

        self.accum_iter = accum_iter
        self.dtype = dtype
        self.device = device

        self.warm_epoch_step_num = warm_epoch_step_num
        self.warm_epoch_num = warm_epoch_num

        self.finetune_step_num = finetune_step_num

        self.step = 0
        self.loss_min = float("inf")

        self.best_params_dict = {}

        self.lr = lr
        self.weight_decay = weight_decay
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr

        self.save_result_folder_path = save_result_folder_path
        self.save_log_folder_path = save_log_folder_path
        self.save_file_idx = 0
        self.logger = Logger()

        self.train_loader = DataLoader(
            MashDataset(dataset_root_folder_path, 'train'),
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.val_loader = DataLoader(
            MashDataset(dataset_root_folder_path, 'val'),
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.model = MashKLAutoEncoder(dtype=self.dtype, device=self.device).to(self.device)

        self.loss_fn = nn.MSELoss()

        self.initRecords()

        if model_file_path is not None:
            self.loadModel(model_file_path)

        self.min_lr_reach_time = 0
        return

    def initRecords(self) -> bool:
        self.save_file_idx = 0

        current_time = getCurrentTime()

        if self.save_result_folder_path == "auto":
            self.save_result_folder_path = "./output/" + current_time + "/"
        if self.save_log_folder_path == "auto":
            self.save_log_folder_path = "./logs/" + current_time + "/"

        if self.save_result_folder_path is not None:
            os.makedirs(self.save_result_folder_path, exist_ok=True)
        if self.save_log_folder_path is not None:
            os.makedirs(self.save_log_folder_path, exist_ok=True)
            self.logger.setLogFolder(self.save_log_folder_path)
        return True

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print("[ERROR][Trainer::loadModel]")
            print("\t model file not exist!")
            print("\t model_file_path:", model_file_path)
            return False

        model_state_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_state_dict["model"])
        return True

    def getLr(self, optimizer) -> float:
        return optimizer.state_dict()["param_groups"][0]["lr"]

    def toTrainStepNum(self, scheduler: LRScheduler) -> int:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            return self.finetune_step_num

        if scheduler.T_mult == 1:
            warm_epoch_num = scheduler.T_0 * self.warm_epoch_num
        else:
            warm_epoch_num = int(
                scheduler.T_mult
                * (1.0 - pow(scheduler.T_mult, self.warm_epoch_num))
                / (1.0 - scheduler.T_mult)
            )

        return self.warm_epoch_step_num * warm_epoch_num

    def trainStep(
        self,
        data: dict,
        optimizer: Optimizer,
    ) -> dict:
        for key in data.keys():
            data[key] = data[key].to(self.device)

        gt_mash_params = data["mash_params"]

        results = self.model(data)

        mash_params = results['mash_params']
        kl = results['kl']

        loss_mash_params = self.loss_fn(mash_params, gt_mash_params)
        loss_kl = torch.sum(kl) / kl.shape[0]

        loss_kl = self.loss_kl_weight * loss_kl

        loss = loss_mash_params + loss_kl

        accum_loss = loss / self.accum_iter
        accum_loss.backward()

        if (self.step + 1) % self.accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_dict = {
            "LossMashParams": loss_mash_params.item(),
            "LossKL": loss_kl.item(),
            "Loss": loss.item(),
        }

        return loss_dict

    @torch.no_grad()
    def valStep(self) -> dict:
        avg_loss = 0
        avg_loss_mash_params = 0
        avg_loss_kl = 0
        avg_positive_occ_percent = 0
        ni = 0

        print("[INFO][Trainer::valStep]")
        print("\t start val loss and acc...")
        for data in tqdm(self.val_loader):
            for key in data.keys():
                data[key] = data[key].to(self.device)

            gt_mash_params = data["mash_params"]

            results = self.model(data)

            mash_params = results['mash_params']
            kl = results['kl']

            loss_mash_params = self.loss_fn(mash_params, gt_mash_params)
            loss_kl = torch.sum(kl) / kl.shape[0]

            loss_kl = self.loss_kl_weight * loss_kl

            loss = loss_mash_params + loss_kl

            avg_loss_mash_params += loss_mash_params.item()
            avg_loss_kl += loss_kl.item()
            avg_loss += loss.item()

            ni += 1

        avg_loss /= ni
        avg_loss_mash_params /= ni
        avg_loss_kl /= ni
        avg_positive_occ_percent /= ni

        loss_dict = {
            "LossMashParams": avg_loss_mash_params,
            "LossKL": avg_loss_kl,
            "Loss": avg_loss,
        }

        return loss_dict

    def checkStop(
        self, optimizer: Optimizer, scheduler: LRScheduler, loss_dict: dict
    ) -> bool:
        if not isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step(loss_dict["Loss"])

            if self.getLr(optimizer) == self.min_lr:
                self.min_lr_reach_time += 1

            return self.min_lr_reach_time > self.patience

        current_warm_epoch = self.step / self.warm_epoch_step_num
        scheduler.step(current_warm_epoch)

        return current_warm_epoch >= self.warm_epoch_num

    def train(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> bool:
        train_step_num = self.toTrainStepNum(scheduler)
        final_step = self.step + train_step_num

        eval_step = 0

        print("[INFO][Trainer::train]")
        print("\t start training ...")

        loss_dict_list = []
        while self.step < final_step:
            self.model.train()

            pbar = tqdm(total=len(self.train_loader))
            for data in self.train_loader:
                train_loss_dict = self.trainStep(data, optimizer)

                loss_dict_list.append(train_loss_dict)

                lr = self.getLr(optimizer)

                if (self.step + 1) % self.accum_iter == 0:
                    for key in train_loss_dict.keys():
                        value = 0
                        for i in range(len(loss_dict_list)):
                            value += loss_dict_list[i][key]
                        value /= len(loss_dict_list)
                        self.logger.addScalar("Train/" + key, value, self.step)
                    self.logger.addScalar("Train/Lr", lr, self.step)

                    loss_dict_list = []

                pbar.set_description(
                    "LOSS %.6f LR %.4f"
                    % (
                        train_loss_dict["Loss"],
                        self.getLr(optimizer) / self.lr,
                    )
                )

                self.step += 1
                pbar.update(1)

                if self.checkStop(optimizer, scheduler, train_loss_dict):
                    break

                if self.step >= final_step:
                    break

            pbar.close()

            eval_step += 1
            if eval_step % 1 == 0:
                print("[INFO][Trainer::train]")
                print("\t start eval on val dataset...")
                self.model.eval()
                eval_loss_dict = self.valStep()

                if self.logger.isValid():
                    for key, item in eval_loss_dict.items():
                        self.logger.addScalar("Eval/" + key, item, self.step)

                print(
                    " loss mash params:",
                    eval_loss_dict["LossMashParams"],
                    " loss kl:",
                    eval_loss_dict["LossKL"],
                    " loss:",
                    eval_loss_dict["Loss"],
                )

                self.autoSaveModel(eval_loss_dict["LossMashParams"])

            # self.autoSaveModel(train_loss_dict['LossMashParams'])

        return True

    def autoTrain(
        self,
    ) -> bool:
        print("[INFO][Trainer::autoTrain]")
        print("\t start auto train mash occ decoder...")

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        warm_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1)
        finetune_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.factor,
            patience=self.patience,
            min_lr=self.min_lr,
        )

        self.train(optimizer, warm_scheduler)
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr
        self.train(optimizer, finetune_scheduler)

        return True

    def saveModel(self, save_model_file_path: str) -> bool:
        createFileFolder(save_model_file_path)

        model_state_dict = {
            "model": self.model.state_dict(),
            "loss_min": self.loss_min,
        }

        torch.save(model_state_dict, save_model_file_path)

        return True

    def autoSaveModel(self, value: float, check_lower: bool = True) -> bool:
        if self.save_result_folder_path is None:
            return False

        save_last_model_file_path = self.save_result_folder_path + "model_last.pth"

        self.saveModel(save_last_model_file_path)

        if self.loss_min == float("inf"):
            if not check_lower:
                self.loss_min = -float("inf")

        if check_lower:
            if value > self.loss_min:
                return False
        else:
            if value < self.loss_min:
                return False

        self.loss_min = value

        save_best_model_file_path = self.save_result_folder_path + "model_best.pth"

        self.saveModel(save_best_model_file_path)

        return True
