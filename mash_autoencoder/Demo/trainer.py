import torch

from mash_autoencoder.Module.Convertor.mash_split import Convertor
from mash_autoencoder.Module.trainer import Trainer


def demo():
    dataset_root_folder_path = "/home/chli/Dataset/"
    batch_size = 128
    accum_iter = 1
    num_workers = 16
    model_file_path = "./output/ptv3-v1-1/model_best.pth"
    model_file_path = None
    dtype = torch.float16
    device = "cuda:0"
    warm_epoch_step_num = 40
    warm_epoch_num = 40
    finetune_step_num = 100000000
    lr = 1e-4
    weight_decay = 1e-10
    factor = 0.99
    patience = 10000
    min_lr = 1e-6
    drop_prob = 0.0
    deterministic = False
    kl_weight = 1.0
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"

    train_scale = 0.95
    val_scale = 0.05

    convertor = Convertor(dataset_root_folder_path)
    convertor.convertToSplitFiles(train_scale, val_scale)

    trainer = Trainer(
        dataset_root_folder_path,
        batch_size,
        accum_iter,
        num_workers,
        model_file_path,
        dtype,
        device,
        warm_epoch_step_num,
        warm_epoch_num,
        finetune_step_num,
        lr,
        weight_decay,
        factor,
        patience,
        min_lr,
        drop_prob,
        deterministic,
        kl_weight,
        save_result_folder_path,
        save_log_folder_path,
    )

    trainer.autoTrain()
    return True
