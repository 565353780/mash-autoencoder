import sys

sys.path.append("../ma-sh")
sys.path.append("../wn-nc")
sys.path.append("../base-trainer")
sys.path.append("../point-cept/")
sys.path.append("../chamfer-distance/")

import torch

from mash_autoencoder.Module.trainer import Trainer


def demo():
    batch_size = 384
    accum_iter = 1
    num_workers = 16
    model_file_path = "./output/ptv3-v2/model_best.pth"
    model_file_path = None
    weights_only = True
    dtype = torch.float16
    device = "auto"
    warm_step_num = 1000
    finetune_step_num = -1
    lr = 1e-4
    lr_batch_size = 256
    ema_start_step = 5000
    ema_decay_init = 0.99
    ema_decay = 0.999
    save_result_folder_path = "auto"
    save_log_folder_path = "auto"
    best_model_metric_name = "Loss"
    is_metric_lower_better = True
    sample_result_freq = -1
    use_amp = False
    quick_test = False
    drop_prob = 0.0
    deterministic = False
    kl_weight = 1.0

    trainer = Trainer(
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
        sample_result_freq,
        use_amp,
        quick_test,
        drop_prob,
        deterministic,
        kl_weight,
    )

    trainer.train()
    return True
