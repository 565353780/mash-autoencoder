import sys

sys.path.append("../ma-sh")
sys.path.append("../wn-nc")
sys.path.append("../point-cept/")
sys.path.append("../mesh-graph-cut/")

import os
import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud

from mash_autoencoder.Module.detector import Detector


def demo():
    model_file_path = "./output/ptv3-v1/model_best.pth"
    dtype = torch.float32
    device = "cuda:0"
    mash_device = "cuda:0"

    mesh_file_path = os.environ["HOME"] + "/chLi/Dataset/vae-eval/mesh/000.obj"
    anchor_num = 4000

    print("model_file_path:", model_file_path)
    detector = Detector(model_file_path, dtype, device)

    result_dict = detector.detectMeshFile(mesh_file_path, anchor_num)

    if result_dict is None:
        print("detectMeshFile failed!")
        return False

    mask_params = result_dict["mask_params"].to(mash_device)
    sh_params = result_dict["sh_params"].to(mash_device)
    rotate_vectors = result_dict["rotate_vectors"].to(mash_device)
    positions = result_dict["positions"].to(mash_device)

    mash = Mash(
        anchor_num=anchor_num,
        mask_degree_max=3,
        sh_degree_max=2,
        mask_boundary_sample_num=90,
        sample_polar_num=1000,
        sample_point_scale=0.8,
        use_inv=True,
        dtype=dtype,
        device=mash_device,
    )

    mash.loadParams(
        mask_params=mask_params,
        sh_params=sh_params,
        rotate_vectors=rotate_vectors,
        positions=positions,
    )

    mash_pcd = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))
    o3d.io.write_point_cloud(
        "./output/vae-eval_000_pcd.ply", mash_pcd, write_ascii=True
    )

    copyfile(mesh_file_path, "./output/vae-eval_000_gt.obj")

    return True
