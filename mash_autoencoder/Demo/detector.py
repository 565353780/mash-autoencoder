import sys

sys.path.append("../ma-sh")
sys.path.append("../wn-nc")
sys.path.append("../point-cept/")
sys.path.append("../diff-curvature/")
sys.path.append("../mesh-graph-cut/")
sys.path.append("../chamfer-distance/")

import os
import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.simple_mash import SimpleMash
from ma_sh.Method.pcd import getPointCloud

from mash_autoencoder.Module.detector import Detector


def demo():
    model_file_path = "./output/ptv3-v1/model_best.pth"
    dtype = torch.float32
    device = "cuda:1"
    mash_device = "cpu"

    mesh_file_path = os.environ["HOME"] + "/chLi/Dataset/vae-eval/mesh/000.obj"
    anchor_num = 4000
    batch_size = 400
    points_per_submesh = 10001

    print("model_file_path:", model_file_path)
    detector = Detector(model_file_path, dtype, device)

    result_dict = detector.detectMeshFile(
        mesh_file_path, anchor_num, points_per_submesh, batch_size
    )

    if result_dict is None:
        print("detectMeshFile failed!")
        return False

    mask_params = result_dict["mask_params"].to(mash_device)
    sh_params = result_dict["sh_params"].to(mash_device)
    rotate_vectors = result_dict["rotate_vectors"].to(mash_device)
    positions = result_dict["positions"].to(mash_device)

    mash = SimpleMash(
        anchor_num=anchor_num,
        mask_degree_max=3,
        sh_degree_max=2,
        dtype=dtype,
        device=mash_device,
    )

    print("start load params...")
    mash.loadParams(
        mask_params=mask_params,
        sh_params=sh_params,
        rotate_vectors=rotate_vectors,
        positions=positions,
    )

    mash.saveParamsFile("./output/vae-eval_000_mash.npy", overwrite=True)

    surface_pts = result_dict["surface_pts"].view(-1, 3)
    boundary_pts = result_dict["boundary_pts"].view(-1, 3)

    surface_pcd = getPointCloud(surface_pts.cpu().numpy())
    boundary_pcd = getPointCloud(boundary_pts.cpu().numpy())

    print("start write_point_cloud...")
    o3d.io.write_point_cloud(
        "./output/vae-eval_000_surface_pcd.ply", surface_pcd, write_ascii=True
    )
    o3d.io.write_point_cloud(
        "./output/vae-eval_000_boundary_pcd.ply", boundary_pcd, write_ascii=True
    )

    # copyfile(mesh_file_path, "./output/vae-eval_000_gt.obj")

    return True
