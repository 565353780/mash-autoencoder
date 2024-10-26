import sys

sys.path.append("../ma-sh")

import torch
import open3d as o3d
from shutil import copyfile

from ma_sh.Model.mash import Mash
from ma_sh.Method.data import toNumpy
from ma_sh.Method.pcd import getPointCloud

from mash_autoencoder.Dataset.mash import MashDataset
from mash_autoencoder.Module.detector import Detector


def demo():
    model_file_path = "./output/ptv3-v3-1/model_best.pth"
    dtype = torch.float32
    device = "cuda:0"
    mash_device = "cuda:0"

    dataset_root_folder_path = "/home/chli/Dataset/"
    mash_dataset = MashDataset(dataset_root_folder_path, "val")

    print('model_file_path:', model_file_path)
    detector = Detector(model_file_path, dtype, device)

    for i in range(10):
        mash_params_file_path = mash_dataset.paths_list[i]
        gt_mesh_file_path = mash_params_file_path.replace(
            mash_dataset.mash_folder_path + "ShapeNet/",
            "/home/chli/chLi/Dataset/NormalizedMesh/ShapeNet/",
        ).replace(".npy", ".obj")

        if True:
            print("start export mesh", i + 1, "...")
            results = detector.detectFile(mash_params_file_path)
            assert results is not None
            mash_params_dict = results['mash_params_dict']
            if 'kl' in results.keys():
                kl = results['kl'][0]
                print("kl:", kl)

            ortho_poses = mash_params_dict['ortho_poses'][0].to(mash_device)
            positions = mash_params_dict['positions'][0].to(mash_device)
            mask_params = mash_params_dict['mask_params'][0].to(mash_device)
            sh_params = mash_params_dict['sh_params'][0].to(mash_device)

            mash = Mash(
                anchor_num=400,
                mask_degree_max=3,
                sh_degree_max=2,
                mask_boundary_sample_num=90,
                sample_polar_num=1000,
                sample_point_scale=0.8,
                use_inv=True,
                dtype=dtype,
                device=mash_device)
            mash.loadParams(mask_params=mask_params, sh_params=sh_params, positions=positions, ortho6d_poses=ortho_poses)
            mash_pcd = getPointCloud(toNumpy(torch.vstack(mash.toSamplePoints()[:2])))
            o3d.io.write_point_cloud(
                "./output/test_mash_pcd" + str(i) + ".ply", mash_pcd, write_ascii=True
            )

        if True:
            gt_mash = Mash.fromParamsFile(mash_params_file_path, device=device)
            gt_mash_pcd = getPointCloud(toNumpy(torch.vstack(gt_mash.toSamplePoints()[:2])))
            o3d.io.write_point_cloud(
                "./output/test_gt_mash_pcd" + str(i) + ".ply", gt_mash_pcd, write_ascii=True
            )

        if False:
            copyfile(gt_mesh_file_path, "./output/test_mash_mesh_gt" + str(i) + ".obj")

    return True
