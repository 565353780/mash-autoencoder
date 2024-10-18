import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
        split: str = "train",
        preload_data: bool = False,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path
        self.split = split
        self.preload_data = preload_data

        self.mash_folder_path = self.dataset_root_folder_path + "MashV4/"
        self.split_folder_path = self.dataset_root_folder_path + "SplitMashKLAutoEncoder/"
        assert os.path.exists(self.mash_folder_path)
        assert os.path.exists(self.split_folder_path)

        self.paths_list = []

        dataset_name_list = os.listdir(self.split_folder_path)

        for dataset_name in dataset_name_list:
            mash_split_folder_path = self.split_folder_path + dataset_name + "/"

            categories = os.listdir(mash_split_folder_path)
            # FIXME: for detect test only
            if self.split == "test":
                # categories = ["02691156"]
                categories = ["03001627"]

            for j, category in enumerate(categories):
                modelid_list_file_path = (
                    mash_split_folder_path + category + "/" + self.split + ".txt"
                )
                if not os.path.exists(modelid_list_file_path):
                    continue

                with open(modelid_list_file_path, "r") as f:
                    modelid_list = f.read().split()

                print("[INFO][MashDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(j + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for modelid in tqdm(modelid_list):
                    mash_file_path = (
                        self.mash_folder_path
                        + dataset_name
                        + "/"
                        + category
                        + "/"
                        + modelid
                        + ".npy"
                    )

                    if self.preload_data:
                        mash_params = np.load(mash_file_path, allow_pickle=True).item()
                        self.paths_list.append(mash_params)
                    else:
                        self.paths_list.append(mash_file_path)

        return

    def __len__(self):
        if self.split == "train":
            return len(self.paths_list)
        else:
            return len(self.paths_list)

    def __getitem__(self, index: int):
        index = index % len(self.paths_list)

        if self.split == "train":
            np.random.seed()
        else:
            np.random.seed(1234)

        if self.preload_data:
            mash_params = self.paths_list[index]
        else:
            mash_file_path = self.paths_list[index]
            mash_params = np.load(mash_file_path, allow_pickle=True).item()

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        if self.split == "train":
            scale_range = [0.8, 1.2]
            move_range = [-0.5, 0.5]

            random_scale = (
                scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
            )
            random_translate = move_range[0] + (
                move_range[1] - move_range[0]
            ) * np.random.rand(3)

            mash_params = np.hstack(
                [
                    positions * random_scale + random_translate,
                    rotate_vectors,
                    mask_params,
                    sh_params * random_scale,
                ]
            )
        else:
            mash_params = np.hstack(
                [
                    positions,
                    rotate_vectors,
                    mask_params,
                    sh_params,
                ]
            )

        mash_params = mash_params[np.random.permutation(mash_params.shape[0])]

        feed_dict = {
            "mash_params": torch.tensor(mash_params).float(),
        }

        return feed_dict
