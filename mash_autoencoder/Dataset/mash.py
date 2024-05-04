import os
import torch
import numpy as np
from torch.utils.data import Dataset

class MashDataset(Dataset):
    def __init__(
        self,
        dataset_root_folder_path: str,
    ) -> None:
        self.dataset_root_folder_path = dataset_root_folder_path

        self.mash_folder_path = self.dataset_root_folder_path + "MashV3/"

        assert os.path.exists(self.mash_folder_path)

        self.path_dict_list = []

        dataset_name_list = os.listdir(self.mash_folder_path)

        for dataset_name in dataset_name_list:
            dataset_folder_path = self.mash_folder_path + dataset_name + "/"

            categories = os.listdir(dataset_folder_path)

            for i, category in enumerate(categories):
                class_folder_path = dataset_folder_path + category + "/"

                mash_filename_list = os.listdir(class_folder_path)

                print("[INFO][MashDataset::__init__]")
                print(
                    "\t start load dataset: "
                    + dataset_name
                    + "["
                    + category
                    + "], "
                    + str(i + 1)
                    + "/"
                    + str(len(categories))
                    + "..."
                )
                for mash_filename in mash_filename_list:
                    path_dict = {}
                    mash_file_path = class_folder_path + mash_filename

                    if not os.path.exists(mash_file_path):
                        continue

                    path_dict['mash'] = mash_file_path

                    self.path_dict_list.append(path_dict)
        return

    def __len__(self):
        return len(self.path_dict_list)

    def __getitem__(self, index):
        index = index % len(self.path_dict_list)

        data = {}

        path_dict = self.path_dict_list[index]

        mash_file_path = path_dict['mash']

        mash_params = np.load(mash_file_path, allow_pickle=True).item()

        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]
        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]

        if True:
            scale_range = [0.9, 1.1]
            move_range = [-0.1, 0.1]

            random_scale = (
                scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
            )
            random_translate = move_range[0] + (
                move_range[1] - move_range[0]
            ) * np.random.rand(3)

            mash_params = np.hstack(
                [
                    rotate_vectors,
                    positions * random_scale + random_translate,
                    mask_params,
                    sh_params * random_scale,
                ]
            )
        else:
            mash_params = np.hstack([rotate_vectors, positions, mask_params, sh_params])

        if False:
            mash_params = mash_params[np.random.permutation(mash_params.shape[0])]

        mash_params = torch.from_numpy(mash_params)

        data['mash_params'] = mash_params.float()
        return data
