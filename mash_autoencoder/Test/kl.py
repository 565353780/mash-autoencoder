import torch

from mash_autoencoder.Dataset.mash import MashDataset
from mash_autoencoder.Model.Layer.diagonal_gaussian_distribution import DiagonalGaussianDistribution

def test():
    dataset = MashDataset('/home/chli/Dataset/')

    for data in dataset:
        mash_params = data['mash_params']

        mean = torch.mean(mash_params.unsqueeze(0).unsqueeze(0), dim=3)
        std = torch.std(mash_params.unsqueeze(0).unsqueeze(0), dim=3)
        print(mean.shape)
        print(std.shape)
        postier = DiagonalGaussianDistribution(mean, std)
        kl = postier.kl()
        print(kl)
        exit()
    return True
