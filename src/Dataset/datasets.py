import torch
import pickle, os
import numpy as np
import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

os_sep = os.sep
class pickleDataset(torch.utils.data.Dataset):
    def __init__(self, fold=0, train=True, samples_per_epoch=100, device='cpu'):
        file_path = os_sep.join(__file__.split(os_sep)[:-3])
        with open(f'{file_path}{os_sep}data{os_sep}tadpole_data.pickle', 'rb') as f:
            x, y, train_mask, test_mask, weight = pickle.load(f) # Load the data

        mask = train_mask if train else test_mask
        self.x = torch.from_numpy(x[:, :, fold]).float().to(device)
        self.y = torch.from_numpy(y[:, :, fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight[:1, fold])).float().to(device)
        self.mask = torch.from_numpy(mask[:,fold]).to(device)
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, self.y, self.mask

if __name__ == "__main__":
    print(pickleDataset())
