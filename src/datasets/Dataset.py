import torch
import pickle, os
import numpy as np
import os.path as osp
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import torch_geometric.transforms as T

from src.utils import *

os_sep = os.sep
path = os.sep.join(__file__.split(os.sep)[:-3])
file_path = osp.join(path, "data")
data_split = ["complete", "public", "full", "random"]

class pickleDataset(torch.utils.data.Dataset):
    def __init__(self, name, samples_per_epoch=100):
        assert_error(name.endswith(".pickle"), ValueError, 5321)

        with open(f'{osp.join(file_path, name)}', 'rb') as f:
            self._load_ = pickle.load(f) # Load the data
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        raise NotImplementedError()

class TadpoleDataset(pickleDataset):
    def __init__(self,
                 name="tadpole_data.pickle",
                 fold=0,
                 mode="train",
                 samples_per_epoch=100,
                 device='cpu',
                 full=False):

        super().__init__(name=name, samples_per_epoch=samples_per_epoch)
        x, y, train_mask, test_mask, weight = self._load_
        if not full:
            x = x[..., :30, :] # For DGM we use modality 1 (M1) for both node representation and graph learning.

        mask = "train_mask" if mode == "train" else "test_mask" if mode in ["test", "val"] else False
        assert_error(mask, ValueError, 123)
        mask = locals()[mask]

        self.x = torch.from_numpy(x[:, :, fold]).float().to(device)
        self.y = torch.from_numpy(y[:, :, fold]).long().to(device)
        self.y = self.y if self.y.dim() == 1 else torch.argmax(self.y, dim=1)
        self.mask = torch.from_numpy(mask[:,fold]).to(device)
        self.weight = torch.from_numpy(np.squeeze(weight[:1, fold])).float().to(device)

    def __getitem__(self, idx):
        return self.x, self.y, self.mask, [[]]

class pyGeometricDataset(torch.utils.data.Dataset):
    def __init__(self, data_load, name,
                 mode="train",
                 samples_per_epoch=100,
                 split="complete",
                 normalize=True,
                 transform=None,
                 device='cpu',
                 train_rate=0.5,
                 val_rate=0.25,
                 ):
        assert_error(train_rate+val_rate < 1, ValueError, 6123)
        self.data_load = data_load
        _load_ = self._get_dataset(name=name, split=split, normalize=normalize, transform=transform)
        mask = "train_mask" if mode == "train" else \
               "val_mask" if mode == "val" else \
               "test_mask" if mode == "test" else \
               "pass" if mode == "pass" else False
        assert_error(mask, ValueError, 123)

        dataset = _load_[0]
        self.mask = getattr(dataset, mask).to(device) if mask != "pass" else False

        self.samples_per_epoch = samples_per_epoch
        self.x = dataset.x.float().to(device)
        self.y = dataset.y.long().to(device)
        self.edge_index = dataset.edge_index.to(device)
        self.n_features = dataset.num_node_features
        self.num_classes = _load_.num_classes

        if not self.mask:
            self.train_rate = train_rate
            self.val_rate = val_rate
            pass

    def _get_dataset(self, name, split, normalize, transform):
        raise NotImplementedError()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        raise NotImplementedError()

class PlanetoidDataset(pyGeometricDataset):
    def __init__(self, name,
                 mode="train",
                 samples_per_epoch=100,
                 split="complete",
                 normalize=True,
                 transform=None,
                 device='cpu',
                 **kwargs
                 ):
        super().__init__(Planetoid, name, mode,
                         samples_per_epoch,
                         split, normalize,
                         transform, device)
    def _get_dataset(self, name, split, normalize, transform):
        if split == "complete":
            dataset = self.data_load(file_path, name)
            dataset[0].train_mask.fill_(False)
            dataset[0].train_mask[:-1000] = 1
            dataset[0].val_mask.fill_(False)
            dataset[0].val_mask[-1000:-500] = 1
            dataset[0].test_mask.fill_(False)
            dataset[0].test_mask[-500:] = 1
        else:
            dataset = self.data_load(file_path, name, split=split)
        if transform is not None and normalize:
            dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
        elif normalize:
            dataset.transform = T.NormalizeFeatures()
        elif transform is not None:
            dataset.transform = transform
        return dataset

    def __getitem__(self, idx):
        return self.x, self.y, self.mask, self.edge_index

class AmazonDataset(pyGeometricDataset):
    def __init__(self, name,
                 mode="pass",
                 samples_per_epoch=100,
                 split="complete",
                 normalize=True,
                 transform=None,
                 device='cpu',
                 train_rate=0.5,
                 val_rate=0.25,
                 ):
        super().__init__(Amazon, name, mode,
                         samples_per_epoch,
                         split, normalize,
                         transform, device,
                         train_rate, val_rate)

    def _get_dataset(self, name, split, normalize, transform):
        dataset = self.data_load(file_path, name)
        if transform is not None and normalize:
            dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
        elif normalize:
            dataset.transform = T.NormalizeFeatures()
        elif transform is not None:
            dataset.transform = transform
        return dataset

    def __getitem__(self, idx):
        return self.x, self.y, self.mask, self.edge_index

if __name__ == "__main__":
    print(TadpoleDataset(full=True).y.shape)
    print(AmazonDataset("Photo").x.shape)



