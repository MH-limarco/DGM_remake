import os
import os.path as osp
import pickle
import inspect

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask
from torch_geometric.datasets import Planetoid, Amazon, Coauthor

from src.utils import *

os_sep = os.sep
path = os.sep.join(__file__.split(os.sep)[:-4])
file_path = osp.join(path, "data")

__all__ = [#loading_function
           "Planetoid", "Amazon", "Coauthor",

           ##dataset_class
           "TadpoleDataset",
           "PlanetoidDataset",
           "pyGeometricDataset"
           ]

class pickleDataset(torch.utils.data.Dataset):
    def __init__(self, name, samples_per_epoch=100):
        assert_error(name.endswith(".pickle"), ValueError, 5321)
        with open(f'{osp.join(file_path, name)}', 'rb') as f:
            self._load_ = pickle.load(f) # Load the data
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self):
        raise NotImplementedError()

class TadpoleDataset(pickleDataset):
    def __init__(self,
                 name="tadpole_data.pickle",
                 fold=0,
                 mode="train",
                 samples_per_epoch=100,
                 full=False,
                 **kwargs
                 ):

        super().__init__(name=name, samples_per_epoch=samples_per_epoch)
        x, y, train_mask, test_mask, weight = self._load_
        if not full:
            x = x[..., :30, :] # For dgm we use modality 1 (M1) for both node representation and graph learning.

        mask = "train_mask" if mode == "train" else "test_mask" if mode in ["test", "val"] else False
        assert_error(mask, ValueError, 123)
        mask = locals()[mask]
        self.x = torch.from_numpy(x[:, :, fold]).float()
        self.y = torch.from_numpy(y[:, :, fold]).long()
        self.y = self.y if self.y.dim() == 1 else torch.argmax(self.y, dim=1)
        self.mask = torch.from_numpy(mask[:,fold]).bool()
        self.weight = torch.from_numpy(np.squeeze(weight[:1, fold])).float()
        self.n_features = self.x.size(1)
        self.num_classes = len(self.y.unique())

    def __getitem__(self, idx):
        return self.x, self.y, self.mask, [[]]

########################################## pick_dataset END ##########################################
class Base_pyGDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_load,
                 mode="train",
                 samples_per_epoch=100,
                 split="complete",
                 normalize=True,
                 transform=None,
                 train_rate=0.5,
                 val_rate=0.5,
                 shuffle=True,
                 parent=None
                 ):
        assert_error(train_rate+val_rate <= 1, ValueError, 6123)

        self.data_load = data_load
        _load_ = self._get_dataset(name=name, split=split, normalize=normalize, transform=transform)
        mask = "train_mask" if mode == "train" else \
               "val_mask" if mode == "val" else \
               "test_mask" if mode == "test" else False
        assert_error(mask, ValueError, 123)

        dataset = _load_[0]
        self.samples_per_epoch = samples_per_epoch
        self.x = dataset.x.float()
        self.y = dataset.y.long()
        self.edge_index = dataset.edge_index
        self.n_features = dataset.num_node_features
        self.num_classes = _load_.num_classes

        if not (hasattr(dataset, "train_mask") or hasattr(dataset, "val_mask") or hasattr(dataset, "test_mask")):
            if parent is None:
                _var_ = f"{self.data_load.__name__}_sampler"
                _build_up_ = _var_ not in globals()
            else:
                _var_ = "_sampler_"
                _build_up_ = not hasattr(parent, _var_)

            self.shuffle = shuffle
            self.train_rate = train_rate
            self.val_rate = val_rate
            if _build_up_:
                if parent is None:
                    globals()[_var_] = self._rand_split_mask()
                else:
                    setattr(parent, _var_, self._rand_split_mask())
            if parent is None:
                self.mask = globals()[_var_][mask]
            else:
                self.mask = getattr(parent, _var_)[mask]
        else:
            self.mask = getattr(dataset, mask)

    def _get_dataset(self, name, split, normalize, transform):
        raise NotImplementedError()

    def _rand_split_mask(self):
        size = self.x.size(0)

        _train_ = int(size*self.train_rate)
        _val_ = int(size*self.val_rate)
        _test_ = True if size - _train_ - _val_ > 0 else False

        full_index = torch.randperm(size) if self.shuffle else torch.arange(size)
        train_mask = index_to_mask(full_index[:_train_], size=size)
        val_mask = index_to_mask(full_index[_train_:_train_+_val_], size=size)
        test_mask = index_to_mask(full_index[_train_+_val_:], size=size) if _test_ else \
                    index_to_mask(full_index[_train_:_train_+_val_], size=size)

        return {"train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask}

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        raise NotImplementedError()

class PlanetoidDataset(Base_pyGDataset):
    def __init__(self, name,
                 mode="train",
                 samples_per_epoch=100,
                 split="complete",
                 normalize=True,
                 transform=None,
                 **kwargs
                 ):
        super().__init__(name, Planetoid, mode,
                         samples_per_epoch,
                         split, normalize,
                         transform, **kwargs)
    def _get_dataset(self, name, split, normalize, transform):
        #assert split in ["complete", "public", "full", "random"]
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

class pyGeometricDataset(Base_pyGDataset):
    def __init__(self, name, data_func,
                 mode="train",
                 samples_per_epoch=100,
                 split="complete",
                 normalize=True,
                 transform=None,
                 train_rate=0.7,
                 val_rate=0.25,
                 shuffle=True,
                 **kwargs
                 ):
        super().__init__(name, data_func, mode,
                         samples_per_epoch,
                         split, normalize,transform,
                         train_rate, val_rate, shuffle, **kwargs)
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
    print(PlanetoidDataset("Cora", mode="train").mask.sum())

    pyGeometricDataset("Photo", Amazon, mode="train")
    print(Amazon_sampler['val_mask'].sum(), Amazon_sampler['test_mask'].sum())
    pyGeometricDataset("Photo", Amazon, mode="test")
    print(Amazon_sampler['val_mask'].sum(), Amazon_sampler['test_mask'].sum())

    pyGeometricDataset("CS", Coauthor, mode="train")
    print(Coauthor_sampler['val_mask'].sum(), Coauthor_sampler['test_mask'].sum())

    print(TadpoleDataset(full=True).num_classes)
    a = TadpoleDataset(full=True)
    b = pyGeometricDataset("Photo", Amazon, mode="train")
    print(a.y.dtype, b.y.dtype)






