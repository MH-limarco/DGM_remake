import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.engine.datasets.dataset import *

__all__ = ["pl_DataModule"]

api_dict = {"tadpole"   :[TadpoleDataset,
                          {"name": "tadpole_data.pickle", "fold": 0, "full": False}],

            "cora"      :[PlanetoidDataset,
                          {"name": "Cora", "split": "complete", "normalize": True, "transform": None}],
            "citeseer"  :[PlanetoidDataset,
                          {"name": "CiteSeer", "split": "complete", "normalize": True, "transform": None}],
            "pubmed"    :[PlanetoidDataset,
                          {"name": "PubMed", "split": "complete", "normalize": True, "transform": None}],

            "photo"     :[pyGeometricDataset,
                          {"name": "Photo", "data_func": Amazon, "normalize": True, "transform": None}],
            "computers" :[pyGeometricDataset,
                          {"name": "Computers", "data_func": Amazon, "normalize": True, "transform": None}],
            "cs"        :[pyGeometricDataset,
                          {"name": "CS", "data_func": Coauthor, "normalize": True, "transform": None}],
            "physics"   :[pyGeometricDataset,
                          {"name": "Physics", "data_func": Coauthor, "normalize": True, "transform": None}]
            }

class pl_DataModule(pl.LightningDataModule):
    def __init__(self, data_name, batch_size=1, num_workers=4):
        self.data_name = data_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        self._load_data(stage)

    def _load_data(self, stage):
        _dataset_, _kwargs_ = api_dict[self.data_name]
        kwargs = _kwargs_.copy()

        if stage == "fit":
            kwargs.update({"mode": "train", "samples_per_epoch": 100})
            self.train_set = _dataset_(parent=self, **kwargs)

            kwargs.update({"mode": "val", "samples_per_epoch": 1})
            self.val_set = _dataset_(parent=self, **kwargs)

            if not hasattr(self, "in_c"):
                self.in_c = self.train_set.n_features

            if not hasattr(self, "nc"):
                self.nc = self.train_set.num_classes

        elif stage in ["test", "predict"]:
            kwargs.update({"mode": "test", "samples_per_epoch": 1})
            self.eval_set = _dataset_(parent=self, **kwargs)

            if not hasattr(self, "in_c"):
                self.in_c = self.eval_set.n_features

            if not hasattr(self, "nc"):
                self.nc = self.eval_set.num_classes
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.eval_set, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == "__main__":
    _test_ = pl_DataModule("tadpole")
    _test_.setup("test")
    _test_.setup("fit")
    print(_test_.in_c, _test_.nc)
