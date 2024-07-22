import pytorch_lightning as pl
from src.utils.utils import *

class _Model_(pl.LightningDataModule):
    def __init__(self, yaml, dataset, seed=0, **kwargs):
        super(_Model_, self).__init__()
        apply_args(self, kwargs)
        set_seed(seed)
        self.set_up()

    def set_up(self):
        raise NotImplementedError()

    def forward(self, x, edges=None):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, train_batch, batch_idx):
        raise NotImplementedError()
