import pytorch_lightning as pl
import torchmetrics

from src.utils.utils import *

class _Model_(pl.LightningModule):
    def __init__(self, yaml, dataset, seed=0, **kwargs):
        super(_Model_, self).__init__()
        apply_args(self, kwargs)
        set_seed(seed)
        self._setup_()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.nc)
        self.trainer = pl.Trainer()

    def _setup_(self):
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

    def fit(self):
        self.trainer.fit(self, datamodule=self.DataModule)
