import inspect, sys

import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
import torchmetrics
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.utils.utils import *

from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar

tb_logger = pl_loggers.CSVLogger("logs", name="my_exp_name")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="max")

pynvml = True

def VRAM_usage():
    vram = [i / (1024 ** 3) for i in torch.cuda.mem_get_info()]
    return round(vram[-1] - vram[0], 2)

class _Model_(pl.LightningModule):
    def __init__(self, seed=0, **hparams):
        super(_Model_, self).__init__()
        set_seed(seed)
        _self_, _trainer_ = self.split_kwargs(**hparams)
        _self_.update({"seed": seed})
        apply_args(self, _self_)

        self.trainer = pl.Trainer(callbacks=[RichProgressBar(leave=True), early_stop_callback], logger=tb_logger, **_trainer_)
        self._setup_()
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=self.nc)
        self.save_hyperparameters(_self_)

    def split_kwargs(self, **kwargs):
        a_init_params = inspect.signature(pl.Trainer).parameters
        _self_ = {k: v for k, v in kwargs.items() if k not in a_init_params}
        _trainer_ = {k: v for k, v in kwargs.items() if k in a_init_params}
        _trainer_["max_epochs"] = _trainer_.get("max_epochs", 10)

        if _self_.get("device"):
            _trainer_["accelerator"] = _self_.pop("device")

        if _self_.get("device_num"):
            _trainer_["devices"] = _self_.pop("device_num")

        return _self_, _trainer_

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

