import copy

from src.engine.model import _Model_
from src.engine.datasets.dataset_control import pl_DataModule
from src.engine.model_parser import parser
from src.utils.utils import args2dict

import torch
import torch.nn.functional as F
from torch_geometric import nn as pyGnn

class DGM(_Model_):
    def __init__(self, yaml, dataset, full=False, **kwargs):
        super(DGM, self).__init__(yaml=yaml, dataset=dataset, full=full, **kwargs)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def _setup_(self):
        self.DataModule = pl_DataModule(self.dataset)

        _dataset_ = copy.deepcopy(self.DataModule)
        _dataset_.setup("test")
        self.nc = _dataset_.nc
        self.model = parser(self.yaml, in_c=_dataset_.in_c, nc=self.nc)

        x = torch.rand((64, _dataset_.in_c))
        ed_index = _dataset_.eval_set.edge_index.size(1) if hasattr(_dataset_.eval_set, "edge_index") else 0
        edge_index = torch.randint(64, size=(2, ed_index)) if ed_index > 0 else None

        _test_ = pyGnn.summary(self.model, x, edge_index, max_depth=1 if not self.full else 2)
        print(_test_)
        del _dataset_

    def forward(self, x, edges):
        return self.model(x, edges)

    def pred_meter(self, pred, y, mask):
        pred = pred.unsqueeze(0) if len(pred.shape) == 2 else pred
        pred, y = pred[:, mask], y[:, mask]
        return self.loss(pred.permute(0, 2, 1), y), self.accuracy(pred.squeeze(0), y.squeeze(0))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def loss(self, pred, y):
        loss = self.loss_function(pred, y)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, y, mask, edges = train_batch

        mask = mask[0]
        edges = edges[0] if edges is not None else edges

        pred = self(x, edges)

        loss, acc = self.pred_meter(pred, y, mask)
        return loss

    def validation_step(self, train_batch, batch_idx):
        x, y, mask, edges = train_batch

        mask = mask[0]
        edges = edges[0] if edges is not None else edges

        pred = self(x, edges)

        loss, acc = self.pred_meter(pred, y, mask)
        self.log_dict({"val_loss":loss, "val_acc":acc})
        #return {"loss"}

    def test_step(self, train_batch, batch_idx):
        x, y, mask, edges = train_batch

        mask = mask[0]
        edges = edges[0] if edges is not None else edges

        pred = self(x, edges)

        loss, acc = self.pred_meter(pred, y, mask)

        #return loss



