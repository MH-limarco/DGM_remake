from src.engine.model import _Model_
from src.engine.datasets.dataset_control import pl_DataModule
from src.engine.model_parser import parser

import torch
from torch_geometric import nn as pyGnn

class DGM(_Model_):
    def __init__(self, yaml, dataset, full=False):
        super().__init__(yaml=yaml, dataset=dataset, full=True)

    def set_up(self):
        self.dataset = pl_DataModule(self.dataset)
        self.dataset.setup("test")

        self.model = parser(self.yaml, in_c=self.dataset.in_c, nc=self.dataset.nc)

        x = torch.rand((64, self.dataset.in_c))
        ed_index = self.dataset.eval_set.edge_index.size(1) if hasattr(self.dataset.eval_set, "edge_index") else 0
        edge_index = torch.randint(64, size=(2, ed_index))

        _test_ = pyGnn.summary(self.model, x, edge_index, max_depth=1 if self.full else 5)
        print(_test_)

