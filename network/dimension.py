from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")

import DLUtils
class AddDimBeforeFirstDim(DLUtils.module.AbstractNetwork):
    def Receive(self, X):
        return torch.unsqueeze(X, 0)

class AddDimAfterLastDim(DLUtils.module.AbstractNetwork):
    def Receive(self, X):
        return torch.unsqueeze(X, -1)

class InsertDim(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("DimensionIndex", "InsertIndex", "Dim"): "InsertIndex"
    })
    def __init__(self, InsertIndex=None, **Dict):
        super().__init__(**Dict)
        if InsertIndex is not None:
            assert isinstance(InsertIndex, int)
        Param = self.Param
        Param.InsertIndex = InsertIndex
    def Receive(self, X):
        return torch.unsqueeze(X, self.InsertIndex)
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("InsertIndex")
        assert isinstance(Param.InsertIndex, int)
        self.DimenInsertIndexsionIndex = Param.InsertIndex
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    