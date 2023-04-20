import numpy as np
import torch
import DLUtils
class AddDimBeforeFirstDim(DLUtils.module.AbstractNetwork):
    def Receive(self, X):
        return torch.unsqueeze(X, 0)

class AddDimAfterLastDim(DLUtils.module.AbstractNetwork):
    def Receive(self, X):
        return torch.unsqueeze(X, -1)

class InsertDim(DLUtils.module.AbstractNetwork):
    SetParamMap = DLUtils.IterableKeyToElement({
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
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("InsertIndex")
        assert isinstance(Param.InsertIndex, int)
        self.DimenInsertIndexsionIndex = Param.InsertIndex
        return super().Init(IsSuper=True, IsRoot=IsRoot)