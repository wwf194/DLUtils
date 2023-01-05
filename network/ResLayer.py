import torch
import numpy as np

import DLUtils
class ResLayer(DLUtils.module.AbstractNetwork):
    def __init__(self, SubModule=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.network.ResLayer"
        if SubModule is not None:
            self.AddSubModule("SubModule", SubModule)
    def Receive(self, X):
        # X: [BatchSize, FeatureNum]
        Y = self.SubModule(X)
        Output = X + Y
        return Output
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("SubModule")
        return super().Init(IsSuper, IsRoot)