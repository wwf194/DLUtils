import torch
import torch.nn.functional as F
import numpy as np

import DLUtils
class Dropout(DLUtils.module.AbstractNetwork):
    def __init__(self, Rate=None):
        super().__init__()
        if Rate is not None:
            Param.Rate = Rate
    def ReceiveTrain(self, In):
        # X: [BatchSize, FeatureNum]
        In = F.dropout(In, self.Rate)
        return In
    def ReceiveTest(In):
        return In
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not hasattr(Param.Rate):
            self.Log("rate is not set. set to default: 0.1.")
            Param.setattr("Rate", 0.1)
        self.Rate = Param.Rate
        if self.hasattr("IsTrain"):
            IsTrain = self.IsTrain
        else:
            IsTrain = True
        if IsTrain:
            self.Receive = self.ReceiveTrain
        else:
            self.Receive = self.ReceiveTest
        return super().Init(IsSuper=True, IsRoot=IsRoot)