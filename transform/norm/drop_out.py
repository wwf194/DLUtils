import torch
import torch.nn.functional as F
import numpy as np

import DLUtils
class Dropout(DLUtils.module.AbstractNetwork):
    def ReceiveTrain(self, In):
        # X: (BatchSize, FeatureNum)
        In = F.dropout(In, self.Rate, inplace=self.InPlace)
        return In
    def ReceiveTest(In):
        return In
    def SetTrain(self, Recur=True):
        self.Receive = self.ReceiveTrain
        return super().SetTrain(Recur)
    def SetTest(self, Recur=True):
        self.Receive = self.ReceiceTest
        return super().SetTest(Recur)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.setdefault("Rate", 0.1)
        Param.setdefault()

        return super().Init(IsSuper=True, IsRoot=IsRoot)