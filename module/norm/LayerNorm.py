import torch
import numpy as np

import DLUtils
import DLUtils.utils as utils
class LayerNorm(DLUtils.module.AbstractNetwork):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, FeatureNum=None, eps=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.norm.LayerNorm"
        if eps is not None:
            self.eps = eps
        self.SetFeatureNum(FeatureNum)
        self.SetEps(eps)
    def SetFeatureNum(self, FeatureNum):
        self.Param.FeatureNum = FeatureNum
        return self
    def SetEps(self, eps):
        self.Param.eps = eps
        return self
    def EnableAffineTransform(self, State):
        Param = self.Param
        if State is True:
            Param.ApplyTransform.Enable = True
        else:
            Param.ApplyAffineTransform = False
    def EnableTrainableAffineTransform(self, State):
        Param = self.Param
        if State:
            Param.AffineTransform.setdefault("Enable", True)
            Param.AffineTransform.Trainable = True
        else:
            if Param.AffineTransform.get("Enable") is True:
                Param.AffineTransform.Trainable = False
    def Init(self, **Dict):
        Param = self.Param
        if not hasattr(Param, "eps"):
            self.SetEps(1.0e-6)
        assert hasattr(Param, "FeatureNum")
        if Param.AffineTransform.get("Enable") is True:
            Param.Tensor = ["A", "B"]
            if Param.Data.get("A") is None:
                Param.Data.A = np.ones((Param.FeatureNum))
                self.Log("LayerNorm.Init: initing A in affine transform to all ones")
            if Param.Data.get("B") is None:
                Param.Data.B = np.zeros((Param.FeatureNum))
                self.Log("LayerNorm.Init: initing B in affine transform to all zeros")
            if Param.AffineTransform.setdefault("Trainable", True) is True:
                Param.TrainParam = ["A", "B"]
        else:
            self.A = 1.0
            self.B = 0.0
        self.eps = Param.Data.setdefault("eps", 1.0e-9)
    def Receive(self, X):
        # X: [BatchSize, FeatureNum]
        XMean = X.mean(1, keepdim=True)
        XStd  = X.std(1, keepdim=True)
        return self.A * (X - XMean) / (XStd + self.eps) + self.B