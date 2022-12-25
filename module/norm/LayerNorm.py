import torch
import numpy as np

import DLUtils
class LayerNorm(DLUtils.module.AbstractNetwork):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, FeatureNum=None, eps=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.norm.LayerNorm"
        if eps is not None:
            self.SetEps(eps)
        self.SetFeatureNum(FeatureNum)
    def SetEps(self, eps):
        Param = self.Param
        Param.Data.eps = eps
    def SetFeatureNum(self, FeatureNum):
        self.Param.FeatureNum = FeatureNum
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
    def Receive(self, X):
        # X: [BatchSize, FeatureNum]
        XMean = X.mean(dim=1, keepdim=True)
        XStd  = X.std(dim=1, keepdim=True)
        return self.A * (X - XMean) / (XStd + self.eps) + self.B
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert hasattr(Param, "FeatureNum")
        if Param.AffineTransform.setdefault("Enable", True):
            Param.Tensor.add("A")
            Param.Tensor.add("B")
            if not Param.Data.hasattr("A"):
                A = np.ones(Param.FeatureNum)
                self.AddTrainParam("A", A)
                self.Log("LayerNorm.Init: initing A in affine transform to all ones")
            if not Param.Data.hasattr("B") is None:
                B = np.zeros((Param.FeatureNum))
                self.AddTrainParam("B", B)
                self.Log("LayerNorm.Init: initing B in affine transform to all zeros")
        else:
            self.A = 1.0
            self.B = 0.0
        self.eps = Param.Data.setdefault("eps", 1.0e-9)
        return super().Init(IsSuper=True, IsRoot=IsRoot)