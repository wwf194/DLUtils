import torch
import numpy as np

import DLUtils
class LayerNorm(DLUtils.module.AbstractNetwork):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, FeatureNum=None, eps=None):
        super().__init__()
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
        Param.Affine.Enable = True
        # if State is True:
        #     Param.ApplyTransform.Enable = True
        # else:
        #     Param.ApplyAffineTransform = False
        return self
    def EnableTrainableAffineTransform(self, State):
        Param = self.Param
        if State:
            Param.Affine.setdefault("Enable", True)
            Param.Affine.Trainable = True
        else:
            if Param.Affine.get("Enable") is True:
                Param.Affine.Trainable = False
    def Receive(self, X):
        # X: [BatchSize, FeatureNum]
        XMean = X.mean(dim=1, keepdim=True)
        XStd  = X.std(dim=1, keepdim=True)
        return self.A * (X - XMean) / (XStd + self.eps) + self.B
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.Feature.hasattr("Num")
        
        # set affine param
        if self.IsInit():
            Param.Affine.setdefault("Enable", True)
            if Param.Affine.Enable:
                if not Param.Affine.Scale.hasattr("Value"):
                    Param.Affine.Scale.Value = DLUtils.ShapeWithSameValue(Param.Unit.Num, 1.0)
                if not Param.Affine.Bias.hasattr("Value"):
                    Param.Affine.Bias.Value = DLUtils.ShapeWithSameValue(Param.Unit.Num, 0.0)
                self.SetTensor()
                
                Param.Affine.setdefault("Trainable", True)
                if not Param.Data.hasattr("A"):
                    A = np.ones(Param.Feature.Num)
                    self.SetTrainParam(A=A)
                    self.Log("initing A in affine transform to all ones")
                if not Param.Data.hasattr("B") is None:
                    B = np.zeros((Param.Feature.Num))
                    self.SetTrainParam(B=B)
                    self.Log("initing B in affine transform to all zeros")
        if Param.Affine.Enable:
            self.Scale = Param.Affine.Scale.Value
            self.Bias = Param.Affine.Bias.Value
        else:
            self.Scale = 0.0
            self.Bias = 0.0

        # set eps
        if self.IsInit():
            Param.Eps.setdefault("Enable", True)
            if Param.Eps.Enable:
                Param.Eps.setdefault("Value", 1.0e-09)
        
        if Param.Eps.Enable:
            self.eps = Param.Eps.Value
        else:
            self.eps = 0.0
        return super().Init(IsSuper=True, IsRoot=IsRoot)