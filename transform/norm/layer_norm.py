import torch
import numpy as np
import DLUtils
class LayerNorm(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("NormShape"): "Norm.Shape",
        ("Eps"): "Eps.Data",
        ("Affine"): "Affine.Enable"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
        
        a = 1
    def EnableTrainableAffineTransform(self, State):
        Param = self.Param
        if State:
            Param.Affine.setdefault("Enable", True)
            Param.Affine.Trainable = True
        else:
            if Param.Affine.get("Enable") is True:
                Param.Affine.Trainable = False
    def Receive(self, X):
        # X: (BatchSize, FeatureNum)
        XMean = torch.mean(X, dim=self.NormDimension, keepdim=True)
        XStd  = torch.std(X, dim=self.NormDimension, keepdim=True)
        return self.Scale * (X - XMean) / (XStd + self.Eps) + self.Bias
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.Norm.hasattr("Shape")
        if isinstance(Param.Norm.Shape, int):
            Param.Norm.Shape = [Param.Norm.Shape]
        assert len(Param.Norm.Shape) > 0
        for Size in Param.Norm.Shape:
            assert isinstance(Size, int)
        self.NormShape = tuple(Param.Norm.Shape)
        self.NormDimension = [- (DimensionIndex + 1) for DimensionIndex in range(len(Param.Norm.Shape))]
        if self.IsInit():
            # set element-wise affine coefficient
            # these coefficients could be trainable.
            Param.Affine.setdefault("Enable", True)
            if Param.Affine.Enable:
                Trainable = Param.Affine.setdefault("Trainable", True)
                if not Param.Affine.hasattr("Scale.Data"):
                    self.SetTensor(
                        "Scale",
                        "Affine.Scale.Data",
                        np.ones(self.NormShape)
                    )
                    if Trainable:
                        self.RegisterTrainParam("Scale", "Affine.Scale.Data")
                if not Param.Affine.hasattr("Bias.Data") is None:
                    self.SetTensor(
                        "Bias",
                        "Affine.Bias.Data",
                        np.zeros(self.NormShape)
                    )
                    if Trainable:
                        self.RegisterTrainParam("Bias", "Affine.Bias.Data")

            # set eps
            if Param.Eps.hasattr("Data"):
                Param.Eps.Enable = True
            else:
                Param.Eps.setdefault("Enable", True)
                if Param.Eps.Enable:
                    Param.Eps.setdefault("Data", 1.0e-09)
        if not Param.Affine.Enable:
            self.Scale = 1.0
            self.Bias = 0.0

        if Param.Eps.Enable:
            self.Eps = Param.Eps.Data
        else:
            self.Eps = 0.0
        return super().Init(IsSuper=True, IsRoot=IsRoot)