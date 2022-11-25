import torch
import numpy as np

import DLUtils
import DLUtils.utils as utils
class LayerNorm(DLUtils.NeuralNetwork.AbstractModule):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, FeatureNum=None, eps=None):
        super().__init__()

        Param = self.Param
        Param.Class = "DLUtils.NN.LayerNorm"
        Param.Tensors = ["A", "B"]
        self.eps = eps
        self.SetFeatureNum(FeatureNum)
        self.SetEps(eps)
    def SetFeatureNum(self, FeatureNum):
        self.Param.FeatureNum = FeatureNum
        return self
    def SetEps(self, eps):
        self.Param.eps = eps
        return self
    def Init(self):
        Param = self.Param
        if not hasattr(Param, "eps"):
            self.SetEps(1.0e-6)
        assert hasattr(Param, "FeatureNum")
        Param.Data.A = np.ones((Param.FeatureNum))
        Param.Data.B = np.ones((Param.FeatureNum))
        self.Tensors = ["A", "B"]
    # def Build(self):
    #     self.Tensors.A = DLUtils.utils.ToTrainableTorchTensor(self.A)
    #     self.Tensors.B = DLUtils.utils.ToTrainableTorchTensor(self.B)
    def forward(self, X):
        # X: [BatchSize, FeatureNum]
        XMean = X.mean(1, keepdim=True)
        XStd  = X.std(1, keepdim=True)
        return self.a * (X - XMean) / (XStd + self.eps) + self.B
    def ToDict(self):
        self.Dict = self.Param.ToDict()
        {
            "Type": "LayerNorm",
            "Data":{
                "A": DLUtils.ToNpArray(self.Tensors.A),
                "B": DLUtils.ToNpArray(self.Tensors.B),
                "eps": self.eps
            }
        }
        return super().ToDict()
    def FromDict(self, Dict):
        DLUtils.NewObj(Dict)
        self.Dict = Dict
        self.Tensors.A = DLUtils.ToTrainableTorchTensor(Dict.Data.A)
        self.Tensors.B = DLUtils.ToTrainableTorchTensor(Dict.Data.B)
        self.eps = Dict.Data.eps
        return
