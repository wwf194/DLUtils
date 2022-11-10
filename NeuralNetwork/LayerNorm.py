import torch
import numpy as np

import DLUtils
import DLUtils.utils as utils
class LayerNorm(DLUtils.NeuralNetwork.AbstractNetwork):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, FeatureNum, eps:float=1e-6):
        super().__init__()
        self.A = np.ones((FeatureNum))
        self.B = np.ones((FeatureNum))
        self.eps = eps
        self.Tensors = DLUtils.EmptyObj()
    def Build(self):
        self.Tensors.A = DLUtils.utils.ToTrainableTorchTensor(self.A)
        self.Tensors.B = DLUtils.utils.ToTrainableTorchTensor(self.B)
    def forward(self, X):
        # X: [BatchSize, FeatureNum]
        XMean = X.mean(1, keepdim=True)
        XStd  = X.std(1, keepdim=True)
        return self.a * (X - XMean) / (XStd + self.eps) + self.B
    def ToDict(self):
        return {
            "Name": self.Name,
            "Type": "LayerNorm",
            "Data":{
                "A": DLUtils.ToNpArray(self.Tensors.A),
                "B": DLUtils.ToNpArray(self.Tensors.B),
                "eps": self.eps
            }
        }
    def FromDict(self, Dict):
        DLUtils.NewObj(Dict)
        self.Dict = Dict
        self.Tensors.A = DLUtils.ToTrainableTorchTensor(Dict.Data.A)
        self.Tensors.B = DLUtils.ToTrainableTorchTensor(Dict.Data.B)
        self.eps = Dict.Data.eps
        return
