import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")

def NonLinearFunction(Type, *List, **Dict):
    if Type in NonLinearModuleMap:
        return NonLinearModuleMap[Type](*List, **Dict)
    else:
        raise Exception(Type)
NonLinearModule = NonLinearTransform = NonLinearFunction

class ReLU(DLUtils.module.AbstractNetwork):
    def __init__(self):
        super().__init__()
        Param = self.Param
        #Param.Type = "NonLinear.ReLU"
    def Receive(self, In):
        return torch.relu(In)

class Sigmoid(DLUtils.module.AbstractNetwork):
    def Receive(self, In):
        return torch.sigmoid(In)

class Linear(DLUtils.module.AbstractNetwork):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param.Type = "Linear"
    def Receive(self, In):
        return In

NonLinearModuleMap = DLUtils.IterableKeyToElement({
    ("ReLU", "relu"): ReLU,
    ("None", "none"): Linear,
    ("Sigmoid", "sigmoid"): Sigmoid
})