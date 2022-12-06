import torch
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
from DLUtils.attr import *

from DLUtils.transform import AbstractTransformWithTensor
class Bias(AbstractTransformWithTensor):
    # def __init__(self, param=None, data=None, **kw):
    #     super(Bias, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="DLUtils.transform.Bias", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        if cache.IsInit:
            data.Bias = torch.nn.Parameter(torch.zeros(param.Size))
        else:
            data.Bias = DLUtils.ToTorchTensor(data.Bias)
        cache.TrainableParam.append([data, "Bias", data.Bias])

        return self
    def forward(self):
        return self.data.Bias
    def __call__(self, *Args, **Kw):
        return self.forward(*Args, **Kw)

__MainClass__ = Bias
# DLUtils.transform.SetMethodForTransformModule(__MainClass__)