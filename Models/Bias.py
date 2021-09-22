import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

class Bias(nn.Module):
    def __init__(self, param=None):
        super(Bias, self).__init__()
        if param is not None:
            self.param = param
            self.cache = utils_torch.EmptyPyObj()
            self.data = utils_torch.EmptyPyObj()
    def InitFromParam(self):
        param = self.param
        data = self.data
        cache = self.cache
        data.Bias = torch.nn.Parameter(torch.zeros(param.Size))
        cache.ParamIndices.append([data, "Bias", data.Bias])
    def forward(self):
        return self.data.Bias
    def SetTensorLocation(self, Location):
        cache = self.cache
        cache.TensorLocation = Location
        for ParamIndex in cache.ParamIndices:
            setattr(ParamIndex[0], ParamIndex[1], ParamIndex[2].to(Location))
    def GetTensorLocation(self, Location):
        return self.cache.TensorLocation
__MainClass__ = Bias