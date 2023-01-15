import DLUtils
import torch

def NonLinearModule(Str):
    if Str in ["relu", "ReLU"]:
        return ReLU()
    else:
        raise Exception()


class ReLU(DLUtils.module.AbstractNetwork):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.network.ReLU"
        Param.Type = "ReLU"
    def Receive(self, In):
        return torch.relu(In)