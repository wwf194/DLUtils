import DLUtils
from ..AbstractModule import AbstractNetwork
import torch

def NonLinearModule(Str):
    if Str in ["relu", "ReLU"]:
        return ReLU()
    else:
        raise Exception()

class ReLU(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.Param.absorb_dict({
            "_CLASS": "DLUtils.NN.ReLU"
        })
    def Receive(self, Input):
        return torch.relu(Input)