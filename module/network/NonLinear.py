

import DLUtils
from ..AbstractModule import AbstractNetwork
import torch
class ReLU(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.Param.absorb_dict({
            "_CLASS": "DLUtils.NN.ReLU"
        })
    def Receive(self, Input):
        return torch.relu(Input)

def BuildNonLinearModule(Str):
    if Str in ["relu", "ReLU"]:
        return ReLU()
    else:
        raise Exception()