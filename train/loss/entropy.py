import torch
import DLUtils

class CrossEntropy(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        return
    def Receive(self, Input, Output):
        # Input: [BatchSize, OutuptNum, Probability]
        # Output: [BatchSize, OutputNum, Probability]
        return Output * torch.log(Input)