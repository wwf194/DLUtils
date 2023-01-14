import torch
import DLUtils

class CrossEntropy(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.train.loss.CrossEntropy"
        return
    def Receive(self, Output, OutputTarget):
        # Input: [BatchSize, OutuptNum, Probability]
        # Output: [BatchSize, OutNum, Probability]
        Loss = - OutputTarget * torch.log(Output) # [BatchSize, OutNum]
        Loss = torch.sum(Loss, dim=1) # [BatchSize]
        return torch.mean(Loss)

class SoftMax(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.module.loss.SoftMax"
    def Receive(self, In):
        # Input: [BatchNum, FeatureNum]
        return torch.softmax(Input, dim=1)

SoftMax1D = SoftMax

class SoftMaxAndCrossEntropy(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.loss.SoftMaxAndCrossEntropy"
    def Receive(self, Output=None, OutputTarget=None):
        OutputProb = self.SoftMax(Output)
        Loss = self.CrossEntropy(OutputProb, OutputTarget)
        return Loss
    def Init(self, IsSuper=False, IsRoot=True):
        self.AddSubModule(
            "CrossEntropy", CrossEntropy()
        )
        self.AddSubModule(
            "SoftMax", SoftMax()
        )
        super().Init(IsSuper=IsSuper, IsRoot=IsRoot)
        return self