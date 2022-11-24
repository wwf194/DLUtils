from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
from DLUtils.attr import *
from DLUtils.transform.SingleLayer import SingleLayer
class LinearLayer(DLUtils.NeuralNetwork.AbstractModule):
    def __init__(self, **kw):
        super().__init__(**kw)
    def ToDict():
        super().ToDict()
    def Init(self):
        Param = self.Param
        assert Param.Data.HasAttr("Weight")
        if Param.Mode != "Wx":
            if not Param.Data.HasAttr("Bias"):
                Param.Data.Bias = 0.0
        
        self.Weight = DLUtils.ToTorchTensor(self.Param.Weight)
        self.Bias = DLUtils.ToTorchTensor(self.Param.Bias)
    def SetMode(self, Mode):
        if Mode in ["Wx"]:
            self.Receive = self.Receive0
        elif Mode in ["Wx+b"]:
            self.Receive = self.Receive1
        elif Mode in ["W(x+b)"]:
            self.Receive = self.Receive2
        else:
            raise Exception(Mode)
        self.Param.Mode = Mode
        return self
    def Receive0(self, Input): #Wx
        return torch.mm(Input, self.Weight)
    def Receive1(self, Input): # Wx+b
        return torch.mm(Input, self.Weight) + self.Bias
    def Receive2(self, Input): # W(x+b)
        return torch.mm(Input + self.Bias, self.Weight)
    def SetWeight(self, Weight):
        Weight = DLUtils.ToNpArrayOrNum(Weight)
        Param = self.Param
        Data = self.Param.Data
        Data.Weight = Weight
        assert len(Weight.shape) == 2
        Param.Input.Num = Weight.shape[0]
        Param.Output.Num = Weight.shape[1]
        if Data.HasAttr("Bias") and Param.HasAttr("Mode"):
            if Param.Mode == "Wx+b":
                assert Param.Output.Num == Data.Weight.shape[1]
            else:
                assert Param.Input.Num == Data.Weight.shape[0]
        return self
    def SetBias(self, Bias):
        Bias = DLUtils.ToNpArrayOrNum(Bias)
        self.Data.Bias = Bias
        return self

__MainClass__ = LinearLayer
# DLUtils.transform.SetMethodForTransformModule(__MainClass__)