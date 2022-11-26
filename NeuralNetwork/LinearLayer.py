from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
from DLUtils.attr import *
from DLUtils.transform.SingleLayer import SingleLayer
from .AbstractModule import AbstractNetwork
class LinearLayer(AbstractNetwork):
    def __init__(self, InputNum=None, OutputNum=None):
        super().__init__()
        Param = self.Param
        Param.Tensors = ["Weight", "Bias"]
        Param._CLASS = "DLUtils.NN.LinearLayer"
        if InputNum is not None:
            Param.Input.Num = InputNum
        if OutputNum is not None:
            Param.Output.Num = OutputNum
    def Init(self):
        Param = self.Param
        assert Param.Data.HasAttr("Weight")
        if Param.Mode != "Wx":
            if not Param.Data.HasAttr("Bias"):
                Param.Data.Bias = 0.0
        if not hasattr(Param, "Mode"):
            self.SetMode("Wx + b")
        return self
    def SetMode(self, Mode):
        Param = self.Param
        if Mode in ["Wx"]:
            self.Receive = self.Receive0
        elif Mode in ["Wx+b"]:
            self.Receive = self.Receive1
        elif Mode in ["W(x+b)"]:
            self.Receive = self.Receive2
        else:
            raise Exception(Mode)
        Param.Mode = Mode
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
        Param = self.Param
        Bias = DLUtils.ToNpArrayOrNum(Bias)
        Param.Data.Bias = Bias
        return self