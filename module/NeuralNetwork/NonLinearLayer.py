from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
from DLUtils.transform.SingleLayer import SingleLayer
#from .AbstractModule import AbstractNetwork
from .LinearLayer import LinearLayer
class NonLinearLayer(LinearLayer):
    def __init__(self, InputNum=None, OutputNum=None):
        super().__init__(InputNum=InputNum, OutputNum=OutputNum)
        Param = self.Param
        Param._CLASS = "DLUtils.NN.NonLinearLayer"
    def SetMode(self, Mode):
        Param = self.Param
        Param.Mode = Mode
        self.SetReceiveMethod()
        return self
    def LoadParam(self, Param):
        super().LoadParam(Param)
        return self
    def SetReceiveMethod(self):
        Param = self.Param
        Mode = Param.Mode
        if Mode in ["f(Wx+b)"]:
            self.Receive = self.ReceiveFAddMulWxb
        elif Mode in ["f(W(x+b))"]:
            self.Receive = self.ReceiveFMulWAddxb
        elif Mode in ["f(Wx)+b"]:
            self.Receive = self.ReceiveAddFMulWxb
        else:
            raise Exception(Mode)
    def SetNonLinear(self, NonLinearModule):
        if isinstance(NonLinearModule, str):
            NonLinearModule = DLUtils.NN.NonLinear.BuildNonLinearModule(NonLinearModule)
        self.AddSubModule("NonLinear", NonLinearModule)
        return self
    def SetNonLinearMethod(self):
        self.NonLinear = self.SubModules["NonLinear"]
        return self
    def ReceiveFAddMulWxb(self, Input):
        return self.NonLinear(torch.mm(Input + self.Bias, self.Weight))
    def ReceiveAddFMulWxb(self, Input):
        return self.NonLinear(torch.mm(Input, self.Weight)) + self.Bias
    def ReceiveFMulWAddxb(self, Input):
        return self.NonLinear(torch.mm(Input, self.Weight)) + self.Bias
    # SetWeight(...) # inherit
    # SetBias(...) # inherit
    def Init(self, IsSuper=False):
        if not IsSuper:
            self.SetReceiveMethod()
            self.SetNonLinearMethod()
        super().Init(True)
        return self