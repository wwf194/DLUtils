from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
# from DLUtils.transform.SingleLayer import SingleLayer
#from .AbstractModule import AbstractNetwork
from .LinearLayer import LinearLayer
class NonLinearLayer(LinearLayer):
    def __init__(self, InNum=None, OutNum=None):
        super().__init__(InNum=InNum, OutNum=OutNum)
        Param = self.Param
        Param._CLASS = "DLUtils.network.NonLinearLayer"
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
        self.LogWithSelfInfo(f"set mode as {Mode}", "initialization")
        return self
    def SetNonLinear(self, NonLinearModule):
        if isinstance(NonLinearModule, str):
            NonLinearModule = DLUtils.network.NonLinear.NonLinearModule(NonLinearModule)
        self.AddSubModule("NonLinear", NonLinearModule)
        self.LogWithSelfInfo(f"set nonlinear type: {NonLinearModule.ClassStr()}", "initialization")
        self.SetNonLinearMethod()
        return self
    def SetNonLinearMethod(self):
        self.NonLinear = self.SubModules.NonLinear
        return self
    def ReceiveFAddMulWxb(self, In):
        return self.NonLinear(torch.mm(Input + self.Bias, self.Weight))
    def ReceiveAddFMulWxb(self, In):
        return self.NonLinear(torch.mm(Input, self.Weight)) + self.Bias
    def ReceiveFMulWAddxb(self, In):
        return self.NonLinear(torch.mm(Input, self.Weight)) + self.Bias
    # SetWeight(...) # inherit
    # SetBias(...) # inherit
    def Init(self, IsSuper=False, **Dict):
        if not IsSuper:
            self.SetReceiveMethod()
            self.SetNonLinearMethod()
        super().Init(IsSuper=True, **Dict)
        return self
    def ClassStr(self):
        return "NonLinearLayer"