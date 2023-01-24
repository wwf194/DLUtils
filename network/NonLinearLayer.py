from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
# from DLUtils.transform.SingleLayer import SingleLayer
#from .AbstractModule import AbstractNetwork
from .LinearLayer import LinearLayer
class NonLinearLayer(LinearLayer):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("NonLinear"): "NonLinear.Type",
        ("InNum, inputNum, InputNum"): "In.Num",
        ("OutNum, outputNum, OutputNum"): "Out.Num"
    })
    def __init__(self, InNum=None, OutNum=None, NonLinear=None, **Dict):
        super().__init__(InNum=InNum, OutNum=OutNum, **Dict)
        if NonLinear is None:
            self.SetParam(NonLinear)
    def SetMode(self, Mode):
        Param = self.Param
        Param.Mode = Mode
        self.SetReceiveMethod()
        return self
    def LoadParam(self, Param):
        super().LoadParam(Param)
        return self
    def SetNonLinearMethod(self):
        self.NonLinear = self.SubModules.NonLinear
        return self
    def ReceiveFAddMulWxb(self, In):
        return self.NonLinear(torch.mm(In + self.Bias, self.Weight))
    def ReceiveAddFMulWxb(self, In):
        return self.NonLinear(torch.mm(In, self.Weight)) + self.Bias
    def ReceiveFMulWAddxb(self, In):
        return self.NonLinear(torch.mm(In, self.Weight)) + self.Bias
    ReceiveMethodMap = DLUtils.IterableKeyToElement({
        "f(Wx+b)": ReceiveFAddMulWxb,
        "f(W(x+b))": ReceiveFMulWAddxb,
        "f(Wx)+b": ReceiveAddFMulWxb
    })
    def SetReceiveMethod(self):
        Param = self.Param
        Param.setdefault("Mode", "f(Wx+b)")
        Mode = Param.Mode
        if Mode in self.ReceiveMethodMap:
            self.Receive = self.ReceiveMethodMap[Mode]
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
    # SetWeight(...) # inherit
    # SetBias(...) # inherit
    def Init(self, IsSuper=False, **Dict):
        Param = self.Param
        if not IsSuper:
            self.SetReceiveMethod()
        if self.IsInit():
            Param.NonLinear.setdefault("Type", "ReLU")
            self.LogWithSelfInfo(f"use default nonlinear type: {Param.NonLinear.Type}", "initialization")
            self.AddSubModule(
                "NonLinear", DLUtils.network.NonLinearModule(Param.NonLinear.Type)
            )
            if not Param.Data.hasattr("Weight"):
                self.SetWeightDefault()
        super().Init(IsSuper=True, **Dict)
        return self
    def SetWeightDefault(self):
        Param = self.Param
        self.SetTrainParam(
            "Weight",
            DLUtils.SampleFromKaimingUniform(
                (Param.In.Num, Param.Out.Num),
                NonLinear=Param.NonLinear.Type,
            )
        )