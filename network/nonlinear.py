from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
# from DLUtils.transform.SingleLayer import SingleLayer
#from .AbstractModule import AbstractNetwork
from .linear import LinearLayer

class NonLinearLayer(LinearLayer):
    ParamMap = DLUtils.IterableKeyToElement({
        ("NonLinear"): "NonLinear.Type",
        ("InNum, inputNum, InputNum", "InSize"): "In.Size",
        ("OutNum, outputNum, OutputNum", "OutSize"): "Out.Size",
        ("Bias"): "Bias.Enable"
    })
    def __init__(self, **Dict):
        super().__init__(**Dict)
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
        return self.NonLinear(torch.matmul(In, self.Weight) + self.Bias)
    def ReceiveAddFMulWxb(self, In):
        return self.NonLinear(torch.matmul(In, self.Weight)) + self.Bias
    def ReceiveFMulWAddxb(self, In):
        return self.NonLinear(torch.matmul(In, self.Weight)) + self.Bias
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
        assert Param.hasattrs("In.Num", "Out.Num")
        
        if not IsSuper:
            self.SetReceiveMethod()
            if self.IsInit():
                Param.NonLinear.setdefault("Type", "ReLU")
                self.LogWithSelfInfo(f"use default nonlinear type: {Param.NonLinear.Type}", "initialization")
                self.AddSubModule(
                    "NonLinear", DLUtils.network.NonLinearModule(Param.NonLinear.Type)
                )

                # weight setting
                if not Param.Weight.hasattr("Data"):
                    self.SetDefaultWeight()
                
                # bias setting
                Param.Bias.setdefault("Enable", True)
                if Param.Bias.Enable:
                    Param.Bias.setdefault("Trainable", True)
                    if self.IsInit(): 
                        if not Param.Bias.hasattr("Data"):
                            self.SetDefaultBias()
                    else:
                        assert Param.Bias.hasattr("Data")
    
            else:
                assert Param.Weight.hasattr("Data")
        super().Init(IsSuper=True, **Dict)
        return self
    def SetDefaultWeight(self):
        Param = self.Param
        self.SetTrainParam(
            Name="Weight",
            Path="Weight.Data",
            Data=DLUtils.DefaultNonLinearLayerWeight(
                Shape=(Param.In.Num, Param.Out.Num),
                NonLinear=Param.NonLinear.Type,
            )
        )
        return self
    def SetDefaultBias(self):
        Param = self.Param
        if Param.Mode in ["f(Wx+b)", "f(Wx)+b"]:
            UnitNum = Param.Out.Num
        elif Param.Mode in ["f(W(x+b))"]:
            UnitNum = Param.In.Num
        else:
            raise Exception()
        self.SetTrainParam(
            Name="Bias",
            Path="Bias.Data",
            Data=DLUtils.DefaultNonLinearLayerBias(UnitNum)
        )
        if Param.Bias.Trainable:
            self.SetTrainable("Bias")
        return self
NonLinear = NonLinearLayer

