from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import DLUtils

class LinearLayer(DLUtils.module.AbstractNetwork):
    def __init__(self, InNum=None, OutNum=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.network.LinearLayer"
        if InNum is not None:
            Param.In.Num = InNum
        if OutNum is not None:
            Param.Out.Num = OutNum
    def LoadParam(self, Param):
        super().LoadParam(Param)
        return self
    def SetMode(self, Mode):
        Param = self.Param
        Param.Mode = Mode
        self.SetReceiveMethod()
        return self
    def SetReceiveMethod(self):
        Param = self.Param
        Mode = Param.Mode
        if Mode in ["Wx"]:
            self.Receive = self.ReceiveMulWx
        elif Mode in ["Wx+b"]:
            self.Receive = self.ReceiveAddMulWxb
        elif Mode in ["W(x+b)"]:
            self.Receive = self.ReceiveMulWAddxb
        else:
            raise Exception(Mode)
    def ReceiveMulWx(self, In): #Wx
        return torch.mm(In, self.Weight)
    def ReceiveAddMulWxb(self, In): # Wx+b
        return torch.mm(In, self.Weight) + self.Bias
    def ReceiveMulWAddxb(self, In): # W(x+b)
        return torch.mm(In + self.Bias, self.Weight)
    def SetWeight(self, Weight, Train=True):
        Weight = DLUtils.ToNpArrayOrNum(Weight)
        Param = self.Param
        Data = self.Param.Data
        Data.Weight = Weight
        assert len(Weight.shape) == 2
        Param.In.Num = Weight.shape[0]
        Param.Out.Num = Weight.shape[1]
        if Data.HasAttr("Bias") and Param.HasAttr("Mode"):
            if Param.Mode == "Wx+b":
                assert Param.Out.Num == Data.Weight.shape[1]
            else:
                assert Param.In.Num == Data.Weight.shape[0]
        Param.Tensor.add("Weight")
        if Train:
            Param.TrainParam.add("Weight")
        else:
            Param.Tensor.discard("Weight")
        return self
    def SetBias(self, Bias, Train=True):
        Param = self.Param
        if isinstance(Bias, str):
            assert Param.hasattr("Mode")
            if Param.Mode in ["Wx+b"]:
                Num = Param.Out.Num
            elif Param.Mode in ["W(x+b)"]:
                Num = Param.In.Num
            else:
                raise Exception()

            if Bias in ["zeros"]:
                Bias = DLUtils.SampleFromConstantDistribution((Num), 0.0)
            elif Bias in ["ones"]:
                Bias = DLUtils.SampleFromConstantDistribution((Num), 1.0)
            else:
                raise Exception()
        else:
            Bias = DLUtils.ToNpArrayOrNum(Bias)
        
        if isinstance(Bias, np.ndarray):
            Param.Tensor.add("Bias")
            if Train:
                assert isinstance(Bias, np.ndarray)
                Param.TrainParam.add("Bias")
        # else:
        #     Param.Tensor.discard("Bias")
        Param.Data.Bias = Bias
        return self
    def Init(self, IsSuper=True, **Dict):
        Param = self.Param
        assert Param.hasattr("Mode")
        if not IsSuper:
            if not Param.Data.hasattr("Weight"):
                self.SetWeightDefault()
            self.SetReceiveMethod()

        if Param.Mode != "Wx":
            if not Param.Data.hasattr("Bias"):
                Param.Data.Bias = 0.0
        if not hasattr(Param, "Mode"):
            self.SetMode("Wx + b")
        if "Bias" not in Param.Tensor:
            self.Bias = Param.Data.Bias
        super().Init(IsSuper=True, **Dict)
        return self
    def SetWeightDefault(self):
        Param = self.Param
        self.SetTrainParam("Weight",
            DLUtils.SampleFromKaimingUniform(
                (Param.In.Num, Param.Out.Num),
                NonLinear=None,
            )
        )
        return self