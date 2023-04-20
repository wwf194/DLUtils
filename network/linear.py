from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import DLUtils

class LinearLayer(DLUtils.module.AbstractNetwork):
    def __init__(self, InNum=None, OutNum=None, **Dict):
        if InNum is not None:
            Dict["InNum"] = InNum
        if OutNum is not None:
            Dict["OutNum"] = OutNum      
        super().__init__(**Dict)
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
    def SetWeight(self, Weight, Trainable=True):
        Weight = DLUtils.ToNpArrayOrNum(Weight)
        Param = self.Param
        Param.Weight.Value = Weight
        Param.In.Num = Weight.shape[0]
        Param.Out.Num = Weight.shape[1]
        assert len(Weight.shape) == 2
        # if Data.HasAttr("Bias") and Param.HasAttr("Mode"):
        #     if Param.Mode == "Wx+b":
        #         assert Param.Out.Num == Data.Weight.shape[1]
        #     else:
        #         assert Param.In.Num == Data.Weight.shape[0]
        if Trainable:
            self.RegisterTrainParam(Name="Weight", Path="Weight.Value")
        else:
            self.RegisterTensor(Name="Weight", Path="Weight.Value")
        return self
    def SetBias(self, Bias, Trainable=True):
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
        if Trainable:
            assert isinstance(Bias, np.ndarray)
            self.RegisterTrainParam(Name="Bias", Path="Bias.Value")
        else:
            if isinstance(Bias, np.ndarray):
                self.RegisterTensor(Name="Bias", Path="Bias.Value")
            else:
                assert isinstance(Bias, float)

        Param.Bias.Value = Bias
        return self
    def Init(self, IsSuper=True, **Dict):
        if IsSuper:
            return super().Init(IsSuper=True, **Dict)
        Param = self.Param

        if self.IsInit():
            Mode = Param.setdefault("Mode", "Wx+b")

            # weight setting
            if not Param.Weight.hasattr("Value"):
                self.SetWeightDefault()

            # bias setting
            if Mode in ["Wx"]:
                Param.Bias.Enable = False
            else:
                Param.Bias.Enable = True
            if not Param.Bias.Enable:
                pass
            else:
                # bias setting
                if not Param.Bias.hasattr("Value"):
                    # set default bias
                    if Mode in ["Wx+b"]:
                        self.SetTrainParam("Bias", "Bias.Value", 
                            DLUtils.DefaultNonLinearLayerBias(Param.Out.Num)
                        )
                    elif Mode in ["W(x+b)"]:
                        self.SetTrainParam("Bias", "Bias.Value", 
                            DLUtils.DefaultNonLinearLayerBias(Param.In.Num)
                        )
                    else:
                        raise Exception()
        else:
            assert Param.hasattr("Mode")
            assert Param.Weight.hasattr("Value")
            assert Param.Bias.hasattr("Enable")
            if Param.Bias.Enable:
                assert Param.Bias.hasattr("Value")

        if Param.Bias.Enable:
            self.Bias = Param.Bias.Value
        else:
            self.Bias = 0.0

        self.SetReceiveMethod()
        super().Init(IsSuper=True, **Dict)
        return self
    def SetWeightDefault(self):
        Param = self.Param
        self.SetTrainParam(
            "Weight", "Weight.Value",
            DLUtils.DefaultLinearLayerWeight((Param.In.Num, Param.Out.Num))
        )
        return self

Linear = LinearLayer