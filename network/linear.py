from re import L
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import DLUtils

class LinearLayer(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.IterableKeyToElement({
        ("InNum, inputNum, InputNum", "InSize"): "In.Size",
        ("OutNum, outputNum, OutputNum", "OutSize"): "Out.Size",
        ("Bias"): "Bias.Enable"
    })
    def __init__(self, InSize=None, OutSize=None, **Dict):
        if InSize is not None:
            Dict["InSize"] = InSize
        if OutSize is not None:
            Dict["OutSize"] = OutSize      
        super().__init__(**Dict)
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
        return torch.matmul(In, self.Weight)
    def ReceiveAddMulWxb(self, In): # Wx+b
        return torch.matmul(In, self.Weight) + self.Bias
    def ReceiveMulWAddxb(self, In): # W(x+b)
        return torch.matmul(In + self.Bias, self.Weight)
    def SetWeight(self, Weight, Trainable=True):
        Weight = DLUtils.ToNpArrayOrNum(Weight)
        Param = self.Param
        Param.In.Size = Weight.shape[0]
        Param.Out.Size = Weight.shape[1]
        assert len(Weight.shape) == 2
        # if Data.HasAttr("Bias") and Param.HasAttr("Mode"):
        #     if Param.Mode == "Wx+b":
        #         assert Param.Out.Size == Data.Weight.shape[1]
        #     else:
        #         assert Param.In.Size == Data.Weight.shape[0]
        self.SetTensor(Name="Weight", Path="Weight.Data", Data=Weight)
        if Trainable:
            self.RegisterTrainParam(Name="Weight")
        return self
    def SetBias(self, Bias, Trainable=True):
        Param = self.Param
        if isinstance(Bias, str):
            assert Param.hasattr("Mode")
            if Param.Mode in ["Wx+b"]:
                Num = Param.Out.Size
            elif Param.Mode in ["W(x+b)"]:
                Num = Param.In.Size
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
            self.SetTensor(
                Name="Bias", Path="Bias.Data",
                Data=Bias
            )
            if Trainable:
                self.RegisterTrainParam(Name="Bias")
        else:
            assert isinstance(Bias, float)
        Param.Bias.Data = Bias
        return self
    def Init(self, IsSuper=True, **Dict):
        if IsSuper:
            return super().Init(IsSuper=True, **Dict)
        Param = self.Param
        if self.IsInit():
            Mode = Param.setdefault("Mode", "Wx+b")

            # weight setting
            if not Param.Weight.hasattr("Data"):
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
                if not Param.Bias.hasattr("Data"):
                    # set default bias
                    if Mode in ["Wx+b"]:
                        self.SetTrainParam(
                            "Bias",
                            "Bias.Data", 
                            DLUtils.DefaultLinearLayerBias(Shape=Param.Out.Size)
                        )
                    elif Mode in ["W(x+b)"]:
                        self.SetTrainParam(
                            "Bias",
                            "Bias.Data", 
                            DLUtils.DefaultLinearLayerBias(Shape=Param.In.Size)
                        )
                    else:
                        raise Exception()
        else:
            assert Param.hasattr("Mode")
            assert Param.Weight.hasattr("Data")
            assert Param.Bias.hasattr("Enable")
            if Param.Bias.Enable:
                assert Param.Bias.hasattr("Data")

        if not Param.Bias.Enable:
            self.Bias = 0.0

        self.SetReceiveMethod()
        super().Init(IsSuper=True, **Dict)
        return self
    def SetWeightDefault(self):
        Param = self.Param
        self.SetTrainParam(
            "Weight",
            "Weight.Data",
            DLUtils.DefaultLinearLayerWeight((Param.In.Size, Param.Out.Size))
        )
        return self

Linear = LinearLayer