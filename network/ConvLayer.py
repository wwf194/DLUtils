import torch
import torch.nn.functional as F
import numpy as np

import DLUtils
class ConvLayer2D(DLUtils.module.AbstractNetwork):
    def __init__(self, InputNum=None, OutputNum=None, KernelSize=None, **Dict):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.network.ConvLayer2D"
        self.SetParam(
            InputNum=InputNum, OutputNum=OutputNum,
            KernelSize=KernelSize, **Dict
        )
    def SetParam(self, **Dict):
        Param = self.Param
        InputNum = Dict.get("InputNum")
        if InputNum is not None:
            Param.Kernel.Input.Num = InputNum
        OutputNum = Dict.get("OutputNum")
        if OutputNum is not None:
            Param.Kernel.Output.Num = OutputNum
        KernelSize = Dict.get("KernelSize")
        if KernelSize is not None:
            Param.Kernel.Size = KernelSize
        Padding = Dict.get("Padding")
        if Padding is not None:
            Param.Conv.Padding = Padding
        Stride = Dict.get("Stride")
        if Stride is not None:
            Param.Conv.Stride = Stride
        GroupNum = Dict.get("GroupNum")
        if GroupNum is not None:
            Param.Conv.Group.Num = GroupNum
        return self
    def SetTrainParam(self, **Dict):
        Param = self.Param
        Kernel = Dict.get("Kernel")
        if Kernel is not None:
            Param.Data.Kernel = Kernel
            InputNum, OutputNum, Height, Width = Kernel.shape
            Param.Kernel.Input.Num = InputNum
            Param.Kernel.Output.Num = OutputNum
            Param.Kernel.Height = Height
            Param.Kernel.Width = Width
        Bias = Dict.get("Bias")
        if Bias is not None:
            self.AddTrainParam("Bias", Bias)
        return self
    def Receive(self, Input):
        Output = F.conv2d(
            input=Input, weight=self.Weight, bias=self.Bias,
            stride=self.Stride,
            padding=self.Padding,
            dilation=self.Dilation,
            groups=self.GroupNum
        )
        Output = self.NonLinear(Output)
        return Output
    def SetWeight(self, Weight):
        Param = self.Param
        self.AddTrainParam("Weight", Weight) # [In, Out, KernelWidth, KernelHeight]
        Param.Kernel.Input.Num = Weight.shape[1]
        Param.Output.Num = Weight.shape[0]
        Param.Kernel.Width = Weight.shape[2]
        Param.Kernel.Height = Weight.shape[3]
        return self
    def SetBias(self, Bias):
        Param = self.Param
        if isinstance(Bias, float):
            Param.Data.Bias = Bias
            return
        if isinstance(Bias, str):
            if Bias in ["zeros"]:
                Bias = np.zeros((Param.Output.Num))
            else:
                raise Exception()
        self.AddTrainParam("Bias", Bias)
        return self
    def SetNonLinear(self, NonLinearModule):
        if isinstance(NonLinearModule, str):
            NonLinearModule = DLUtils.transform.NonLinear(NonLinearModule)
        self.AddSubModule("NonLinear", NonLinearModule)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not Param.Data.hasattr("Bias"):
            Param.Data.Bias = 0.0
        self.Stride = Param.Conv.setdefault("Stride", 1)
        Padding = Param.Conv.setdefault("Param", "KeepInOutHW")
        if Padding in ["KeepInOutHW"]:
            self.Padding = "same"
        elif isinstance(Padding, int):
            self.Padding = Padding
        else:
            raise Exception()
        self.Dilation = Param.Conv.setdefault("Dilation", 1)
        self.GroupNum = Param.Conv.Group.setdefault("Num", 1)
        Param.Input.Num = Param.Kernel.Input.Num * Param.Conv.Group.Num
        Param.Kernel.Output.Num = Param.Output.Num // Param.Conv.Group.Num
        Param.NonLinear.setdefault("Enable", False)
        if Param.NonLinear.Enable:
            assert Param.SubModules.hasattr("NonLinear")
        else:
            self.NonLinear = lambda x:x
        super().Init(IsSuper=True, IsRoot=IsRoot)
        if hasattr(self, "Bias"):
            if isinstance(self.Bias, float):
                assert self.Bias == 0.0
                self.Bias = None
        else:
            self.Bias = None
        return self