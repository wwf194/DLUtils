import torch
import torch.nn.functional as F
import numpy as np

import DLUtils
class Conv2D(DLUtils.module.AbstractNetwork):
    def __init__(self, InNum=None, OutNum=None, **Dict):
        super().__init__()
        Param = self.Param
        if InNum is not None:
            Dict["InNum"] = InNum
        if OutNum is not None:
            Dict["OutNum"] = OutNum
        self.SetParam(**Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        _Dict = {}
        for Key, Value in Dict.items():    
            if Key in ["In", "InNum", "In.Num"]:
                Param.In.Num = Value
            elif Key in ["Out", "OutNum", "Out.Num", "OutputNum"]:
                Param.Out.Num = Value
            elif Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
            elif Key in ["Padding"]:
                Param.Padding = Value
            elif Key in ["Stride"]:
                Param.Stride = Value
            elif Key in ["GroupNum", "NumGroup", "Group.Num"]:
                Param.Group.Num = Value
            else:
                _Dict[Key] = Value
        super().SetParam(**_Dict)
        return self
    def SetTrainParam(self, **Dict):
        Param = self.Param
        for Name, Value in Dict.items():
            if Name in ["Kernel", "Weight"]:
                Param.Data.Kernel = Value
                InNum, OutNum, Height, Width = Value.shape
                Param.Kernel.In.Num = InNum
                Param.Kernel.Out.Num = OutNum
                Param.Kernel.Height = Height
                Param.Kernel.Width = Width
                Bias = Dict.get("Bias")
                super().SetTrainParam(Kernel=Value)
            elif Name in ["Bias", "bias"]:
                super().SetTrainParam(Bias=Bias)
            else:
                raise Exception()
            return self
    def Receive(self, In):
        Output = F.conv2d(
            input=In, weight=self.Weight, bias=self.Bias,
            stride=self.Stride,
            padding=self.Padding,
            dilation=self.Dilation,
            groups=self.GroupNum
        )
        Output = self.NonLinear(Output)
        return Output
    def SetWeight(self, Weight):
        Param = self.Param
        self.SetTrainParam(Weight=Weight) # [In, Out, KernelWidth, KernelHeight]
        Param.Kernel.In.Num = Weight.shape[1]
        Param.Out.Num = Weight.shape[0]
        Param.Kernel.Height = Weight.shape[2] # Height at dim 2
        Param.Kernel.Width = Weight.shape[3] # Width at dim 3
        return self
    def SetBias(self, Bias):
        Param = self.Param
        if isinstance(Bias, float):
            Param.Data.Bias = Bias
            return
        if isinstance(Bias, str):
            if Bias in ["zeros"]:
                Bias = np.zeros((Param.Out.Num))
            else:
                raise Exception()
        self.SetTrainParam(Bias=Bias)
        return self
    def SetNonLinear(self, NonLinearModule):
        if isinstance(NonLinearModule, str):
            NonLinearModule = DLUtils.transform.NonLinear(NonLinearModule)
        self.AddSubModule("NonLinear", NonLinearModule)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        if not IsSuper:
            Param = self.Param
            if not self.IsLoad():
                if not Param.Data.hasattr("Bias"):
                    Param.Data.Bias = 0.0
                self.Stride = Param.setdefault("Stride", 1)
                Padding = Param.setdefault("Param", "KeepFeatureMapHeightWidth")
                if Padding in ["KeepFeatureMapHeightWidth"]:
                    self.Padding = "same"
                elif isinstance(Padding, int):
                    self.Padding = Padding
                else:
                    raise Exception()
                self.Dilation = Param.setdefault("Dilation", 1)
                self.GroupNum = Param.Group.setdefault("Num", 1)
                
                if Param.Kernel.hasattr("Size"):
                    if not Param.Kernel.hasattr("Height"):
                        Param.Kernel.Height = Param.Kernel.Size
                    if not Param.Kernel.hasattr("Width"):
                        Param.Kernel.Width = Param.Kernel.Size
                else:
                    assert Param.Kernel.hasattr("Height")
                    assert Param.Kernel.hasattr("Width")
                    if Param.Kernel.Height == Param.Kernel.Width:
                        Param.Kernel.Size = Param.Kernel.Height

                # torch group convolution kernel: 
                #   [OutNum, InNum // GroupNum, KernelHeight, KernelWidth]
                #   In dimension during computation will be divided into groups.
                assert Param.In.Num % Param.Group.Num == 0
                assert Param.Out.Num % Param.Group.Num == 0
                Param.Kernel.Out.Num = Param.Out.Num // Param.Group.Num
                Param.Kernel.In.Num = Param.In.Num // Param.Group.Num
                Param.NonLinear.setdefault("Enable", False)

                if Param.NonLinear.Enable:
                    assert Param.SubModules.hasattr("NonLinear")
                else:
                    self.NonLinear = lambda x:x


                if not Param.Data.hasattr("Weight"):
                    # [OutNum // GroupNum, InpNum, KernelHeight, KernelWidth]
                    self.SetWeight(
                        DLUtils.Conv2DKernel(
                            (
                                Param.In.Num, Param.Out.Num, 
                                Param.Kernel.Height, Param.Kernel.Width
                            ),
                            GroupNum=Param.Group.Num,
                            # NonLinear="ReLU"
                            # default torch weight initialization does not consider NonLinear.
                        )
                    )
                Param.Bias.setdefault("Enable", True)
                if Param.Bias.Enable:
                    if not Param.Data.hasattr("Bias"):
                        self.SetTrainParam(Bias=torch.zeros(Param.Out.Num))
                    else:
                        assert "Bias" in self.TensorList
            if Param.Bias.Enable:
                pass
            else:
                self.Bias = 0.0
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class UpConv2D(Conv2D):
    def Receive(self, In):
        Output = F.conv_transpose2d(
            input=In, weight=self.Weight, bias=self.Bias,
            stride=self.Stride,
            padding=self.Padding,
            dilation=self.Dilation,
            groups=self.GroupNum
        )
        Output = self.NonLinear(Output)
        return Output
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not Param.Data.hasattr("Bias"):
            Param.Data.Bias = 0.0
        self.Stride = Param.setdefault("Stride", 1)
        Padding = Param.setdefault("Param", 1)
        if isinstance(Padding, int):
            self.Padding = Padding
        else:
            raise Exception()
        self.Dilation = Param.setdefault("Dilation", 1)
        self.GroupNum = Param.Group.setdefault("Num", 1)
        Param.In.Num = Param.Kernel.In.Num * Param.Group.Num
        Param.Kernel.Out.Num = Param.Out.Num // Param.Group.Num
        Param.NonLinear.setdefault("Enable", False)
        
        if Param.NonLinear.Enable:
            assert Param.SubModules.hasattr("NonLinear")
        else:
            self.NonLinear = lambda x:x
        # super().Init(IsSuper=True, IsRoot=IsRoot)
        if hasattr(self, "Bias"):
            if isinstance(self.Bias, float):
                assert self.Bias == 0.0
                self.Bias = None
        else:
            self.Bias = None
        return super().Init(IsSuper=True, IsRoot=IsRoot)
