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
                assert isinstance(Value, int)
            elif Key in ["Out", "OutNum", "Out.Num", "OutputNum"]:
                Param.Out.Num = Value
                assert isinstance(Value, int)
            elif Key in ["KernelSize", "Kernel.Size"]:
                Param.Kernel.Size = Value
                assert isinstance(Value, int)
            elif Key in ["Padding"]:
                Param.Padding = Value
                assert isinstance(Value, int) or Value in ["KeepFeatureMapHeightWidth"]
            elif Key in ["Stride"]:
                Param.Stride = Value
                assert isinstance(Value, int)
            elif Key in ["GroupNum", "NumGroup", "Group.Num"]:
                Param.Group.Num = Value
                assert isinstance(Value, int)
            else:
                _Dict[Key] = Value
        assert len(_Dict) == 0 
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
                super().SetTrainParam(Bias=Value)
            else:
                raise Exception()
            return self
    def Receive(self, In):
        Out = F.conv2d(
            input=In, weight=self.Kernel, bias=self.Bias,
            stride=self.Stride,
            padding=self.Padding,
            dilation=self.Dilation,
            groups=self.GroupNum
        )
        Output = self.NonLinear(Out)
        return Output
    def SetKernel(self, Kernel):
        Param = self.Param
        self.SetTrainParam(Kernel=Kernel) # [OutNum, InNum // GroupNum, KernelWidth, KernelHeight]
        Param.Kernel.In.Num = Kernel.shape[1]
        Param.In.Num = Param.Kernel.In.Num * Param.Group.Num
        Param.Out.Num = Kernel.shape[0]
        Param.Kernel.Out.Num = Param.Out.Num // Param.Group.Num
        Param.Kernel.Height = Kernel.shape[2] # Height at dim 2
        Param.Kernel.Width = Kernel.shape[3] # Width at dim 3
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
                Param.setdefault("Stride", 1)
                Padding = Param.setdefault("Param", "KeepFeatureMapHeightWidth")

                Param.setdefault("Dilation", 1)
                Param.Group.setdefault("Num", 1)
                
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

                if not Param.Data.hasattr("Kernel"):
                    # [OutNum // GroupNum, InpNum, KernelHeight, KernelWidth]
                    self.SetKernel(
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
                    assert "Bias" in Param.Tensor
            if Param.Bias.Enable:
                pass
            else:
                self.Bias = 0.0
        
            Padding = Param.Padding
            if Padding in ["KeepFeatureMapHeightWidth"]:
                self.Padding = "same"
            elif isinstance(Padding, int):
                self.Padding = Padding
            else:
                raise Exception()
            self.Dilation = Param.Dilation
            self.GroupNum = Param.Group.Num
            self.Stride = Param.Stride

            if Param.NonLinear.Enable:
                assert Param.SubModules.hasattr("NonLinear")
            else:
                self.NonLinear = lambda x:x

        return super().Init(IsSuper=True, IsRoot=IsRoot)

class UpConv2D(Conv2D):
    def __init__(self, InNum=None, OutNum=None, **Dict):
        super().__init__(InNum, OutNum, **Dict)
    def Receive(self, In):
        """
        In: [BatchNum, ChannelNum, Height, Width]
        """
        Output = F.conv_transpose2d(
            input=In, weight=self.Kernel, bias=self.Bias,
            stride=self.Stride,
            padding=self.Padding,
            dilation=self.Dilation,
            groups=self.GroupNum
        )
        Output = self.NonLinear(Output)
        return Output
    def SetKernel(self, Kernel):
        Param = self.Param
        self.SetTrainParam(Kernel=Kernel) # [InNum, OutNum // GroupNum, KernelWidth, KernelHeight]
        Param.In.Num = Kernel.shape[0]
        Param.Kernel.In.Num = Param.In.Num // Param.Group.Num
        Param.Kernel.Out.Num = Kernel.shape[1]
        Param.Out.Num = Param.Kernel.Out.Num * Param.Group.Num
        Param.Kernel.Height = Kernel.shape[2] # Height at dim 2
        Param.Kernel.Width = Kernel.shape[3] # Width at dim 3
    def Init(self, IsSuper=False, IsRoot=True):
        if not IsSuper:
            Param = self.Param
            if not self.IsLoad():
                Param.setdefault("Stride", 1)
                Param.setdefault("Padding", 0)

                Param.setdefault("Dilation", 1)
                Param.Group.setdefault("Num", 1)
                
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



                if not Param.Data.hasattr("Weight"):
                    # [OutNum // GroupNum, InpNum, KernelHeight, KernelWidth]
                    self.SetKernel(
                        DLUtils.UpConv2DKernel(
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
                        # Param.Tensor.add("Bias")
                    assert "Bias" in Param.Tensor
            if Param.Bias.Enable:
                pass
            else:
                self.Bias = 0.0

            Padding = Param.Padding
            if isinstance(Padding, int):
                self.Padding = Padding
            else:
                raise Exception()
            self.Dilation = Param.Dilation
            self.GroupNum = Param.Group.Num
            self.Stride = Param.Stride
            if Param.NonLinear.Enable:
                assert Param.SubModules.hasattr("NonLinear")
            else:
                self.NonLinear = lambda x:x
        return super().Init(IsSuper=True, IsRoot=IsRoot)
