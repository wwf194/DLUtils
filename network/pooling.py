import DLUtils

import torch
import torch.nn.functional as F

class MaxPool2D(DLUtils.module.AbstractNetwork):
    ParamMap = DLUtils.ExpandIterableKey({
        ("Kernel", "KernelSize"): "Kernel.Size",
        ("KernelHeight"): "Kernel.Height",
        ("KernelWidth"): "Kernel.Width"
    })
    def Receive(self, In):
        # X: [BatchSize, FeatureNum]
        Output = F.max_pool2d(
            input=In, 
            kernel_size=self.KernelSize, 
            padding=self.Padding, 
            stride=self.Stride
        )
        return Output
    # def SetParam(self, **Dict):
    #     Param = self.Param
    #     KernelSize = Dict.pop("KernelSize", None)
    #     if KernelSize is not None:
    #         Param.Kernel.Size = KernelSize
    #     KernelWidth = Dict.pop("KernelWidth", None)
    #     if KernelWidth is not None:
    #         Param.Kernel.Width = KernelWidth
    #     KernelHeight = Dict.pop("KernelHeight", None)
    #     if KernelHeight is not None:
    #         Param.Kernel.Height = KernelHeight
    #     return super().SetParam(**Dict)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if self.IsInit():
            self.Padding = Param.setdefault("Padding", 0)
            if Param.Kernel.hasattr("Size"):
                Param.Kernel.Width = Param.Kernel.Size
                Param.Kernel.Height = Param.Kernel.Height
                Param.setdefault("Stride", self.KernelSize)
            elif Param.Kernel.hasattr("Width"):
                assert Param.Kernel.hasattr("Height")
                Param.setdefault("Stride", self.KernelSize)
            else:
                raise Exception()
        
        self.KernelSize = (Param.Kernel.Width, Param.Kernel.Height)
        assert Param.hasattr("Stride")
        self.Stride = Param.Stride
        return super().Init(IsSuper=True, IsRoot=False)

class AvgPool2D(MaxPool2D):
    def __init__(self, **Dict):
        super().__init__()
        self.SetParam(**Dict)
    def Receive(self, In):
        # X: [BatchSize, FeatureNum]
        Output = F.avg_pool2d(input=In, kernel_size=self.KernelSize, padding=self.Padding, stride=self.Stride)
        return Output
    def SetParam(self, **Dict):
        Param = self.Param
        KernelSize = Dict.pop("KernelSize", None)
        if KernelSize is not None:
            Param.Kernel.Size = KernelSize
        KernelWidth = Dict.pop("KernelWidth", None)
        if KernelWidth is not None:
            Param.Kernel.Width = KernelWidth
        KernelHeight = Dict.pop("KernelHeight", None)
        if KernelHeight is not None:
            Param.Kernel.Height = KernelHeight
        return super().SetParam(**Dict)
