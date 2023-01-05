import DLUtils

import torch
import torch.nn.functional as F
class AvgPool2D(DLUtils.module.AbstractNetwork):
    def __init__(self, KernelSize=None, **Dict):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.network.AvgPool2D"
        self.SetParam(KernelSize=KernelSize, **Dict)
    def Receive(self, Input):
        # X: [BatchSize, FeatureNum]
        Output = F.avg_pool2d(input=Input, kernel_size=self.KernelSize, padding=self.Padding, stride=self.Stride)
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
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Padding = Param.setdefault("Padding", 0)
        
        if Param.Kernel.hasattr("Size"):
            if Param.Kernel.hasattr("Width"):
                assert Param.Kernel.Width == Param.Kernel.Size
                Param.Kernel.delattr("Width")
            if Param.Kernel.hasattr("Height"):
                assert Param.Kernel.Height == Param.Kernel.Size
                Param.Kernel.delattr("Height")
            Param.Kernel.Shape = [Param.Kernel.Size, Param.Kernel.Size]
            self.KernelSize = Param.Kernel.Size
            self.Stride = Param.setdefault("Stride", self.KernelSize)
        elif Param.Kernel.hasattr("Width"):
            Param.Kernel.Shape = [Param.Kernel.Width, Param.Kernel.Height]
            self.KernelSize = (Param.Kernel.Widht, Param.Kernel.Height)
            self.Stride = Param.setdefault("Stride", self.KernelSize)
        else:
            raise Exception()

        
        return super().Init(IsSuper, IsRoot)