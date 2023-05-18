from .reshape import Reshape, Index2OneHot, ChangeDimOrder, Permute
from .norm import LayerNorm, BatchNorm2D
from .norm import ShiftRange 
from .norm import NormOnColorChannel

import torch
import numpy as np
import DLUtils
import DLUtils.transform.nonlinear as nonlinear
from .nonlinear import NonLinearFunction, NonLinearModule, NonLinearTransform

class Sum(DLUtils.module.AbstractOperator):
    def Receive(self, *List):
        return sum(List)

class WeightedSum(DLUtils.module.AbstractOperator):
    def __init__(self, *List, **Dict):
        super().__init__(**Dict)
        if len(List) > 0:
            self.SetCoeff(*List)
    def SetCoeff(self, *List):
        Param = self.Param
        Param.Coeff.List = list(List)
        return self
    def Receive(self, *List):
        # Result = 0.0
        # for Element, Coeff in List:
        #     Result += Element * Coeff
        # return Result
        Result = 0.0
        for Index in self.IndexList:
            Result += self.CoeffList[Index] * List[Index]
        return Result
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.Coeff.hasattr("List")
        self.CoeffList = list(Param.Coeff.List)
        self.IndexList = range(len(self.CoeffList))
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    
class NpArray2TorchTensor(DLUtils.module.AbstractOperator):
    def Receive(self, In):
        return torch.from_numpy(In).to(self.Device, dtype=self.DataType)
    def Init(self, IsSuper=False, IsRoot=True):
        if not hasattr(self, "Device"):
            self.Device = "cpu"

        Param = self.Param
        Param.setdefault("DataType", "torch.float32")
        
        if Param.DataType in ["torch.float32", "float32"]:
            self.DataType = torch.float32
        else:
            raise Exception()
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def SetDevice(self, Device=None, IsRoot=True):
        self.Device = Device
        return super().SetDevice(Device=Device, IsRoot=IsRoot)

class MoveTensor2Device(DLUtils.module.AbstractOperator):
    ParamMap = DLUtils.IterableKeyToElement({
        ("Device"): "Device.Target"
    })
    def __init__(self, Device=None, **Dict):
        if Device is not None:
            Dict["Device"] = Device
        super().__init__(**Dict)
    def Receive(self, In):
        return In.to(self.DeviceTarget)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Device.setdefault("Target", "cpu")
        self.DeviceTarget = Param.Device.Target
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def SetDevice(self, Device=None, IsRoot=True):
        self.DeviceTarget = self.Param.Device.Target = Device
        return super().SetDevice(Device=Device, IsRoot=IsRoot)