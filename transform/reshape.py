import torch
import torch.nn.functional as F
from .. import AbstractOperator

import DLUtils
class Reshape(AbstractOperator):
    def __init__(self, *List, **Dict):
        if len(List) > 0:
            assert Dict.get("Shape") is None
            Dict["Shape"] = List
        super().__init__(**Dict)
    def SetParam(self, **Dict):
        for Key, Value in Dict.items():
            if Key in ["Shape"]:
                Param = self.Param
                if len(Value) > 0:
                    if len(Value) == 1 and (isinstance(Value[0], tuple) or isinstance(Value[0], list)):
                        Value = list(Value[0])
                    else:
                        Value = Value
                    Param.Shape.After = Value
                else:
                    raise Exception()
            else:
                raise Exception()
        return self
    def Receive(self, In):
        return torch.reshape(In, self.Shape)
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Shape = tuple(Param.Shape.After)
        super().Init(IsSuper=True, IsRoot=IsRoot)

class ChangeDimOrder(AbstractOperator):
    ParamMap = DLUtils.IterableKeyToElement({
        ("Order"): "Order"
    })
    def __init__(self, *List, **Dict):
        if len(List) > 0:
            assert Dict.get("Order") is None
            Dict["Order"] = List
        super().__init__(**Dict)
    def Receive(self, In):
        return torch.permute(In, self.Order)
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        assert Param.hasattr("Order")
        self.Order = tuple(Param.Order)
        super().Init(IsSuper=True, IsRoot=IsRoot)
Permute = ChangeDimOrder

class Index2OneHot(AbstractOperator):
    def __init__(self, FeatureNum):
        super().__init__()
        Param = self.Param
        Param.FeatureNum = FeatureNum
    def Receive(self, In):
        return F.one_hot(In.long(), num_classes=self.FeatureNum)
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.FeatureNum = Param.FeatureNum
        return super().Init(IsSuper=True, IsRoot=IsRoot)

