import torch
import torch.nn.functional as F
from .. import AbstractOperator
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
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Shape = Param.Shape.After
        super().Init(IsSuper=True, IsRoot=IsRoot)

class Index2OneHot(AbstractOperator):
    def __init__(self, FeatureNum):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.transform.Index2OneHot"
        Param.FeatureNum = FeatureNum
    def Receive(self, In):
        return F.one_hot(In.long(), num_classes=self.FeatureNum)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.FeatureNum = Param.FeatureNum
        return super().Init(IsSuper=True, IsRoot=IsRoot)

