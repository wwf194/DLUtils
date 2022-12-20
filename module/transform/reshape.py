import torch
import torch.nn.functional as F
from ..AbstractModule import AbstractOperator
class Reshape(AbstractOperator):
    def __init__(self, *List):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.transform.Reshape"
        if len(List) > 0:
            if len(List) == 1 and isinstance(List[0], tuple) or isinstance(List[0], list):
                Shape = list(List[0])
            else:
                Shape = List
            Param.Shape.After = Shape
    def Receive(self, Input):
        return torch.reshape(Input, self.Shape)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Shape = Param.Shape.After
        super().Init(IsSuper=IsSuper, IsRoot=IsRoot)

class Index2OneHot(AbstractOperator):
    def __init__(self, FeatureNum):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.transform.Index2OneHot"
        Param.FeatureNum = FeatureNum
    def Receive(self, Input):
        return F.one_hot(Input.long(), num_classes=self.FeatureNum)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.FeatureNum = Param.FeatureNum
        return super().Init(IsSuper=True, IsRoot=IsRoot)

