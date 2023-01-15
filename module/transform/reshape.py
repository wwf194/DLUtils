import torch
import torch.nn.functional as F
from .. import AbstractOperator
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

