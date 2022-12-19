import DLUtils
from ..AbstractModule import AbstractOperator
class Norm(AbstractOperator):
    def __init__(self, Min1=None, Max1=None, Min2=None, Max2=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.transform.Norm"
        Param.Min1 = Min1
        Param.Max1 = Max1
        Param.Min2 = Min2
        Param.Max2 = Max2
    def Receive(self, Input):
        Output = (Input - self.Min1) * self.Scale + self.Min2
        return Output
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Min1 = Param.Min1
        self.Max1 = Param.Max1
        self.Min2 = Param.Min2
        self.Max2 = Param.Max2
        self.Scale = (self.Max2 - self.Min2) / (self.Max1 - self.Min1)
        super().Init(IsSuper=IsSuper, IsRoot=IsRoot)
        return self