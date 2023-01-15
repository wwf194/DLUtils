import DLUtils
from .. import AbstractOperator
class Norm(AbstractOperator):
    def __init__(self, Min1=None, Max1=None, Min2=None, Max2=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.transform.Norm"
        Param.Min1 = Min1
        Param.Max1 = Max1
        Param.Min2 = Min2
        Param.Max2 = Max2
    def Receive(self, In):
        Output = (In - self.Min1) * self.Scale + self.Min2
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

class NormOnColorChannel(AbstractOperator):
    def __init__(self, Mean0=None, Std0=None, Mean1=None, Std1=None):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.transform.Norm"
        if Mean0 is not None:
            Param.Before.Min = Mean0
        if Std0 is not None:
            Param.Before.Std = Std0
        if Mean1 is not None:
            Param.After.Min = Mean1
        if Std1 is not None:
            Param.After.Std = Std1
    def _ReceiveMean0Std1(self, In):
        # Input : [BatchSize, Channel, Width, Height]
        Output = (Input - self.Mean0[None, :, None, None]) / self.Std0[None, :, None, None]
        return Output
    def _Receive(self, In):
        return Output
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.After.setdefault("Mean", 0.0)
        Param.After.setdefault("Std", 1.0)
        self.Mean0 = Param.Before.Mean
        self.Std0 = Param.Before.Std
        self.Mean1 = Param.After.Mean
        self.Std1 = Param.After.Std1

        if self.Mean1 == 0.0 and self.Std1 == 1.0:
            self.Receive = self._ReceiveMean0Std1
        super().Init(IsSuper=IsSuper, IsRoot=IsRoot)
        return self