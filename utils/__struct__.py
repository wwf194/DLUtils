import DLUtils
import re
import numpy as np

class FixedSizeQueuePassiveOut(): # first in, first out. out is passive. queue is fixed size.
    def __init__(self, NumMax):
        assert NumMax > 0
        self.NumMax = NumMax
        self.Index = 0
        self.Num = 0
        self.IsFull = False
        self.append = self.put
    def _Sum(self):
        return self.Data.sum()
    def Average(self):
        return self.Data.mean()
    def put(self, In):
        Index = self.Index
        Data = self.Data
        Out = self.Data[Index]
        Data[Index] = In
        self.Index += 1
        if self.Index == self.NumMax:
            self.Index = 0
        self.SumCache += In
        if self.IsFull:
            self.SumCache -= Out
            return Out
        else:
            self.Num += 1
            if self.Num == self.NumMax:
                self.IsFull = True
            return None
    def _SumCached(self):
        return self.SumCache

class FixedSizeQueuePassiveOutInt(FixedSizeQueuePassiveOut):
    def __init__(self, NumMax, KeepSum=True):
        self.Data = np.zeros((NumMax), np.int32)
        if KeepSum:
            self.Sum = self._SumCached
        else:
            self.Sum = self._Sum
        self.SumCache = 0
        super().__init__(NumMax)

class FixedSizeQueuePassiveOutFloat(FixedSizeQueuePassiveOut):
    def __init__(self, NumMax, KeepSum=True):
        self.Data = np.zeros((NumMax), np.float32)
        if KeepSum:
            self.Sum = self._SumCached
        else:
            self.Sum = self._Sum
        self.SumCache = 0
        super().__init__(NumMax)

class IntRange(DLUtils.module.AbstractModule):
    # represesnt a single int range.
    # to be implemented. represents multiple int ranges.
    # to be implemented. automatically detect overlap and converge ranges.
    def __init__(self, Logger=None):
        super().__init__(Logger)
        Param = self.Param
        self.Start = None
        self.Next = None
        self.append = self._appendFirst
    def _appendFirst(self, Num):
        self.Start = Num
        self.Next = Num + 1
        self.append = self._appendNext
        return self
    def _appendNext(self, Num):
        if Num == self.Next:
            self.Next += 1
        return self
    def ExtractParam(self, *List, RetainSelf=True, **Dict):
        Param = self.Param
        Start = self.Start
        if self.Next is None:
            End = None
        else:
            End = self.Next - 1
        Param.Range.Start = Start
        Param.Range.End = End
        Param.Range.IncludeRight = True
        return super().ExtractParam(RetainSelf=RetainSelf)
    def LoadParam(self, Param):
        return super().LoadParam(Param)
    def Build(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.Start = Param.Range.Start
        self.Next = Param.Range.End
        self.append = self._appendNext
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def Extract(self):
        if self.Start is None:
            return None
        else:
            return [self.Start, self.End]