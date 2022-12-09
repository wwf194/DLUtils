import DLUtils

class LogForEpochBatchTrain(DLUtils.module.AbstractModule):
    def __init__(self, **kw):
        #kw.setdefault("DataOnly", True) # Log class do not need param
        super().__init__(**kw)
    def SetEpochNum(self, EpochNum):
        self.EpochNum = EpochNum
        return self
    def SetBatchNum(self, BatchNum):
        self.BatchNum = BatchNum
        return self
    def SetEpochIndex(self, EpochIndex):
        self.EpochIndex = EpochIndex
        return self
    def SetBatchIndex(self, BatchIndex):
        self.BatchIndex = BatchIndex
        return self
    def GetEpochNum(self):
        return self.EpochNum
    def GetBatchNum(self):
        return self.BatchNum
    def GetEpochIndex(self):
        return self.EpochIndex
    def GetBatchIndex(self):
        return self.BatchIndex
    def SetEpochBatchIndex(self, EpochIndex, BatchIndex):
        self.EpochIndex = EpochIndex
        self.BatchIndex = BatchIndex
        return self

class AbstractLogAlongEpochBatchTrain(LogForEpochBatchTrain):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)

AbstractModuleAlongEpochBatchTrain = AbstractLogAlongEpochBatchTrain

class AbstractLogAlongBatch(LogForEpochBatchTrain):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        super().BeforeBuild(IsLoad=IsLoad)
        return self
