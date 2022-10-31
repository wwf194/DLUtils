import DLUtils
from DLUtils.attr import *

class AbstractModuleAlongEpochBatchTrain(DLUtils.module.AbstractModule):
    # Child Class: trainer, log
    def __init__(self, ChildClass, **kw):
        MountLocation = kw.setdefault("MountLocation", "data")
        super().__init__(**kw)
        DLUtils.train.SetEpochBatchMethodForModule(ChildClass, **kw)
    def SetEpochBatchIndexData(self, EpochIndex, BatchIndex):
        self.data.EpochIndex = EpochIndex
        self.data.BatchIndex = BatchIndex
    def SetEpochBatchIndexCache(self, EpochIndex, BatchIndex):
        self.cache.EpochIndex = EpochIndex
        self.cache.BatchIndex = BatchIndex

class TrainerEpochBatch(AbstractModuleAlongEpochBatchTrain):
    def __init__(self, param, **kw):
        super().__init__(self.__class__, MountLocation="data")
        DLUtils.transform.InitForNonModel(self, param, **kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        cache = self.cache
        data = self.data
        
        Modules = self.Modules
        Modules.LogTrain = DLUtils.log.LogForEpochBatchTrain()
        cache.LogTrain = DLUtils.log.LogForEpochBatchTrain()

        Modules.LogTest = DLUtils.log.LogForEpochBatchTrain()
        cache.LogTest = DLUtils.log.LogForEpochBatchTrain()    

        cache.SetEpochBatchList = []
        cache.CheckPointList = []
        self.Register2SetEpochBatchList([cache.LogTrain, cache.LogTest])
        self.BuildModules()
        self.InitModules()
        self.ParseRouters()
        self.ClearEpoch()
        self.ClearBatch()
        self.RegisterCheckPoint()
    def RegisterCheckPoint(self):
        # Scan all modules and add checkpoints among them to CheckPointList.
        cache = self.cache
        cache.CheckPointList = []
        for Name, Module in ListAttrsAndValues(cache.Modules, Exceptions=["__ResolveRef__", "__Entry__"]):
            if hasattr(Module, "IsCheckPoint") and Module.IsCheckPoint is True:
                self.cache.CheckPointList.append(Module)
                if hasattr(Module, "SetBatchNum") and hasattr(cache, "BatchNum"):
                    Module.SetBatchNum(cache.BatchNum)

    def NotifyEpochIndex(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetEpochIndex(cache.EpochIndex)
    def NotifyBatchIndex(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetBatchIndex(cache.BatchIndex)
    def NotifyEpochNum(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetEpochNum(cache.EpochNum)
    def NotifyBatchNum(self):
        cache = self.cache
        for Obj in self.cache.SetEpochBatchList:
            Obj.SetBatchNum(cache.BatchNum)
    def Register2SetEpochBatchList(self, List):
        cache = self.cache
        #cache.SetEpochBatchList = []
        for Obj in List:
            Obj = DLUtils.parse.ResolveStr(Obj)
            cache.SetEpochBatchList.append(Obj)
    def GenerateContextInfo(self):
        cache = self.cache
        return DLUtils.PyObj({
            "Trainer": self,
            "EpochNum": cache.EpochNum,
            "BatchNum": cache.BatchNum,
            "EpochIndex": cache.EpochIndex,
            "BatchIndex": cache.BatchIndex,
        })
    # def __call__(self):
    #     DLUtils.CallGraph(self.Dynamics.Main)
    def ReportEpochBatch(self):
        cache = self.cache
        DLUtils.AddLog("Epoch%d-Batch%d"%(cache.EpochIndex, cache.BatchIndex))

#DLUtils.transform.SetMethodForNonModelClass(TrainerEpochBatch)
#DLUtils.transform.SetEpochBatchMethodForModule(TrainerEpochBatch)