import DLUtils
import numpy as np

class EpochBatchTrainComponent(DLUtils.module.AbstractModule):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param.Data.Log = DLUtils.Param([])
    def BindTrainSession(self, TrainSession):
        assert isinstance(TrainSession, DLUtils.train.EpochBatchTrainSession)
        if hasattr(self, "BeforeTrain"):
            TrainSession.AddBeforeTrainEvent(self.BeforeTrain)
        if hasattr(self, "BeforeEpoch"):
            TrainSession.AddBeforeEpochEvent(self.BeforeEpoch)
        if hasattr(self, "BeforeBatch"):
            TrainSession.AddBeforeBatchEvent(self.BeforeBatch)
        if hasattr(self, "AfterBatch"):
            TrainSession.AddAfterBatchEvent(self.AfterBatch)
        if hasattr(self, "AfterEpoch"):
            TrainSession.AddAfterEpochEvent(self.AfterEpoch)
        if hasattr(self, "AfterTrain"):
            TrainSession.AddAfterTrainEvent(self.AfterTrain)
    # def Log(self, Content):
    #     Param = self.Param
    #     Param.Data.Log.append(Content)
    #     return self

class EventAfterEpoch(EpochBatchTrainComponent):
    def __init__(self, **Dict):
        super().__init__()
        self.SetParam(**Dict)
    def SetParam(self, **Dict):
        Param = self.Param
        Num = Dict.get("Num")
        if Num is not None:
            Param.Event.Num = Num
        Mode = Dict.get("Mode")
        if Mode is not None:
            Param.Test.Mode = Mode
        return self
    def BindEvaluator(self, Evaluator):
        self.Evaluator = Evaluator
        return self
    def BindTestData(self, TestData):
        self.TestData = TestData
        return self
    def BindModel(self, Model):
        self.Model = Model
        return self
    def Bind(self, **Dict):
        Model = Dict.get("Model")
        if Model is not None:
            self.BindModel(Model)
        TestData = Dict.get("TestData")
        if TestData is not None:
            self.BindTestData(TestData)
        Evaluator = Dict.get("Evaluator")
        if Evaluator is not None:
            self.BindEvaluator(Evaluator)
        return self
    def _AfterEpoch(self, Dict):
        EpochIndex = Dict.EpochIndex
        if EpochIndex == self.EventEpochNext:
            self.Event(Dict)
            self.EventEpochNext = self.NextEventEpoch()
        return self
    def _AfterEveryEpoch(self, Dict):
        self.Event(Dict)
        return self
    def NextEventEpoch(self):
        self.Index += 1
        if self.Index < self.EventEpochNum:
            EventEpochNext = self.EventEpochIndexList[self.Index]
            return EventEpochNext
        else:
            return None
    def BindTrainSession(self, TrainSession):
        self.TrainSession = TrainSession
        return super().BindTrainSession(TrainSession)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Mode = Param.Test.setdefault("Mode", "Same Interval")
        if Mode in ["Same Interval"]:
            if not Param.Event.hasattr("Num"):
                Param.Event.Num = 10
                self.Log("Event.Num is not set. Set to default value: 10")
            EventNum = Param.Event.Num
            if EventNum in ["All"]:
                self.AfterEpoch = self._AfterEveryEpoch
            else:
                self.AfterEpoch = self._AfterEpoch
            self.EventNum = EventNum
        else:
            raise Exception()
        return super().Init(IsSuper, IsRoot)
    def BeforeTrain(self, Dict):
        Param = self.Param
        Mode = Param.Event.setdefault("Mode", "Same Interval")
        if Mode in ["Same Interval"]:
            EventNum = Param.Event.Num
            if EventNum in ["All"]:
                pass
            else:
                self.AfterEpoch = self._AfterEpoch
                if Dict.EpochNum <= EventNum:
                    self.EventEpochIndexList = list(range(Dict.EpochNum))
                else:
                    IndexFloat = np.asarray(range(EventNum), dtype=np.float32) / (EventNum - 1) * (Dict.EpochNum - 1)
                    self.EventEpochIndexList = np.round(IndexFloat).astype(np.int32)
                    # np.round change values to float number nearest to nearby integer.
                self.Index = 0
                self.EventEpochNext = self.EventEpochIndexList[0]
                self.EventEpochNum = len(self.EventEpochIndexList)
        else:
            raise Exception()
        return self

class EventAfterFixedBatch(EpochBatchTrainComponent):
    def __init__(self, BatchInterval=None, Event=None):
        super().__init__()
        if BatchInterval is not None:
            assert BatchInterval > 0
            self.BatchInterval = BatchInterval
        if Event is not None:
            self.BindEvent(Event)
    def AfterBatch(self, Dict):
        self.BatchNum += 1
        if self.BatchNum == self.BatchInterval:
            self.Event(Dict)
            self.BatchNum = 0
        return self
    def BindEvent(self, Event):
        self.Event = Event
    def BeforeTrain(self, Dict):
        self.BatchNum = 0

class EventAfterEveryBatch(EpochBatchTrainComponent):
    def __init__(self):
        super().__init__()
    def AfterBatch(self, Dict):
        self.Event(**Dict) # Called if child class has not overwrite AfterBatch.
        #raise Exception() # AfterBatch method must be overwritten by child class.
        return self

class EpochBatchTrainSession(DLUtils.module.AbstractModule):
    def __init__(self, Logger=None):
        super().__init__(Logger=Logger)
        self.BeforeTrainList = []
        self.BeforeEpochList = []
        self.BeforeBatchList = []
        self.AfterBatchList = []
        self.AfterEpochList = []
        self.AfterTrainList = []
    def BindTask(self, Task):
        self.AddSubModule("Task", Task)
        return self
    def BindTrainData(self, TrainData):
        self.AddSubModule("TrainData", TrainData)
        return self
    def BindTestData(self, TestData):
        self.AddSubModule("TestData", TestData)
        return self
    def BindEvaluator(self, Evaluator):
        self.AddSubModule("Evaluator", Evaluator)
        if hasattr(Evaluator, "BindTrainSession"):
            Evaluator.BindTrainSession(self)
        return self
    def BindModel(self, Model):
        self.AddSubModule("Model", Model)
        return self
    def Bind(self, **Dict):
        for Name, SubModule in Dict.items():
            self.AddSubModule(Name, SubModule)
        return self
    def BindOptimizer(self, Optimizer):
        self.AddSubModule("Optimizer", Optimizer)
        return self
    def AddSubModule(self, Name, SubModule):
        super().AddSubModule(Name, SubModule)
        return self
    def BeforeTrain(self, Dict):
        Param = self.Param
        Task = self.Task
        TrainData = Task.TrainData(BatchSize=Param.BatchSize)
        TestData  = Task.TestData(BatchSize=Param.BatchSize)
        self.BatchNum = TrainData.BatchNum()
        self.AddSubModule("TrainData", TrainData)
        self.AddSubModule("TestData", TestData)
        # self.TrainData = TrainData
        # self.TestData = TestData
        self.AddSubModule("EpochIndexLog", DLUtils.structure.IntRange())
        self.AddSubModule("BatchIndexLog", DLUtils.module.EmptyModule())
        
        for Name, SubModule in dict(self.SubModules).items():
            if hasattr(SubModule, "BindTrainSession"):
                SubModule.BindTrainSession(self)
            else:
                if hasattr(SubModule, "BeforeTrain"):
                    self.AddBeforeTrainEvent(SubModule.BeforeTrain)
                if hasattr(SubModule, "BeforeEpoch"):
                    self.AddBeforeEpochEvent(SubModule.BeforeEpoch)
                if hasattr(SubModule, "BeforeBatch"):
                    self.AddBeforeBatchEvent(SubModule.BeforeBatch)
                if hasattr(SubModule, "AfterBatch"):
                    self.AddAfterBatchEvent(SubModule.AfterBatch)
                if hasattr(SubModule, "AfterEpoch"):
                    self.AddAfterEpochEvent(SubModule.AfterEpoch)
                if hasattr(SubModule, "AfterTrain"):
                    self.AddAfterTrainEvent(SubModule.AfterTrain)
        for Event in self.BeforeTrainList:
            Event(Dict)
        # if len(self.BeforeTrainList) == 0:
        #     self.BeforeTrain = DLUtils.EmptyFunction
        if len(self.AfterTrainList) == 0:
            self.AfterTrain = DLUtils.EmptyFunction
        if len(self.BeforeEpochList) == 0:
            self.BeforeEpoch = DLUtils.EmptyFunction
        # if len(self.AfterEpochList) == 0:
        #     self.AfterEpoch = DLUtils.EmptyFunction
        if len(self.BeforeBatchList) == 0:
            self.BeforeBatch = DLUtils.EmptyFunction
        if len(self.AfterBatchList) == 0:
            self.AfterBatch = DLUtils.EmptyFunction
        return self
    def BeforeEpoch(self, Dict):
        Name = "%d"%Dict.EpochIndex
        SubModule = DLUtils.structure.IntRange()
        self.BatchIndexLog.AddSubModule(Name, SubModule)
        self._BatchIndexList = SubModule
        for Event in self.BeforeEpochList:
            Event(Dict)
        return self
    def BeforeBatch(self, Dict):
        for Event in self.BeforeBatchList:
            Event(Dict)
        return self
    def AfterBatch(self, Dict):
        for Event in self.AfterBatchList:
            Event(Dict)
        self._BatchIndexList.append(Dict.EpochIndex)
        return self
    def AfterEpoch(self, Dict):
        self.EpochIndexLog.append(Dict.EpochIndex)
        for Event in self.AfterEpochList:
            Event(Dict)
        return self
    def AfterTrain(self, Dict):
        for Event in self.AfterTrainList:
            Event(Dict)
        self.RemoveSubModule("TrainData")
        self.RemoveSubModule("TestData")
        return self
    def AddBeforeTrainEvent(self, Event):
        self.BeforeTrainList.append(Event)
    def AddBeforeEpochEvent(self, Event):
        self.BeforeEpochList.append(Event)
    def AddBeforeBatchEvent(self, Event):
        self.BeforeBatchList.append(Event)
    def AddAfterBatchEvent(self, Event):
        self.AfterBatchList.append(Event)
    def AddAfterEpochEvent(self, Event):
        self.AfterEpochList.append(Event)
    def AddAfterTrainEvent(self, Event):
        self.AfterTrainList.append(Event)
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in Dict.items():
            if Key in ["EpochNum"]:
                Param.Epoch.Num = Value
                self.EpochNum = Value
            elif Key in ["BatchNum"]:
                Param.Batch.Num = Value
                self.BatchNum = Value
            else:
                Param.setattr(Key, Value)

        return self
    def EpochIndexList(self):
        return self.EpochIndexLog.Extract()
    def BatchIndexList(self):
        _BatchIndexList = {}
        for Name, SubModule in self.BatchIndex:
            _BatchIndex = int(Name)
            _BatchIndexList[_BatchIndex] = SubModule.Extract()
        return _BatchIndexList
    def Start(self):
        Param = self.Param
        Model = self.Model
        Evaluator = self.Evaluator
        EvaluationLog = self.EvaluationLog
        Task = self.Task 
        BatchSize = self.BatchSize
        Optimizer = self.Optimizer
        Dict = DLUtils.param({
            "Optimizer": Optimizer, "Model": Model, 
            "Evaluator": Evaluator, "EvaluationLog": EvaluationLog,
            "EpochNum": self.EpochNum,
            "TrainSession": self
        })
        Dict.EpochIndex = self.EpochStart # default：0
        Dict.BatchIndex = self.BatchStart - 1 # default: 0
        self.BeforeTrain(Dict)
        TrainData = Dict.TrainData = self.TrainData
        TestData  = Dict.TestData  = self.TestData
        for EpochIndex in range(self.EpochStart, self.EpochStart + self.EpochNum):
            self.Log(f"EpochIndex: {EpochIndex}", "TrainSession")
            Dict.EpochIndex = EpochIndex
            self.BeforeEpoch(Dict)
            for BatchIndex in range(self.BatchNum):
                Dict.BatchIndex = BatchIndex
                self.BeforeBatch(Dict)
                Input, OutputTarget = TrainData.Get(BatchIndex)
                # if BatchIndex < 3:
                    # DLUtils.Tensor2TextFile2D(Input[3], "./test/mnist/mlp/Input-Epoch%3d-Batch%3d"%(EpochIndex, BatchIndex))
                    # DLUtils.Tensor2ImageFile(Input[3], "./test/mnist/mlp/Input-Epoch%3d-Batch%3d-Index03.png"%(EpochIndex, BatchIndex))
                Output = Model(Input)
                Dict.Input = Input
                Dict.Output = Output
                Dict.OutputTarget = OutputTarget
                Evaluation = Evaluator.Evaluate(Dict)
                Dict.Evaluation = Evaluation
                Optimizer.Optimize(Dict)
                self.AfterBatch(Dict)
            self.AfterEpoch(Dict)
        self.AfterTrain(Dict)
        return self

    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.BatchSize = Param.Batch.setdefault("Size", "NotRegistered")
        self.BatchNum = Param.Batch.setdefault("Num", "Auto")
        self.BatchStart = Param.Batch.setdefault("Start", 0)
        self.EqualBatchNumPerBatch = Param.Epoch.setdefault("EqualBatchNumPerBatch", True)
        self.EpochStart = Param.Epoch.setdefault("Start", 0)
        self.EpochNum = Param.Epoch.Num
        super().Init(IsSuper=True, IsRoot=IsRoot)
        return self
    def BatchIndexFloatList(self, EpochIndexList, BatchIndexDict, **kw):
        if self.EqualBatchNumPerBatch:
            assert self.BatchStart == 0
            BatchNum = self.BatchNum
            BatchIndexFloat = np.zeros((self.EpochNum * self.BatchNum))
            BatchIndexFloatBase = np.asarray(range(BatchNum)) / BatchNum
            Index = 0
            EpochIndex = float(self.EpochIndexStart)
            for EpochIndex in range(self.EpochNum):
                IndexNext = Index + self.BatchNum
                BatchIndexFloat[Index:IndexNext] = BatchIndexFloatBase + EpochIndex
                Index = IndexNext
                EpochIndex += 1.0
        return BatchIndexFloat
    def ToFile(self, SaveDir, RetainSelf=True):
        Param = DLUtils.Param(self.Param)
        if not RetainSelf:
            self.RemoveSubModule("TrainData")
            self.RemoveSubModule("TestData")
        for Name, SubModule in self.SubModules.items():
            if Name in ["TrainData", "TestData"]:
                Param.SubModules.delattr(Name)
                continue
            else:
                Param.SubModules.setattr(Name, SubModule.ExtractParam())
        Param.ToFile(SaveDir + "TrainSession.dat")
        return self
from .Component import \
    Save, \
    Test, \
    AnalysisAfterTrain, XFixedSizeYFixedSizeProb
    

import DLUtils.train.SingleClassification as SingleClassification