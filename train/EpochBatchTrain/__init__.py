import DLUtils
import numpy as np



class EpochBatchTrainComponent(DLUtils.module.AbstractModule):
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
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
    def EventList(self, Dict):
        for _Event in self._EventList:
            _Event(Dict)
        return self
    # def Log(self, Content):
    #     Param = self.Param
    #     Param.Data.Log.append(Content)
    #     return self

class EventAfterEveryEpoch(EpochBatchTrainComponent):
    def SetEvent(self, Event):
        self.Event = Event
        return self
    def AfterEpoch(self, Dict):
        self.Event(Dict)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        assert hasattr(self, "Event")
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class EventAfterTrain(EpochBatchTrainComponent):
    def SetEvent(self, Event):
        self.Event = Event
        return self
    def SetEventList(self, *EventList):
        self._EventList = EventList
        self.Event = self.EventList
        return self
    def AfterTrain(self, Dict):
        self.Event(Dict)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        assert hasattr(self, "Event")
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class EventAfterEpoch(EpochBatchTrainComponent):
    SetParamMap = DLUtils.IterableKeyToElement({
        ("Num"): "Event.Num",
        ("Mode"): "Test.Mode"
    })
    def __init__(self, **Dict):
        super().__init__()
        self.SetParam(**Dict)
    # def SetParam(self, **Dict):
    #     Param = self.Param
    #     Num = Dict.get("Num")
    #     if Num is not None:
    #         Param.Event.Num = Num
    #     Mode = Dict.get("Mode")
    #     if Mode is not None:
    #         Param.Test.Mode = Mode
    #     return self
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
    SetParamMap = DLUtils.IterableKeyToElement({
        ("BatchInterval"): "Batch.Interval"
    })
    def __init__(self, BatchInterval=None, *List, **Dict):
        if BatchInterval is not None:
            Dict["BatchInterval"] = BatchInterval
        super().__init__(*List, **Dict)
    def AfterBatch(self, Dict):
        self.BatchIndex += 1
        if self.BatchIndex == self.BatchInterval:
            self.Event(Dict)
            self.BatchIndex = 0
        return self
    def BindEvent(self, Event):
        self.Event = Event
    def BeforeTrain(self, Dict):
        self.BatchIndex = 0
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.BatchInterval = Param.Batch.Interval
        assert self.BatchInterval > 0
        # decide event to be triggered after batch interval
        assert hasattr(self, "Event")
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class EventAfterEveryBatch(EpochBatchTrainComponent):
    def __init__(self):
        super().__init__()
    def AfterBatch(self, Dict):
        self.Event(**Dict) # Called if child class has not overwrite AfterBatch.
        #raise Exception() # AfterBatch method must be overwritten by child class.
        return self

class EpochBatchTrainSession(DLUtils.module.AbstractModule):
    def __init__(self, Log=None, *List, **Dict):
        if Log is not None:
            Dict["Log"] = Log
        super().__init__(*List, **Dict)
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
        super().Bind(**Dict)
        return self
    def BindOptimizer(self, Optimizer):
        self.AddSubModule("Optimizer", Optimizer)
        return self
    def AddSubModule(self, *List, **Dict):
        super().AddSubModule(*List, **Dict)
        return self
    def BeforeTrain(self, Dict):
        Param = self.Param
        Task = self.Task
        TrainData = Task.TrainData(BatchSize=Param.Batch.Size)
        TestData  = Task.TestData(BatchSize=Param.Batch.Size)
        self.BatchNum = TrainData.GetBatchNum()
        self.BindModule("TrainData", TrainData)
        self.BindModule("TestData", TestData)
        # self.TrainData = TrainData
        # self.TestData = TestData
        self.AddSubModule("EpochIndexLog", DLUtils.struct.IntRange())
        self.AddSubModule("BatchIndexLog", DLUtils.module.EmptyModule())
        
        Modules = dict(self.SubModules)
        Modules.update(self.BindModules)
        for Name, SubModule in Modules.items():
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
        SubModule = DLUtils.struct.IntRange()
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
        self.UnBind("TrainData")
        self.UnBind("TestData")
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
    def SimulateAfterTrain(self, Module):
        Param = self.Param
        Dict = DLUtils.new_param(
            TrainSession=self,
            EpochIndex = Param.Epoch.Num - 1,
            BatchIndex = Param.Batch.Num - 1,
            EpochNum = Param.Epoch.Num,
            BatchNum = Param.Batch.Num,
            Model = self.Model,
            EvaluationLog = self.EvaluationLog,
            Evaluator = self.Evaluator,
            SaveDir= Param.SaveDir
        )
        if hasattr(self, "Device"):
            Dict.Device = self.Device
        Module.AfterTrain(Dict)
        return self
    def SetParam(self, **Dict):
        super().SetParam(**Dict)
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
            Dict.SaveDir = Param.SaveDir
            self.BeforeEpoch(Dict)
            for BatchIndex in range(self.BatchNum):
                Dict.BatchIndex = BatchIndex
                self.BeforeBatch(Dict)
                In, OutTarget = TrainData.Get(BatchIndex)
                # if BatchIndex < 3:
                    # DLUtils.Tensor2TextFile2D(Input[3], "./test/mnist/mlp/Input-Epoch%3d-Batch%3d"%(EpochIndex, BatchIndex))
                    # DLUtils.Tensor2ImageFile(Input[3], "./test/mnist/mlp/Input-Epoch%3d-Batch%3d-Index03.png"%(EpochIndex, BatchIndex))
                Output = Model(In)
                Dict.In = In
                Dict.Out = Output
                Dict.OutTarget = OutTarget
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
    def ToFile(self, SavePath=None, RetainSelf=True):
        Param = DLUtils.Param(self.Param)
        if SavePath is None:
            assert hasattr(Param, "SaveDir")
            SavePath = Param.SaveDir + "TrainSession.dat"
        else:
            if SavePath.endswith("/"):
                SavePath = SavePath + "TrainSession.dat"
        if not RetainSelf:
            self.UnBind("TrainData")
            self.UnBind("TestData")
        for Name, SubModule in self.SubModules.items():
            if Name in ["TrainData", "TestData"]:
                Param.SubModules.delattr(Name)
                continue
            else:
                Param.SubModules.setattr(Name, SubModule.ExtractParam())
        Param.ToFile(SavePath)
        return self

from .Component import \
    Save, \
    Test, \
    AnalysisAfterTrain, EvaluatorPredAndTargetSelect1FromN, \
    DataLoaderForEpochBatchTrain

import DLUtils.train.Select1FromN as Select1FromN