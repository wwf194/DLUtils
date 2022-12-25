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
        TestNum = Dict.get("TestNum")
        if TestNum is not None:
            Param.Test.Num = TestNum
        TestMode = Dict.get("TestDistribution")
        if TestMode is not None:
            Param.Test.Mode = TestMode
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
        if EpochIndex == self.TestEpochNext:
            self.Event(Dict)
            self.TestEpochNext = self.NextTestEpoch()
        return self
    def _AfterEveryEpoch(self, Dict):
        self.Event(Dict)
        return self
    def NextTestEpoch(self):
        if self.Index < self.TestEpochNum:
            TestEpoch = self.TestEpochIndexList[self.Index]
            self.Index += 1
            return TestEpoch
        else:
            return None
    def BindTrainSession(self, TrainSession):
        self.TrainSession = TrainSession
        return super().BindTrainSession(TrainSession)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Mode = Param.Test.setdefault("Mode", "Same Interval")
        if Mode in ["Same Interval"]:
            if not Param.Test.hasattr("Num"):
                Param.Test.Num = 10
                self.Log("Test.Num is not set. Set to default value: 10")
            TestNum = Param.Test.Num
            if TestNum in ["All"]:
                self.AfterEpoch = self._AfterEveryEpoch
            else:
                self.AfterEpoch = self._AfterEpoch
        else:
            raise Exception()
        return super().Init(IsSuper, IsRoot)
    def BeforeTrain(self, Dict):
        Param = self.Param
        Mode = Param.Test.setdefault("Mode", "Same Interval")
        if Mode in ["Same Interval"]:
            TestNum = Param.Test.Num
            if TestNum in ["All"]:
                pass
            else:
                self.AfterEpoch = self._AfterEpoch
                if Dict.EpochNum <= TestNum:
                    self.TestEpochIndexList = range(Dict.EpochNum)
                else:
                    IndexFloat = np.asarray(range(TestNum), dtype=np.float32) * (Dict.EpochNum - 1)
                    self.TestEpochIndexList = np.round(IndexFloat)
                self.Index = 0
                self.TestEpochNext = self.TestEpochIndexList[0]
                self.TestEpochNum = len(self.TestEpochIndexList)
        return self

class EventAfterEveryBatch(EpochBatchTrainComponent):
    def __init__(self):
        super().__init__()
    def AfterBatch(self, **Dict):
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
    def BeforeTrain(self, Dict):
        Param = self.Param
        Task = self.Task
        TrainData = Task.TrainData(BatchSize=Param.BatchSize)
        TestData  = Task.TestData(BatchSize=Param.BatchSize)
        self.AddSubModule("TrainData", TrainData)
        self.AddSubModule("TestData", TestData)

        for Name, SubModule in self.SubModules.items():
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
        # if len(self.BeforeTrainList) == 0:
        #     self.BeforeTrain = DLUtils.EmptyFunction
        if len(self.AfterTrainList) == 0:
            self.AfterTrain = DLUtils.EmptyFunction
        if len(self.BeforeEpochList) == 0:
            self.BeforeEpoch = DLUtils.EmptyFunction
        if len(self.AfterEpochList) == 0:
            self.AfterEpoch = DLUtils.EmptyFunction
        if len(self.BeforeBatchList) == 0:
            self.BeforeBatch = DLUtils.EmptyFunction
        if len(self.AfterBatchList) == 0:
            self.AfterBatch = DLUtils.EmptyFunction
        for Event in self.BeforeTrainList:
            Event(Dict)
        return self
    def BeforeEpoch(self, Dict):
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
        return self
    def AfterEpoch(self, Dict):
        for Event in self.AfterEpochList:
            Event(Dict)
        return self
    def AfterTrain(self, Dict):
        for Event in self.AfterTrainList:
            Event(Dict)
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
            "EpochNum":self.EpochNum,
            
            "TrainSession": self
        })
        self.BeforeTrain(Dict)

        TrainData = Dict.TrainData = self.TrainData
        TestData  = Dict.TestData  = self.TestData
        
        for EpochIndex in range(self.EpochNum):
            self.Log(f"EpochIndex: {EpochIndex}", "TrainSession")
            Dict.EpochIndex = EpochIndex
            self.BeforeEpoch(Dict)
            for BatchIndex in range(self.BatchNum):
                Dict.BatchIndex = BatchIndex
                self.BeforeBatch(Dict)
                Input, OutputTarget = TrainData.Get(BatchIndex)
                if BatchIndex < 3:
                    #DLUtils.Tensor2TextFile2D(Input[3], "./test/mnist/mlp/Input-Epoch%3d-Batch%3d"%(EpochIndex, BatchIndex))
                    DLUtils.Tensor2ImageFile(Input[3], "./test/mnist/mlp/Input-Epoch%3d-Batch%3d-Index03.png"%(EpochIndex, BatchIndex))
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
        assert Param.hasattr("BatchSize")
        self.BatchSize = Param.BatchSize
        super().Init(IsSuper=True, IsRoot=IsRoot)
        return self
    def ToFile(self, SaveDir):
        Param = self.Param
        for Name, SubModule in self.SubModules.items():
            Param.setattr(Name, SubModule.ExtractParam())
        Param.ToFile(SaveDir + "TrainSession")
        return self
    

from .Component import Save, Test