import DLUtils
from DLUtils.attr import *



import numpy as np
from DLUtils.module.AbstractModule import AbstractModule
from .. import EpochBatchTrainComponent
from .. import EventAfterEpoch

class TestInEpochBatchTrainProcess(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        self.Event = self.Test # to be called
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
    def Test(self, **Dict):
        TestData = self.TestData
        Model = self.Model
        Evaluator = self.Evaluator
        EvaluationLog = self.EvaluationLog
        BatchNum = TestData.BatchNum()
        EvaluationLog.BeforeTestEpoch(**Dict)
        for TestBatchIndex in range(BatchNum):
            Input, OutputTarget = TestData.Get(TestBatchIndex)
            Output = Model(Input)
            Evaluation = Evaluator.Evaluate(
                Input, Output, OutputTarget, Model
            )
            Dict["BatchIndex"] = TestBatchIndex
            EvaluationLog.AfterTestBatch(Evaluation=Evaluation, **Dict)
        return self

class CheckPointForEpochBatchTrain(DLUtils.log.AbstractLogAlongEpochBatchTrain):
    def __init__(self, **kw):
        #DLUtils.transform.InitForNonModel(self, param, ClassPath="DLUtils.Train.CheckPointForEpochBatchTrain", **kw)
        super().__init__(**kw)
    def SetMethod(self, Method):
        assert callable(Method)
        self.cache.Method = Method
        self.param.Method = str(Method)
        return self
    def Build(self, IsLoad=False):
        #DLUtils.transform.BuildForNonModel(self, IsLoad)
        super().BeforeBuild(IsLoad=IsLoad)
        # Intervals are calculated in batches, not epochs.
        param = self.param
        cache = self.cache
        EnsureAttrs(param, "CalculateCheckPointMode", default="EndOfEpoch")
        if param.CalculateCheckPointMode in ["Static"]: # For cases where EpochNum and BatchNum is known before training.
            assert HasAttrs(param, "Epoch.Num")
            assert HasAttrs(param, "Batch.Num")
            EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
            EnsureAttrs(param, "Interval.Start", value=10)
            EnsureAttrs(param, "Interval.Max", value=10 * param.Batch.Num)
            cache.CheckPointList = self.CalculateCheckPointList(param)
            cache.CheckPointNextIndex = 0
            cache.CheckPointNext = self.CheckPointList[self.CheckPointNextIndex]
            self.AddBatch = self.AddBatchStatic
        elif param.CalculateCheckPointMode in ["Online"]:
            # No need to know BatchNum in advance
            EnsureAttrs(param, "Interval.IncreaseCoefficient", value=1.5)
            EnsureAttrs(param, "Interval.Start", value=10)
            EnsureAttrs(param, "Interval.Max", value=10000)
            cache.IntervalCurrent = param.Interval.Start
            cache.IntervalIndex = 0
            cache.IntervalMax = param.Interval.Max
            cache.IntervalIncreaseCoefficient = param.Interval.IncreaseCoefficient
            self.AddBatch = self.AddBatchOnline
        elif param.CalculateCheckPointMode in ["Always", "EveryBatch"]:
            cache.IntervalIndex = 0
            self.AddBatch = self.AddBatchAlwaysTrue
        elif param.CalculateCheckPointMode in ["EndOfEpoch"]:
            pass
        else:
            raise Exception(param.CalculateCheckPointMode)
        
        if param.CalculateCheckPointMode in ["EndOfEpoch"]:
            self.NotifyEndOfEpoch = self.NotifyEndOfEpochAlwaysTrue
            self.AddBatch = self.AddBatchAlwaysFalse
            cache.IsCheckPoint = False
        else:
            self.NotifyEndOfEpoch = self.NotifyEndOfEpochAlwaysFalse

        self.AddEpoch = self.AddEpochAlwaysFalse

        cache.EpochIndex = 0
        cache.BatchIndex = -1
        cache.BatchIndexTotal = -1

        if hasattr(param, "Method"):
            #EnsureAttrs(param, "Method", default="&#DLUtils.functions.NullFunction")
            cache.Method = DLUtils.parse.ResolveStr(
                param.Method,
                ObjCurrent=self.param,
                ObjRoot=DLUtils.GetGlobalParam()
            )
        
        #self.SetBatchIndex = self.AddBatch
        return self
    def SetBatchIndex(self, BatchIndex):
        self.AddBatch()
    def CalculateCheckPointList(param):
        BatchNumTotal = param.Epoch.Num * param.Batch.Num
        CheckPointBatchIndices = []
        BatchIndexTotal = 0
        CheckPointBatchIndices.append(BatchIndexTotal)
        Interval = param.Interval.Start
        while BatchIndexTotal < BatchNumTotal:
            BatchIndexTotal += round(Interval)
            CheckPointBatchIndices.append(BatchIndexTotal)
            Interval *= param.Interval.IncreaseCoefficient
            if Interval > param.Interval.Max:
                Interval =param.Interval.Max
        return CheckPointBatchIndices
    def IsCheckPoint(self):
        return self.cache.IsCheckPoint
    def AddBatchStatic(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IsCheckPoint = False
        if cache.BatchIndexTotal >= self.CheckPointNext:
            cache.IsCheckPoint = True
            cache.CheckPointNextIndex += 1
            cache.CheckPointNext = cache.CheckPointList[self.CheckPointNextIndex]
            return cache.IsCheckPoint, self.GetMethod()
        else:
            return cache.IsCheckPoint, None
    def AddBatchOnline(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IntervalIndex += 1
        if cache.IntervalIndex >= cache.IntervalCurrent:
            cache.IsCheckPoint = True
            cache.IntervalCurrent = round(cache.IntervalCurrent * cache.IntervalIncreaseCoefficient)
            if cache.IntervalCurrent > cache.IntervalMax:
                cache.IntervalCurrent = cache.IntervalMax
            cache.IntervalIndex = 0
            return cache.IsCheckPoint, self.GetMethod()
        else:
            cache.IsCheckPoint = False
            return cache.IsCheckPoint, None
    def AddBatchAlwaysTrue(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IsCheckPoint = True
        return cache.IsCheckPoint, self.cache.Method
    def AddBatchAlwaysFalse(self, *arg, **kw):
        cache = self.cache
        cache.BatchIndex += 1
        cache.BatchIndexTotal += 1
        cache.IsCheckPoint = False
        return cache.IsCheckPoint, self.cache.Method
    def GetMethod(self):
        return self.cache.Method
    def AddEpochAlwaysFalse(self, *arg, **kw):
        cache = self.cache
        cache.EpochIndex += 1
        cache.BatchIndex = 0
        return False, None
    def NotifyEndOfEpochAlwaysTrue(self, **kw):
        return True, self.GetMethod()
    def NotifyEndOfEpochAlwaysFalse(self, **kw):
        return False, None

#CheckPointForEpochBatchTrain.IsCheckPoint = True
#DLUtils.transform.SetEpochBatchMethodForModule(CheckPointForEpochBatchTrain)