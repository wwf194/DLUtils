

import DLUtils
import torch

from . import EpochBatchTrainComponent
from . import EventAfterEpoch
from . import EventAfterEveryBatch

class EvaluationLog(EventAfterEveryBatch):
    def __init__(self):
        super().__init__()
        Param = self.Param
        if hasattr(self, "AfterBatch"):
            pass
        else:
            self.Event = self.Log

class Test(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
        Param._CLASS = "DLUtils.train.EpochBatchTrain.Test"
        self.Event = self.Test # to be called

        

    def Test(self, Dict):
        TestData = Dict.TestData
        Evaluator = Dict.Evaluator
        Model = Dict.Model
        EvaluationLog = Dict.EvaluationLog
        BatchNum = TestData.BatchNum()
        EvaluationLog.BeforeTestEpoch(Dict)
        DictTest = DLUtils.param({
            "Model": Model,
            "BatchNum": BatchNum,
            "EpochIndex": Dict.EpochIndex,
            "TestSession": self
        })
        EvaluationLog.BindTestSession(self)
        EvaluationLog.BeforeTestEpoch(DictTest)
        self.BeforeTest(DictTest)
        self.BeforeTestEpoch(DictTest)
        for TestBatchIndex in range(BatchNum):
            DictTest.BatchIndex = TestBatchIndex
            self.BeforeTestBatch(DictTest)
            Input, OutputTarget = TestData.Get(TestBatchIndex)
            DLUtils.NpArray2Str
            Output = Model(Input)
            DictTest.Input = Input
            DictTest.Output = Output
            DictTest.OutputTarget = OutputTarget
            Evaluation = Evaluator.Evaluate(DictTest)
            DictTest.Evaluation = Evaluation
            self.AfterTestBatch(DictTest)
            EvaluationLog.AfterTestBatch(DictTest)
        EvaluationLog.AfterTestEpoch(DictTest)
        self.AfterTestEpoch(DictTest)
        return self
    def Bind(self, **Dict):
        for Name, SubModule in Dict.items():
            self.AddSubModule(Name, SubModule)
            if hasattr(SubModule, "BeforeTrain"):
                self.AddBeforeTestEvent(SubModule.BeforeTrain)
            if hasattr(SubModule, "AfterBatch"):
                self.AddAfterTestBatchEvent(SubModule.AfterBatch)
            if hasattr(SubModule, "AfterEpoch"):
                self.AddAfterTestEpochEvent(SubModule.AfterEpoch)
    def AddBeforeTestEvent(self, Event):
        self.BeforeTestEventList.append(Event)
        return self
    def AddAfterTestBatchEvent(self, Event):
        self.AfterTestBatchEventList.append(Event)
        return self
    def AddAfterTestBatchEvent(self, Event):
        self.AfterTestBatchEventList.append(Event)
        return self
    def AddAfterTestEpochEvent(self, Event):
        self.AfterTestEpochEventList.append(Event)
        return self
    def BeforeTest(self, Dict):
        for Event in self.BeforeTestEventList:
            Event(Dict)
        return self
    def BeforeTestEpoch(self, Dict):
        for Event in self.BeforeTestEpochEventList:
            Event(Dict)
        return self
    def BeforeTestBatch(self, Dict):
        for Event in self.BeforeTestBatchEventList:
            Event(Dict)
        return self
    def AfterTestBatch(self, Dict):
        for Event in self.AfterTestBatchEventList:
            Event(Dict)
        return self
    def AfterTestEpoch(self, Dict):
        for Event in self.AfterTestEpochEventList:
            Event(Dict)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        self.BeforeTestEventList = []
        self.BeforeTestEpochEventList = []
        self.BeforeTestBatchEventList = []
        self.AfterTestBatchEventList = []
        self.AfterTestEpochEventList = []
        self.AfterTestEventList = []

        return super().Init(IsSuper=True, IsRoot=IsRoot)
class Save(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
        Param._CLASS = "DLUtils.train.EpochBatchTrain.Save"
        self.Event = self.Save # to be called
    def SetParam(self, **Dict):
        Param = self.Param
        SaveDir = Dict.get("SaveDir")
        if SaveDir is not None:
            Param.SaveDir = SaveDir
        super().SetParam(**Dict)
        return self
    def Save(self, Dict, ModelSaveFilePath=None):
        if ModelSaveFilePath is None:
            ModelSaveFilePath = self.SaveDir + "model/" + f"model-Epoch{Dict.EpochIndex}.dat"
        Dict.Model.ToFile(ModelSaveFilePath, RetainSelf=False)
        Dict.Model.Clear()
        Dict.Model = Dict.Model.FromFile(ModelSaveFilePath).Init()
        Dict.TrainSession.Model = Dict.Model
        Dict.TrainSession.ToFile(self.SaveDir)
        Dict.TrainSession.SetConnectEvents()
        Dict.TrainSession.SetDevice()
        print("saving model. Epoch {0:>3} Batch {1:>3}. FilePath:{2}".format(Dict.EpochIndex, Dict.BatchIndex, ModelSaveFilePath))
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.SaveDir = Param.setdefault("SaveDir", "./TrainSession/")
        Param.Save.setdefault("BeforeTrain", True)
        Param.Save.setdefault("AfterTrain", True)
        return super().Init(IsSuper=False, IsRoot=IsRoot)
    def BeforeTrain(self, Dict):
        Param = self.Param
        super().BeforeTrain(Dict)
        if Param.Save.Before.Train:
            Dict.EpochIndex = "0(BeforeTrain)"
            Dict.BatchIndex = -1
            self.Save(Dict)
        return self
    def AfterTrain(self, Dict):
        Param = self.Param
        ModelSaveFilePath = f"model-Epoch{Dict.EpochIndex}(AfterTrain).dat"
        ModelSaveDir = self.SaveDir + "model/"
        ModelSaveFileNameLast = f"model-Epoch{Dict.EpochIndex}.dat"
        if Dict.EpochIndex == self.EventEpochIndexList[-1] and DLUtils.FileExists(ModelSaveDir + ModelSaveFileNameLast): # Already saved after train
            DLUtils.file.RenameFile(ModelSaveDir, ModelSaveFileNameLast, ModelSaveFilePath)
        else:
            self.Save(Dict, self.SaveDir + "model/" + ModelSaveFilePath)
        return self

class XFixedSizeYFixedSizeProb(EpochBatchTrainComponent):
    def __init__(self):
        super().__init__()
        Param = self.Param
        Param._CLASS = "DLUtils.train.EpochBatchTrain.XFixedSizeYFixedSizeProb"
    def BeforeTrain(self, Dict):
        self.Log(f"Before Train. {DLUtils.system.Time()}")
        Param = self.Param
        TrainLog = DLUtils.param({
            "Type": "TrainSession",
            "EpochIndex":[], "BatchIndex": [],
            "NumTotal": [], # BatchSize
            "NumCorrect": [],
            "RateCorrect": [] # AccuracyRate
        })
        Param.Log.append(
            TrainLog
        )
        self.TrainLog = TrainLog
        # Model = self.Model
        # self.TrainParam = Model.ExtractTrainParam()
    # def AfterBatch(self, EpochIndex, BatchIndex):
    #     TrainLog = self.TrainLog
    #     TrainLog.EpochIndex.append(EpochIndex)
    #     TrainLog.BatchIndex.append(BatchIndex)
    #     TrainLog.NumCorrect.append(self.NumCorrect)
    #     TrainLog.NumTotal.append(self.NumTotal)
        
    #     TrainLog.RateCorrect.append(RateCorrect)
        
    #     return self
    # def Optimize(self, Input=None, OutputTarget=None, Model=None, Evaluation=None):
    #     Model.ClearGrad()
    #     Evaluation.Loss.backward()
    #     Output = Model(Input)
    #     Loss = self.LossModule(Output, OutputTarget)
    #     self.optimizer.UpdateParam()
    #     return self
    def SetLoss(self, LossModule, *List, **Dict):
        if isinstance(LossModule, str):
            LossModule = DLUtils.Loss(LossModule, *List, **Dict)
        self.LossModule = LossModule
        return self
    def Evaluate(self, Dict):
        Loss = self.LossModule(Output=Dict.Output, OutputTarget=Dict.OutputTarget)
        NumTotal, NumCorrect = RateCorrectSingelClassPrediction(Dict.Output, Dict.OutputTarget)
        self.Loss = Loss
        self.NumTotal = NumTotal
        self.NumCorrect = NumCorrect
        Evaluation = DLUtils.param({
            "Loss": Loss, "NumTotal": NumTotal, "NumCorrect": NumCorrect
        })
        return Evaluation
    def AfterTrain(self, Dict):
        self.Log(f"Ater Train.")

def LogAccuracyForSingleClassPrediction(Accuracy, Output, OutputTarget):
    # Output: np.ndarray. Predicted class indices in shape of [BatchNum]
    # OutputTarget: np.ndarray. Ground Truth class indices in shape of [BatchNum]
    NumCorrect, NumTotal = RateCorrectSingelClassPrediction(Output, OutputTarget)
    Accuracy.NumTotal += NumTotal
    Accuracy.NumCorrect += NumCorrect

def RateCorrectSingelClassPrediction(ScorePredicted, IndexTruth):
    NumTotal = ScorePredicted.shape[0]
    IndexPredicted = ScorePredicted.argmax(dim=1)
    #IndexTruth = ScoreTruth.argmax(dim=1)
    NumCorrect = torch.sum(IndexPredicted==IndexTruth).item()
    return NumTotal, NumCorrect

class AnalysisAfterTrain(EpochBatchTrainComponent):
    def __init__(self, SaveDir=None):
        super().__init__()
        Param = self.Param
        #Param._CLASS = "DLUtils.train.EpochBatchTrain.AfterTrainAnalysis"
        if SaveDir is not None:
            self.SaveDir = SaveDir
    def AfterTrain(self, Dict):
        PlotRateCorrect(Dict, self.SaveDir)
        return self

def PlotRateCorrect(Dict, SaveDir=None):
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    TrainSession = Dict.TrainSession
    Param = TrainSession.Param
    EvaluationLog = TrainSession.EvaluationLog
    #EpochIndexFloat = TrainSession.BatchIndexFloatList()
    #EpochIndexList = TrainSession.EpochIndexList
    XsTrain = EvaluationLog.EpochIndexList(IsTrain=True)
    XsTest = EvaluationLog.EpochIndexList(IsTrain=False)
    YsTrain = EvaluationLog.CorrectRateEpoch(IsTrain=True)
    YsTest = EvaluationLog.CorrectRateEpoch(IsTrain=False)
    DLUtils.plot.PlotMultiLineChart(
        Xs = [XsTrain, XsTest],
        Ys = [YsTrain, YsTest],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Correct Rate",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "test"],
        SavePath=SaveDir + "Epoch ~ Correct Rate.svg"
    )