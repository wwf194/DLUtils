

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
            "EpochIndex": Dict.EpochIndex
        })

        EvaluationLog.BeforeTestEpoch(DictTest)
        for TestBatchIndex in range(BatchNum):
            Input, OutputTarget = TestData.Get(TestBatchIndex)
            DLUtils.NpArray2Str
            Output = Model(Input)
            DictTest.Input = Input
            DictTest.Output = Output
            DictTest.OutputTarget = OutputTarget
            Evaluation = Evaluator.Evaluate(DictTest)
            DictTest.Evaluation = Evaluation
            DictTest.BatchIndex = TestBatchIndex
            EvaluationLog.AfterTestBatch(DictTest)
        EvaluationLog.AfterTestEpoch(DictTest)
        return self

class Save(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        self.Event = self.Save # to be called
    def SetParam(self, **Dict):
        Param = self.Param
        SaveDir = Dict.get("SaveDir")
        if SaveDir is not None:
            Param.SaveDir = SaveDir
        super().SetParam(**Dict)
        return self
    def Save(self, Dict):
        ModelSaveDir = self.SaveDir + "model/" + f"model-Epoch{Dict.EpochIndex}-Batch{Dict.BatchIndex}.dat"
        Dict.Model.ToFile(ModelSaveDir, RetainSelf=False)
        Dict.Model.Clear()
        Dict.Model = Dict.Model.FromFile(ModelSaveDir).Init()
        Dict.TrainSession.Model = Dict.Model
        Dict.TrainSession.ToFile(self.SaveDir)
        Dict.TrainSession.SetDevice()
        print("Save on Epoch %3d Batch %3d"%(Dict.EpochIndex, Dict.BatchIndex))
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.SaveDir = Param.setdefault("SaveDir", "./TrainSession/")
        return super().Init(IsSuper=False, IsRoot=IsRoot)

class AfterTrainAnalysis(EpochBatchTrainComponent):
    def AfterTrain(self, Dict):
        self.PlotRateCorrect(Dict)
    def PlotRateCorrect(self, Dict):
        TrainLog = self.TrainLog
        EpochIndex = Dict.EpochIndex
        BatchIndex = Dict.BatchIndex
        EvaluationLog = Dict.EvaluationLog
        EpochIndexFloat = DLUtils.train.EpochBatchIndices2EpochsFloat(
            EpochIndex, BatchIndex
        )
        DLUtils.plot.PlotLineChart(
            Xs = EpochIndexFloat, Ys = TrainLog.RateCorrect,
            XLabel="Epoch", YLabel="Correct Rate",
            SavePath=self.GetSaveDir() + "Epoch ~ Correct Rate"
        )
        return self

class XFixedSizeYFixedSizeProb(EpochBatchTrainComponent):
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