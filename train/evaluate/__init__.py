import DLUtils
import torch
import numpy as np

def Evaluator(Type):
    if Type in ["ImageClassification"]:
        return XFixedSizeYFixedSizeProb()
    else:
        raise Exception()

class _Evaluator:
    def __init__(self):
        self.Param = DLUtils.Param({})
        Param = self.Param
        Param.Log = DLUtils.Param([])

class EpochBatchTrainComponent:
    def __init__(self):
        self.Param = DLUtils.Param()
        Param = self.Param
        Param.Data.Log = DLUtils.Param([])
    def BindTrainProcess(self, TrainProcess):
        assert isinstance(TrainProcess, DLUtils.train.EpochBatchTrainProcess)
        if hasattr(self, "BeforeTrain"):
            TrainProcess.AddBeforeTrainEvent(self.BeforeTrain)
        if hasattr(self, "BeforeEpoch"):
            TrainProcess.AddBeforeEpochEvent(self.BeforeEpoch)
        if hasattr(self, "BeforeBatch"):
            TrainProcess.AddBeforeBatchEvent(self.BeforeBatch)
        if hasattr(self, "AfterBatch"):
            TrainProcess.AddAfterBatchEvent(self.AfterBatch)
        if hasattr(self, "AfterEpoch"):
            TrainProcess.AddAfterEpochEvent(self.AfterEpoch)
        if hasattr(self, "AfterTrain"):
            TrainProcess.AddAfterTrainEvent(self.AfterTrain)
    def AddLog(self, Content):
        Param = self.Param
        Param.Data.Log.append(Content)

class XFixedSizeYFixedSizeProb(EpochBatchTrainComponent):
    def BeforeTrain(self):
        self.AddLog(f"Before Train. {DLUtils.system.Time()}")
        Param = self.Param
        TrainLog = DLUtils.param({
            "Type": "TrainProcess",
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
    def AfterBatch(self, EpochIndex, BatchIndex):
        TrainLog = self.TrainLog
        TrainLog.EpochIndex.append(EpochIndex)
        TrainLog.BatchIndex.append(BatchIndex)
        TrainLog.NumCorrect.append(self.NumCorrect)
        TrainLog.NumTotal.append(self.NumTotal)
        RateCorrect = 1.0 * self.NumCorrect / self.NumTotal
        TrainLog.RateCorrect =RateCorrect
        print("Epoch %3d Batch %3d RateCorrect:%.3f"%(EpochIndex, BatchIndex, RateCorrect))
        return self
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
    def Evaluate(self, Input, Output, OutputTarget, Model):
        self.Loss = self.LossModule(Output=Output, OutputTarget=OutputTarget)
        self.NumTotal, self.NumCorrect = RateCorrectSingelClassPrediction(Output, OutputTarget)

    def AfterTrain(self):
        self.AddLog(f"Ater Train.")
        self.PlotRateCorrect()
    def PlotRateCorrect(self):
        TrainLog = self.TrainLog
        EpochIndex = TrainLog.EpochIndex
        BatchIndex = TrainLog.BatchIndex
        EpochIndexFloat = DLUtils.train.EpochBatchIndices2EpochsFloat(
            EpochIndex, BatchIndex
        )
        DLUtils.plot.PlotLineChart(
            Xs = EpochIndexFloat, Ys = TrainLog.RateCorrect,
            XLabel="Epoch", YLabel="Correct Rate",
            SavePath=self.GetSaveDir() + "Epoch ~ Correct Rate"
        )
        return self

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