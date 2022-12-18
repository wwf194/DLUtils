import DLUtils
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

class XFixedSizeYFixedSizeProb:
    def BeforeTrain(self):
        Param = self.Param
        TrainLog = DLUtils.param({
            "Type": "TrainProcess",
            "EpochIndex":[], "BatchIndexIndex": [],
            "NumTotal": [], # BatchSize
            "NumCorrect": [], 
            "RatioCorrect": [] # AccuracyRate
        })
        Param.Log.append(
            TrainLog
        )
        self.TrainLog = TrainLog

        Model = self.Model
        self.TrainParam = Model.ExtractTrainParam()
    def AfterBatch(self, EpochIndex, BatchIndex):
        TrainLog = self.TrainLog
        TrainLog.EpochIndex.append(EpochIndex)
        TrainLog.BatchIndex.append(BatchIndex)
        TrainLog.NumCorrect.append(self.NumCorrect)
        TrainLog.NumTotal.append(self.NumTotal)
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
        self.NumTotal, self.NumCorrect = RatioCorrectSingelClassPrediction(Output, OutputTarget)
    def BeforeTrain(self):
        self.AddLog(f"Before Train. {DLUtils.system.Time()}")

def LogAccuracyForSingleClassPrediction(Accuracy, Output, OutputTarget):
    # Output: np.ndarray. Predicted class indices in shape of [BatchNum]
    # OutputTarget: np.ndarray. Ground Truth class indices in shape of [BatchNum]
    NumCorrect, NumTotal = RatioCorrectSingelClassPrediction(Output, OutputTarget)
    Accuracy.NumTotal += NumTotal
    Accuracy.NumCorrect += NumCorrect

def RatioCorrectSingelClassPrediction(ScorePredicted, ScoreTruth):
    NumTotal = ScorePredicted.shape[0]
    IndexPredicted = ScorePredicted.argmax(dim=1)
    IndexTruth = ScoreTruth.argmax(dim=1)
    NumCorrect = np.sum(IndexPredicted==IndexTruth)
    return NumTotal, NumCorrect