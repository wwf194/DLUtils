
from ..EpochBatchTrain.Component import EvaluationLog

class EvaluationLogSingleClassification(EvaluationLog):
    def AfterBatch(self, Dict):
        Evaluation = Dict.Evaluation
        EpochLog = self.EpochLog
        EpochLog.NumTotal.append(Evaluation.NumTotal)
        EpochLog.NumCorrect.append(Evaluation.NumCorrect)
        EpochLog.Loss.append(Evaluation.Loss.item())
        RateCorrect = 1.0 * Evaluation.NumCorrect / Evaluation.NumTotal
        EpochLog.RateCorrect.append(RateCorrect)
        self.BatchNum += 1
        if Dict.get("IsTest"):
            print("TestEpoch %3d Batch %3d RateCorrect:%.3f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
        else:
            print("Epoch %3d Batch %3d RateCorrect:%.3f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
        return self
    def BeforeEpoch(self, Dict):
        Param = self.Param
        EpochLog = Param.setemptyattr("Epoch%d"%Dict.EpochIndex)
        EpochLog.Epoch.Index = Dict.EpochIndex
        EpochLog.NumCorrect = []
        EpochLog.NumTotal = []
        EpochLog.RateCorrect = []
        self.EpochLog = EpochLog
        self.BatchNum = 0
        return self
    def AfterEpoch(self, Dict):
        self.EpochLog.BatchNum = self.BatchNum
        return self
    def AfterTestBatch(self, Dict):
        Dict.IsTest = True
        self.AfterBatch(Dict)
        return self
    def BeforeTestEpoch(self, Dict):
        Param = self.Param
        EpochLog = Param.Test.Epoch.setemptyattr(Dict.EpochIndex)
        EpochLog.NumCorrect = []
        EpochLog.NumTotal = []
        EpochLog.RateCorrect = []
        self.EpochLog = EpochLog
        self.BatchNum = 0
        return self
    def AfterTestEpoch(self, Dict):
        self.AfterEpoch(Dict)
        return self