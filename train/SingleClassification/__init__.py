
import DLUtils
from ..EpochBatchTrain.Component import EvaluationLog
import queue

class EvaluationLogSingleClassification(EvaluationLog):
    def BindTestSession(self, TestSession):
        Param = self.Param
        if Param.OnlineMonitor.Enable:
            OnlineMonitor = DLUtils.train.EpochBatchTrain.EventAfterFixedBatch(
                BatchInterval=self.OnlineMonitorBatchNum,
                Event=self.OnlineReport
            ).Init()
            # OnlineMonitor.AfterEpoch = lambda self, Dict: Dict.TestSession.RemoveSubModule(self)
            OnlineMonitor.AfterEpoch = lambda Dict: Dict.TestSession.RemoveSubModule("OnlineMonitor")
            TestSession.Bind(OnlineMonitor=OnlineMonitor)
        return self
    def BindTrainSession(self, TrainSession):
        Param = self.Param
        if Param.OnlineMonitor.Enable:
            OnlineMonitor = DLUtils.train.EpochBatchTrain.EventAfterFixedBatch(
                BatchInterval=self.OnlineMonitorBatchNum,
                Event=self.OnlineReport
            )
            TrainSession.Bind(OnlineMonitor=OnlineMonitor)
            OnlineMonitor.BindTrainSession(TrainSession)
        return super().BindTrainSession(TrainSession)
    def OnlineReport(self, Dict):
        RateCorrect = self.OnlineMonitorNumCorrectList.Sum() / self.OnlineMonitorNumTotalList.Sum()
        if Dict.get("IsTest"):
            print("TestEpoch %3d Batch %3d RateCorrect:%.3f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
        else:
            print("Epoch %3d Batch %3d RateCorrect:%.3f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
    def BeforeTrain(self, Dict):
        Param = self.Param
        Param.Epochs.Test.setdefault("IndexList", [])
        Param.Epochs.Train.setdefault("IndexList", [])
    def AfterBatch(self, Dict):
        Evaluation = Dict.Evaluation
        EpochLog = self.EpochLog
        Num = Evaluation.NumTotal
        NumCorrect = Evaluation.NumCorrect
        EpochLog.NumTotal.append(Num)
        EpochLog.NumCorrect.append(NumCorrect)
        EpochLog.RateCorrectList.append(1.0 * NumCorrect / Num)
        EpochLog.LossList.append(Evaluation.Loss.item())
        self.BatchNum += 1
        self.OnlineMonitorNumCorrectList.append(NumCorrect)
        self.OnlineMonitorNumTotalList.append(Num)
        return self
    def EpochIndexList(self, IsTrain=True):
        if IsTrain:
            return self.Param.Epochs.Train.IndexList
        else:
            return self.Param.Epochs.Test.IndexList
    def CorrectRateEpoch(self, IsTrain=True):
        Param = self.Param
        if IsTrain:
            if Param.Epochs.Train.hasattr("RateCorrectList"):
                return Param.Epochs.Train.RateCorrect
            else:
                RateCorrectList = Param.Epochs.Train.RateCorrectList = []
                for EpochIndex in Param.Epochs.Train.IndexList:
                    RateCorrectList.append(Param.getattr("Epoch%d"%EpochIndex).RateCorrect)
            return list(RateCorrectList)
        else:
            if Param.Epochs.Test.hasattr("RateCorrectList"):
                return Param.Epochs.Test.RateCorrect
            else:
                RateCorrectList = Param.Epochs.Test.RateCorrectList = []
                for EpochIndex in Param.Epochs.Test.IndexList:
                    RateCorrectList.append(Param.getattr("TestEpoch%d"%EpochIndex).RateCorrect)
            return list(RateCorrectList)
    def LossEpoch(self, IsTrain=True):
        Param = self.Param
        if IsTrain:
            if Param.Epochs.Train.hasattr("LossList"):
                return Param.Epochs.Train.LossList
            else:
                LossList = Param.Epochs.Train.LossList = []
                for EpochIndex in Param.Epochs.Train.IndexList:
                    LossList.append(Param.getattr("Epoch%d"%EpochIndex).Loss)
            return LossList
        else:
            if Param.Epochs.Test.hasattr("LossList"):
                return Param.Epochs.Test.LossList
            else:
                LossList = Param.Epochs.Test.LossList = []
                for EpochIndex in Param.Epochs.Test.IndexList:
                    LossList.append(Param.getattr("TestEpoch%d"%EpochIndex).Loss)
            return LossList
    def BeforeEpoch(self, Dict, EpochLogName=None, IsTrain=True):
        Param = self.Param
        if IsTrain:
            Param.Epochs.Train.IndexList.append(Dict.EpochIndex)
        else:
            Param.Epochs.Test.IndexList.append(Dict.EpochIndex)
        if EpochLogName is None:
            EpochLog = Param.setemptyattr("Epoch%d"%Dict.EpochIndex)
        else:
            EpochLog = Param.setemptyattr(EpochLogName)
        EpochLog.Epoch.Index = Dict.EpochIndex
        EpochLog.NumCorrect = []
        EpochLog.NumTotal = []
        EpochLog.RateCorrectList = []
        EpochLog.LossList = []
        self.EpochLog = EpochLog
        self.BatchNum = 0
        return self
    def BeforeTestEpoch(self, Dict):
        self.BeforeEpoch(Dict, EpochLogName="TestEpoch%d"%Dict.EpochIndex, IsTrain=False)
        return self
    def AfterTestBatch(self, Dict):
        Dict.IsTest = True
        self.AfterBatch(Dict)
        return self
    def AfterEpoch(self, Dict):
        EpochLog = self.EpochLog
        EpochLog.BatchNum = self.BatchNum
        EpochLog.RateCorrect = 1.0 * sum(EpochLog.NumCorrect) / sum(EpochLog.NumTotal)
        EpochLog.Loss = 1.0 * sum(EpochLog.LossList) / len(EpochLog.LossList)
        return self
    def AfterTestEpoch(self, Dict):
        self.AfterEpoch(Dict)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.OnlineMonitor.setdefault("Enable", True)
        Param.OnlineMonitor.Batch.setdefault("Num", 50)
        self.OnlineMonitorBatchNum = Param.OnlineMonitor.Batch.Num
        self.OnlineMonitorNumTotalList = DLUtils.FixedSizeQueuePassiveOutInt32(self.OnlineMonitorBatchNum)
        self.OnlineMonitorNumCorrectList = DLUtils.FixedSizeQueuePassiveOutInt32(self.OnlineMonitorBatchNum)
        return super().Init(IsSuper=True, IsRoot=IsRoot)