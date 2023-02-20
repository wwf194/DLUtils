
import DLUtils
from ..EpochBatchTrain.Component import EvaluationLog
import queue
import functools

class EvaluationLogLoss(EvaluationLog):
    def AfterBatch(self, Dict):
        Evaluation = Dict.Evaluation
        EpochLog = self.EpochLog
        EpochLog.LossList.append(Evaluation["Loss"].item())
        self.BatchNum += 1
        return self
    def BeforeEpoch(self, Dict, EpochLogName=None, IsTrain=True):
        super().BeforeEpoch(Dict=Dict, EpochLogName=EpochLogName, IsTrain=IsTrain)
        Param = self.Param
        EpochLog = self.EpochLog
        EpochLog.LossList = []
        self.EpochLog = EpochLog
        return self
    def AfterEpoch(self, Dict):
        EpochLog = self.EpochLog
        EpochLog.Loss = 1.0 * sum(EpochLog.LossList) / len(EpochLog.LossList)
        return self

from DLUtils.train.EpochBatchTrain import EventAfterFixedBatch
class OnlineReporterMultiLossAndAcc(EventAfterFixedBatch):
    def OnlineReport(self, Dict):
        RateCorrect = self.OnlineMonitorNumCorrectList.Sum() / self.OnlineMonitorNumTotalList.Sum()
        if Dict.get("IsTest"):
            print("TestEpoch %3d Batch %3d RateCorrect:%.3f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
        else:
            print("Epoch %3d Batch %3d RateCorrect:%.3f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
    Event = OnlineReport

class OnlineReporterMultiLoss(EventAfterFixedBatch):
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
        Param = self.Param
        Param.Log.setdefault("List", [])
        Param.Report.setdefault("List", [])
    def AddLogAndReportItem(self, Name, WatchName=None, ReportName=None, Type="Float"):
        self.AddLogItem(Name, WatchName, Type)
        self.AddReportItem(Name, ReportName, Type)
        return self
    def AddLogItem(self, Name, WatchName=None, Type="Float"):
        Param = self.Param
        if WatchName is None:
            WatchName = Name
        Param.Log.List.append(
            DLUtils.Param({
                "Name": Name, "WatchName": Name, "Type": Type
            })
        )
        return self
    def AddReportItem(self, Name, ReportName=None, Type="Float"):
        Param = self.Param
        assert isinstance(Name, str) or isinstance(Name, list)
        if ReportName is None:
            ReportName = Name
        Param.Report.List.append(
            DLUtils.Param({
                "Name": Name, "ReportName": ReportName, "Type": Type
            })
        )
        return self
    def OnlineReport(self, Dict):
        ReportStrList = ["Epoch %3d Batch %3d"%(Dict.EpochIndex, Dict.BatchIndex)]
        for Index in self.ReportIndexList:
            Value = self.ReportFuncList[Index]()
            ReportStrList.append("%s:%.3f"%(self.ReportNameList[Index], Value))
        ReportStr = " ".join(ReportStrList)
        print(ReportStr)
        return ReportStr
    def _ReportFloat(self, Log):
        return Log.Average()
    def _ReportAcc(self, LogCorrect, LogTotal):
        return LogCorrect.Sum() / LogTotal.Sum()
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.ReportBatchInterval = Param.setdefault("Batch.Interval", 50)
        # set log item
        self.LogItemDict = {}
        self.LogNameList = []
        for LogItem in Param.Log.List:
            LogName = LogItem.Name
            WatchName = LogItem.WatchName
            Log = DLUtils.FixedSizeQueuePassiveOutFloat(self.ReportBatchInterval)
            self.LogItemDict[LogName] = Log
            self.LogNameList.append((WatchName, LogName))
        # set report item
        self.ReportFuncList = []
        self.ReportNameList = []
        self.ReportValue2StrFuncList = []
        Param.Report.Num = len(Param.Report.List)
        self.ReportIndexList = range(Param.Report.Num)
        for ReportItem in Param.Report.List:
            if ReportItem.Type in ["Float"]:
                Log = self.LogItemDict[ReportItem.Name]
                self.ReportFuncList.append(
                    functools.partial(self._ReportFloat, Log=Log)
                )
            elif ReportItem.Type in ["Acc"]:
                ReportNameCorrect = LogItem.Name[0]
                ReportNameTotal = LogItem.Name[1]
                self.ReportFuncList.append(
                    functools.partial(self._ReportAcc, ReportNameCorrect, ReportNameTotal)
                )
                self.ReportValue2StrFuncList.append(self._Float2Str)
            else:
                raise Exception()
            self.ReportNameList.append(ReportItem.ReportName)
        self.Event = self.OnlineReport
        return super().Init(IsSuper=IsSuper, IsRoot=IsRoot)
    def AfterBatch(self, Dict):
        self.LogAfterBatch(Dict)
        super().AfterBatch(Dict)
        return self
    def LogAfterBatch(self, Dict):
        for WatchName, LogName in self.LogNameList:
            self.LogItemDict[WatchName].append(Dict.Evaluation[LogName])
        return self

class EvaluationLogSelect1FromN(EvaluationLog):
    def BeforeTrain(self, Dict):
        super().BeforeTrain(Dict)
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
        return self
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

    def BeforeEpoch(self, Dict, EpochLogName=None, IsTrain=True):
        super().BeforeEpoch(Dict=Dict, EpochLogName=EpochLogName, IsTrain=IsTrain)
        Param = self.Param
        EpochLog = self.EpochLog
        EpochLog.NumCorrect = []
        EpochLog.NumTotal = []
        EpochLog.RateCorrectList = []
        EpochLog.LossList = []
        return self
    def AfterEpoch(self, Dict):
        EpochLog = self.EpochLog
        EpochLog.BatchNum = self.BatchNum
        EpochLog.RateCorrect = 1.0 * sum(EpochLog.NumCorrect) / sum(EpochLog.NumTotal)
        EpochLog.Loss = 1.0 * sum(EpochLog.LossList) / len(EpochLog.LossList)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.OnlineMonitor.setdefault("Enable", True)
        Param.OnlineMonitor.Batch.setdefault("Num", 50)
        return super().Init(IsSuper=True, IsRoot=IsRoot)