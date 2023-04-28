
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
            print("TestEpoch %3d Batch %3d RateCorrect:%.4f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
        else:
            print("Epoch %3d Batch %3d RateCorrect:%.4f"%(Dict.EpochIndex, Dict.BatchIndex, RateCorrect))
    Event = OnlineReport

class OnlineReporterMultiLoss(EventAfterFixedBatch):
    def __init__(self, BatchInterval=None, NumPerEpoch=None):
        Dict = {}
        if BatchInterval is not None:
            Dict["BatchInterval"] = BatchInterval
        elif NumPerEpoch is not None:
            Dict["NumPerEpoch"] = NumPerEpoch
        else:
            raise Exception()
        super().__init__(BatchInterval)
        Param = self.Param
        Param.Log.setdefault("List", [])
        Param.Report.setdefault("List", [])
    def AddLogAndReportItem(self, WatchName=None, ReportName=None, LogName=None, Type="Float"):
        if LogName is None:
            LogName = WatchName
        if ReportName is None:
            ReportName = LogName
        self.AddLogItem(WatchName=WatchName, LogName=LogName, Type=Type)
        self.AddReportItem(LogName=LogName, ReportName=ReportName, Type=Type)
        return self
    def AddLogItem(self, LogName, WatchName=None, Type="Float"):
        Param = self.Param
        if WatchName is None:
            WatchName = LogName
        Param.Log.List.append(
            DLUtils.Param({
                "LogName": LogName, "WatchName": LogName, "Type": Type
            })
        )
        return self
    def AddReportItem(self, LogName, ReportName=None, Type="Float"):
        Param = self.Param
        assert isinstance(LogName, str) or isinstance(LogName, list) or isinstance(LogName, tuple)
        if ReportName is None:
            ReportName = LogName
        Param.Report.List.append(
            DLUtils.Param({
                "ReportName": ReportName, "LogName": LogName, "Type": Type
            })
        )
        return self
    def OnlineReport(self, Dict):
        if Dict.IsValidate:
            ReportStrList = ["TestEpoch %3d/%3d Batch %3d/%3d"%(Dict.TrainEpochIndex, Dict.TrainEpochNum, Dict.BatchIndex, Dict.BatchNum)]
        else:
            ReportStrList = ["Epoch %3d/%3d Batch %3d/%3d"%(Dict.EpochIndex, Dict.EpochNum, Dict.BatchIndex, Dict.BatchNum)]
        for Index in self.ReportIndexList:
            # Value = self.ReportFuncList[Index]()
            # ReportStrList.append("%s:%.3f"%(self.ReportNameList[Index], Value))
            ReportStrList.append("%s:%s"%(self.ReportNameList[Index], self.ReportStrFuncList[Index]()))
        ReportStr = " ".join(ReportStrList)
        print(ReportStr)
        return ReportStr
    def _ReportFloat(self, Log):
        return Log.Average()
    def _ReportFloatStr(self, Log):
        return "%.4e"%(Log.Average())
    def _ReportAcc(self, LogCorrect, LogTotal):
        return LogCorrect.Sum() / LogTotal.Sum()
    def _ReportAccStr(self, LogCorrect, LogTotal):
        NumCorrect = LogCorrect.Sum()
        NumTotal = LogTotal.Sum()
        return "%.4f(%d/%d)"%(NumCorrect / NumTotal, NumCorrect, NumTotal)
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.ReportBatchInterval = Param.setdefault("Batch.Interval", 50)
        # set log item
        self.LogItemDict = {}
        self.LogNameList = []
        for LogItem in Param.Log.List:
            LogName = LogItem.LogName
            WatchName = LogItem.WatchName
            Log = DLUtils.FixedSizeQueuePassiveOutFloat(self.ReportBatchInterval)
            self.LogItemDict[LogName] = Log
            self.LogNameList.append((WatchName, LogName))
        
        # set report item
        self.ReportFuncList = []
        self.ReportNameList = []
        self.ReportStrFuncList = []
        self.ReportValue2StrFuncList = []

        # report items
        Param.Report.Num = len(Param.Report.List)
        self.ReportIndexList = range(Param.Report.Num)
        for ReportItem in Param.Report.List:
            if ReportItem.Type in ["Float"]:
                Log = self.LogItemDict[ReportItem.LogName]
                self.ReportStrFuncList.append(
                    functools.partial(self._ReportFloatStr, Log=Log)
                )
                # self.ReportValue2StrFuncList.append(lambda Float:"%.4e"%Float)
            elif ReportItem.Type in ["Acc", "RateCorrect", "AccTop1"]:
                ReportNameCorrect = self.LogItemDict[ReportItem.LogName[0]]
                ReportNameTotal = self.LogItemDict[ReportItem.LogName[1]]
                # self.ReportFuncList.append(
                #     functools.partial(self._ReportAcc, ReportNameCorrect, ReportNameTotal)
                # )
                self.ReportStrFuncList.append(
                    functools.partial(self._ReportAccStr, ReportNameCorrect, ReportNameTotal)
                )
                # self.ReportValue2StrFuncList.append(lambda RateAccurate:"%.4f"%RateAccurate)
            
            else:
                raise Exception()
            self.ReportNameList.append(ReportItem.ReportName)
        
        self.Event = self.OnlineReport
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def AfterBatch(self, Dict):
        self.LogAfterBatch(Dict)
        super().AfterBatchSuper(Dict)
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
        NumTotal = Evaluation.NumTotal
        NumCorrect = Evaluation.NumCorrect
        EpochLog.NumTotal.append(NumTotal)
        EpochLog.NumCorrect.append(NumCorrect)
        EpochLog.RateCorrectList.append(1.0 * NumCorrect / NumTotal)
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