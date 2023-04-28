import numpy as np

import DLUtils
import torch

from . import EpochBatchTrainComponent
from . import EventAfterEpoch
from . import EventAfterEveryBatch

class EvaluationLog(EventAfterEveryBatch):
    def __init__(self):
        super().__init__()
        if hasattr(self, "AfterBatch"):
            pass
        else:
            self.Event = self.Log
    def BindValSession(self, ValSession):
        Param = self.Param
        # if Param.OnlineMonitor.Enable:
        #     OnlineMonitor = DLUtils.train.EpochBatchTrain.EventAfterFixedBatch(
        #         BatchInterval=self.OnlineMonitorBatchNum,
        #         Event=self.OnlineReport
        #     ).Init()
        #     # OnlineMonitor.AfterEpoch = lambda self, Dict: Dict.ValSession.RemoveSubModule(self)
        #     OnlineMonitor.AfterEpoch = lambda Dict: Dict.ValSession.RemoveSubModule("OnlineMonitor")
        #     ValSession.Bind(OnlineMonitor=OnlineMonitor)
        return self
    def BindTrainSession(self, TrainSession):
        Param = self.Param
        # if Param.OnlineMonitor.Enable:
        #     OnlineMonitor = DLUtils.train.EpochBatchTrain.EventAfterFixedBatch(
        #         BatchInterval=self.OnlineMonitorBatchNum,
        #         Event=self.OnlineReport
        #     )
        #     TrainSession.Bind(OnlineMonitor=OnlineMonitor)
        #     OnlineMonitor.BindTrainSession(TrainSession)
        return super().BindTrainSession(TrainSession)
    def BeforeTrain(self, Dict):
        Param = self.Param
        Param.Epochs.Val.setdefault("IndexList", [])
        Param.Epochs.Train.setdefault("IndexList", [])
        return self
    def BeforeEpoch(self, Dict, EpochLogName=None, IsTrain=True):
        Param = self.Param
        if IsTrain:
            Param.Epochs.Train.IndexList.append(Dict.EpochIndex)
        else:
            Param.Epochs.Val.IndexList.append(Dict.EpochIndex)
        if EpochLogName is None:
            EpochLog = Param.setemptyattr("Epoch%d"%Dict.EpochIndex)
        else:
            EpochLog = Param.setemptyattr(EpochLogName)
        self.BatchNum = 0
        EpochLog.Epoch.Index = Dict.EpochIndex
        self.EpochLog = EpochLog
        return self
    def BeforeValEpoch(self, Dict):
        self.BeforeEpoch(Dict, EpochLogName="ValEpoch%d"%Dict.EpochIndex, IsTrain=False)
        return self
    def AfterValBatch(self, Dict):
        Dict.IsValidate = True
        self.AfterBatch(Dict)
        return self
    def AfterValEpoch(self, Dict):
        self.AfterEpoch(Dict)
        return self
    def AfterEpoch(self, Dict):
        self.EpochLog.Batch.Num = Dict.BatchSize
    def GetEpochLog(self, EpochIndex, IsTrain=True):
        Param = self.Param
        if IsTrain:
            EpochLog = Param.getattr("Epoch%d"%EpochIndex)
        else:
            EpochLog = Param.getattr("ValEpoch%d"%EpochIndex)
        return EpochLog
    def BatchIndexFloatList1Epoch(self, EpochIndex, IsTrain=True):
        Param = self.Param
        EpochLog = self.GetEpochLog(EpochIndex, IsTrain)
        if EpochLog.hasattr("Batch.Num"):
            BatchSize = EpochLog.Batch.Num
        else:
            BatchSize = len(EpochLog.LossList)
        BatchIndexList = np.asarray(range(BatchSize), dtype=np.float32)
        BatchIndexList = BatchIndexList / BatchSize
        BatchIndexList = BatchIndexList + 1.0 * EpochIndex
        return BatchIndexList
    BatchIndexFloatList = BatchIndexFloatInEpoch = BatchIndexFloatList1Epoch
    def LossList1Epoch(self, EpochIndex, IsTrain=True):
        Param = self.Param
        EpochLog = self.GetEpochLog(EpochIndex, IsTrain)
        return list(EpochLog.LossList)
    LossListInEpoch = LossList1Epoch
    def BatchIndexFloatAllEpoch(self, IsTrain=True):
        Param = self.Param
        BatchIndexListAllEpoch = []
        if IsTrain:
            EpochIndexList = Param.Epochs.Train.IndexList
        else:
            EpochIndexList = Param.Epochs.Val.IndexList
        for EpochIndex in EpochIndexList:
            BatchIndexList = self.BatchIndexFloatInEpoch(EpochIndex, IsTrain=IsTrain)
            BatchIndexListAllEpoch.append(BatchIndexList)
        BatchIndexList = np.concatenate(BatchIndexListAllEpoch, axis=0)
        return BatchIndexList
    def LossListEpoch(self, IsTrain=True):
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
            if Param.Epochs.Val.hasattr("LossList"):
                return Param.Epochs.Val.LossList
            else:
                LossList = Param.Epochs.Val.LossList = []
                for EpochIndex in Param.Epochs.Val.IndexList:
                    LossList.append(Param.getattr("ValEpoch%d"%EpochIndex).Loss)
            return LossList
    def LossListBatch(self, IsTrain=True):
        Param = self.Param
        BatchIndexListAllEpoch = []
        LossListAllEpoch = []
        if IsTrain:
            EpochIndexList = Param.Epochs.Train.IndexList
        else:
            EpochIndexList = Param.Epochs.Val.IndexList
        for EpochIndex in EpochIndexList:
            BatchIndexFloatList1Epoch = self.BatchIndexFloatList1Epoch(EpochIndex, IsTrain=IsTrain)
            LossList1Epoch = self.LossList1Epoch(EpochIndex, IsTrain=IsTrain)
            assert len(LossList1Epoch) == len(BatchIndexFloatList1Epoch)
            BatchIndexListAllEpoch.append(BatchIndexFloatList1Epoch)
            LossListAllEpoch.append(LossList1Epoch)
        BatchIndexListAllEpoch = np.concatenate(BatchIndexListAllEpoch, axis=0)
        LossListAllEpoch = np.concatenate(LossListAllEpoch, axis=0)
        return BatchIndexListAllEpoch, LossListAllEpoch
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        # Param.OnlineMonitor.setdefault("Enable", False)
        # if Param.OnlineMonitor.Enable:
        #     Param.OnlineMonitor.Batch.setdefault("Num", 50)
        #     self.OnlineMonitorBatchNum = Param.OnlineMonitor.Batch.Num
        #     self.OnlineMonitorNumTotalList = DLUtils.FixedSizeQueuePassiveOutInt(self.OnlineMonitorBatchNum)
        #     self.OnlineMonitorNumCorrectList = DLUtils.FixedSizeQueuePassiveOutInt(self.OnlineMonitorBatchNum)
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    
class Validate(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
        self.Event = self.ValidateEpoch # to be called
    def ValidateEpoch(self, Dict):
        ValidationData = Dict.ValidationData
        Evaluator = Dict.Evaluator
        Model = Dict.Model
        BatchNum = ValidationData.GetBatchNum()
        DictValidate = DLUtils.param({
            "Model": Model,
            "BatchNum": BatchNum,
            "EpochIndex": Dict.EpochIndex,
            "ValidationSession": self,
            "Session": self,
            "IsValidate": True,
            "TrainEpochIndex": Dict.EpochIndex,
            "TrainEpochNum": Dict.EpochNum
        })
        EvaluationLog = Dict.EvaluationLog
        EvaluationLog.BindValSession(self)
        self.Bind(EvaluationLog=EvaluationLog)
        self.BeforeVal(DictValidate)
        self.BeforeValEpoch(DictValidate)

        with torch.no_grad():
            # since no optimizer in validate epoch. grad needs to be be turned off.
            # otherwise cuda memory will explode without grad clear.
            for ValBatchIndex in range(BatchNum):
                DictValidate.BatchIndex = ValBatchIndex
                self.BeforeValBatch(DictValidate)
                In, OutTarget = ValidationData.Get(ValBatchIndex)
                Out = Model(In)
                DictValidate.In = In
                DictValidate.Out = Out
                DictValidate.OutTarget = OutTarget
                Evaluation = Evaluator.Evaluate(DictValidate)
                DictValidate.Evaluation = Evaluation
                self.AfterValBatch(DictValidate)
        self.AfterValEpoch(DictValidate)
        self.UnBind("EvaluationLog")
        return self
    Validate = ValidateEpoch
    def Bind(self, **Dict):
        for Name, SubModule in Dict.items():
            super().Bind(Name, SubModule)
        return self
    def RegisterEvent(self, Module):
        if hasattr(Module, "BeforeVal"):
            self.AddBeforeValEvent(Module.BeforeVal)
        elif hasattr(Module, "BeforeTrain"):
            self.AddBeforeValEvent(Module.BeforeTrain)

        if hasattr(Module, "BeforeValEpoch"):
            self.AddBeforeValEpochEvent(Module.BeforeValEpoch)
        elif hasattr(Module, "BeforeEpoch"):
            self.AddBeforeValEpochEvent(Module.BeforeEpoch)

        if hasattr(Module, "BeforeValBatch"):
            self.AddBeforeValBatchEvent(Module.BeforeValBatch)
        elif hasattr(Module, "BeforeBatch"):
            self.AddBeforeValBatchEvent(Module.BeforeBatch)
            
        if hasattr(Module, "AfterValBatch"):
            self.AddAfterValBatchEvent(Module.AfterValBatch)
        elif hasattr(Module, "AfterBatch"):
            self.AddAfterValBatchEvent(Module.AfterBatch)
        
        if hasattr(Module, "AfterValEpoch"):
            self.AddAfterValEpochEvent(Module.AfterValEpoch)
        elif hasattr(Module, "AfterEpoch"):
            self.AddAfterValEpochEvent(Module.AfterEpoch)

        if hasattr(Module, "AfterVal"):
            self.AddAfterValEvent(Module.AfterVal)
        elif hasattr(Module, "AfterTrain"):
            self.AddAfterValEvent(Module.AfterTrain)
        return self
    def AddBeforeValEvent(self, Event):
        self.BeforeValEventList.append(Event)
        return self
    def RemoveBeforeValEvent(self, Event):
        self.BeforeValEventList.remove(Event)
        return self
    def AddBeforeValEpochEvent(self, Event):
        self.BeforeValEpochEventList.append(Event)
        return self
    def RemoveBeforeValEpochEvent(self, Event):
        self.BeforeValEpochEventList.remove(Event)
        return self
    def AddBeforeValBatchEvent(self, Event):
        self.BeforeValBatchEventList.append(Event)
        return self
    def RemoveBeforeValBatchEvent(self, Event):
        self.BeforeValBatchEventList.remove(Event)
        return self
    def AddAfterValBatchEvent(self, Event):
        self.AfterValBatchEventList.append(Event)
        return self
    def RemoveAfterValBatchEvent(self, Event):
        self.AfterValBatchEventList.remove(Event)
        return self
    def AddAfterValEpochEvent(self, Event):
        self.AfterValEpochEventList.append(Event)
        return self
    def RemoveAfterValEpochEvent(self, Event):
        self.AfterValEpochEventList.remove(Event)
        return self
    def AddAfterValEvent(self, Event):
        self.AfterValEventList.append(Event)
        return self
    def RemoveAfterValEvent(self, Event):
        self.AfterValEventList.remove(Event)
        return self
    def BeforeVal(self, Dict):
        for Event in list(self.BeforeValEventList):
            Event(Dict)
        return self
    def BeforeValEpoch(self, Dict):
        for Event in list(self.BeforeValEpochEventList):
            Event(Dict)
        return self
    def BeforeValBatch(self, Dict):
        for Event in list(self.BeforeValBatchEventList):
            Event(Dict)
        return self
    def AfterValBatch(self, Dict):
        for Event in list(self.AfterValBatchEventList):
            Event(Dict)
        return self
    def AfterValEpoch(self, Dict):
        for Event in list(self.AfterValEpochEventList):
            Event(Dict)
        return self
    def AfterVal(self, Dict):
        for Event in list(self.AfterValEventList):
            Event(Dict)
        return self
    def RemoveSubModule(self, Name=None, SubModule=None):
        if SubModule is None:
            SubModule = self.GetSubModule(Name)

        if hasattr(SubModule, "BeforeVal"):
            self.RemoveBeforeValEvent(SubModule.BeforeVal)
        elif hasattr(SubModule, "BeforeTrain"):
            self.RemoveBeforeValEvent(SubModule.BeforeTrain)

        if hasattr(SubModule, "BeforeValEpoch"):
            self.RemoveBeforeValEpochEvent(SubModule.BeforeValEpoch)
        elif hasattr(SubModule, "BeforeEpoch"):
            self.RemoveBeforeValEpochEvent(SubModule.BeforeEpoch)

        if hasattr(SubModule, "BeforeValBatch"):
            self.RemoveBeforeValBatchEvent(SubModule.BeforeValBatch)
        elif hasattr(SubModule, "BeforeBatch"):
            self.RemoveBeforeValBatchEvent(SubModule.BeforeBatch)

        if hasattr(SubModule, "AfterValBatch"):
            self.RemoveAfterValBatchEvent(SubModule.AfterValBatch)
        elif hasattr(SubModule, "AfterBatch"):
            self.RemoveAfterValBatchEvent(SubModule.AfterBatch)

        if hasattr(SubModule, "AfterValEpoch"):
            self.RemoveAfterValEpochEvent(SubModule.AfterValEpoch)
        elif hasattr(SubModule, "AfterEpoch"):
            self.RemoveAfterValEpochEvent(SubModule.AfterEpoch)

        if hasattr(SubModule, "AfterVal"):
            self.RemoveAfterValEvent(SubModule.AfterValEpoch)
        elif hasattr(SubModule, "AfterTrain"):
            self.RemoveAfterValEvent(SubModule.AfterEpoch)

        return super().RemoveSubModule(Name=Name, SubModule=SubModule)
    def Init(self, IsSuper=False, IsRoot=True):
        self.BeforeValEventList = []
        self.BeforeValEpochEventList = []
        self.BeforeValBatchEventList = []
        self.AfterValBatchEventList = []
        self.AfterValEpochEventList = []
        self.AfterValEventList = []

        super().Init(IsSuper=True, IsRoot=IsRoot)
        
        # some module might prepare event function in Init.
        for ModuleName, Module in self.SubModules.items():
            self.RegisterEvent(Module)
        for ModuleName, Module in self.BindModules.items():
            self.RegisterEvent(Module)
    
        return self

class Save(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
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
            ModelSaveFilePath = self.SaveDir + "model-saved/" + f"model-Epoch{Dict.EpochIndex}.dat"
        Dict.Model.ToFile(ModelSaveFilePath, RetainSelf=False)
        Dict.Model.Clear()
        Dict.Model = Dict.Model.FromFile(ModelSaveFilePath).Init()
        Dict.TrainSession.Model = Dict.Model
        Dict.TrainSession.ToFile(self.SaveDir + "TrainSession.dat")
        Dict.TrainSession.ToJsonFile(self.SaveDir + "TrainSession-config.jsonc")
        print("Saving TrainSession. Epoch {0:>3} Batch {1:>3}. FilePath:{2}"
                .format(Dict.EpochIndex, Dict.BatchIndex, self.SaveDir + "TrainSession.dat"))
        Dict.TrainSession.SetConnectEvents()
        Dict.TrainSession.SetDevice()
        Dict.Optimizer.ResetOptimizer()
        print("Saving Model. Epoch {0:>3} Batch {1:>3}. FilePath:{2}".format(Dict.EpochIndex, Dict.BatchIndex, ModelSaveFilePath))
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        self.SaveDir = Param.setdefault("SaveDir", "./Val/")
        #assert hasattr(self, "SaveDir")
        Param.Save.setdefault("BeforeTrain", True)
        Param.Save.setdefault("AfterTrain", True)
        return super().Init(IsSuper=True, IsRoot=IsRoot)
    def BeforeTrain(self, Dict):
        Param = self.Param
        super().BeforeTrain(Dict)
        _Dict = DLUtils.param(Dict)
        if Param.Save.BeforeTrain:
            _Dict.EpochIndex = "0(BeforeTrain)"
            _Dict.BatchIndex = -1
            self.Save(_Dict)
        return self
    def AfterTrain(self, Dict):
        Param = self.Param
        ModelSaveDir = self.SaveDir + "model-saved/"
        ModelSaveFilePath = f"model-Epoch{Dict.EpochIndex}(AfterTrain).dat"
        ModelSaveFileNameLast = f"model-Epoch{Dict.EpochIndex}.dat"
        if Dict.EpochIndex == self.EventEpochIndexList[-1] and DLUtils.FileExists(ModelSaveDir + ModelSaveFileNameLast): # Already saved after train
            DLUtils.file.RenameFile(ModelSaveDir, ModelSaveFileNameLast, ModelSaveFilePath)
        else:
            self.Save(Dict, self.SaveDir + "model/" + ModelSaveFilePath)
        return self

class EvaluatorPredAndTarget(EpochBatchTrainComponent):
    def BeforeTrain(self, Dict):
        self.Log(f"Before Train. {DLUtils.system.Time()}")
    def Evaluate(self, Dict):
        # NumTotal, NumCorrect = RateCorrectSingelClassPrediction(Dict.Out["Out"], OutTarget)
        # self.NumTotal = NumTotal
        # self.NumCorrect = NumCorrect
        EvaluationDict = DLUtils.param({})
        for Func in self.EvaluateFuncList:
            Func(Dict, EvaluationDict)
        return EvaluationDict
    def _Evaluate1Loss(self, Dict):
        Loss = self.LossModule(Out=Dict.Out, OutTarget=Dict.OutTarget)["Loss"]
        self.Loss = Loss
        Evaluation = Loss
        return Evaluation
    def _EvaluateNLoss(self, Dict):
        Loss = self.LossModule(Out=Dict.Out, OutTarget=Dict.OutTarget)["Loss"]
        self.Loss = Loss
        Evaluation = Loss
        return Evaluation
    def AfterTrain(self, Dict):
        self.Log(f"After Train.")
    def EvaluateLoss(self, Dict, EvaluationDict):
        OutTarget = Dict.OutTarget
        Loss = self.LossModule(Out=Dict.Out, OutTarget=OutTarget)["Loss"]
        EvaluationDict["Loss"] = Loss
        self.Loss = Loss
        return self
    def AddEvaluationItem(self, **Dict):
        self.Param.EvaluateItemList.append(DLUtils.Param(Dict))
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not IsSuper: # subclass EvaluatorPredAndTargetSelect1FromN
            Type = Param.setdefault("Type", "1Loss")
            if Type in ["1Loss"]:
                self.Evaluate = self._Evaluate1Loss
            elif Type in ["NLoss"]:
                self.Evaluate = self._EvaluateNLoss
            else:
                raise Exception()
        self.EvaluateFuncList = []
        for Item in Param.Evaluate.ItemList:
            Type = Item.Type
            if Type in ["Loss"]:
                self.EvaluateFuncList.append(self.EvaluateLoss)
            elif Type in ["Acc"]:
                self.EvaluateFuncList.append(self.EvaluateAcc)
                self.Ks = Item.setdefault("TopK", [1, 5]) # default: calculate top1 and top5
                self.K2NumCorrectStr = {}
                for K in self.Ks:
                    if K == 1:
                        self.K2NumCorrectStr[K] = "NumCorrect"
                    else:
                        self.K2NumCorrectStr[K] = "NumCorrectTop%d"%K
            else:
                raise Exception()
        return super().Init(IsSuper=True, IsRoot=IsRoot)

class EvaluatorPredAndTargetSelect1FromN(EvaluatorPredAndTarget):
    def __init__(self):
        super().__init__()
        # Param = self.Param
        # Param._CLASS = "DLUtils.train.EpochBatchTrain.EvaluatorPredAndTargetSelect1FromN"
    def BeforeTrain(self, Dict):
        self.Log(f"Before Train. {DLUtils.system.Time()}")
        Param = self.Param
        TrainLog = DLUtils.param({
            "Type": "TrainSession",
            "EpochIndex":[], "BatchIndex": [],
            "NumTotal": [], # BatchSize
            "NumCorrect": [], # CorrectNum
            "RateCorrect": [] # AccuracyRate
        })
        Param.Log.append(
            TrainLog
        )
        self.TrainLog = TrainLog
    def Evaluate(self, Dict):
        # NumTotal, NumCorrect = RateCorrectSingelClassPrediction(Dict.Out["Out"], OutTarget)
        # self.NumTotal = NumTotal
        # self.NumCorrect = NumCorrect
        EvaluationDict = DLUtils.param({})
        for Func in self.EvaluateFuncList:
            Func(Dict, EvaluationDict)
        return EvaluationDict
    def EvaluateAcc(self, Dict, EvaluationDict):
        NumTotal, NumCorrectList = AccTopK(Dict.Out["Out"], Dict.OutTarget, Ks=self.Ks)
        EvaluationDict["NumTotal"] = NumTotal
        for Index, K in enumerate(self.Ks):
            EvaluationDict[self.K2NumCorrectStr[K]] = NumCorrectList[Index]
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if not Param.Evaluate.hasattr("ItemList"):
            Param.Evaluate.ItemList = DLUtils.Param([])
            Param.Evaluate.ItemList.append(
                DLUtils.Param({
                    "Type": "Loss"
                }))
            Param.Evaluate.ItemList.append(DLUtils.Param({
                "Type": "Acc",
                "TopK": (1, 5)
            }))
        return super().Init(IsSuper=True, IsRoot=IsRoot)

def LogAccuracyForSingleClassPrediction(Accuracy, Out, OutTarget):
    # Out: np.ndarray. Predicted class indices in shape of (BatchNum)
    # OutTarget: np.ndarray. Ground Truth class indices in shape of (BatchNum)
    NumCorrect, NumTotal = RateCorrectSingelClassPrediction(Out, OutTarget)
    Accuracy.NumTotal += NumTotal
    Accuracy.NumCorrect += NumCorrect

def RateCorrectSingelClassPrediction(ScorePredicted, IndexTruth):
    # IndexTruth: (BatchSize)
    NumTotal = ScorePredicted.shape[0]
    IndexPredicted = ScorePredicted.argmax(dim=1)
    # IndexTruth: (BatchSize, ClassNum)
    # IndexTruth = ScoreTruth.argmax(dim=1)
    NumCorrect = torch.sum(IndexPredicted==IndexTruth).item()
    return NumTotal, NumCorrect

def AccTopK(ScorePredicted, IndexTruth, Ks=(1,)):
    # TopK: support multiple k values provided in a tuple.
        # for example, TopK=(1, 5) will results in calculation of both Top1 and Top5.
    """Computes the precision@k for the specified values of k"""
    # IndexTruth: (BatchSize)
    KMax = max(Ks)
    BatchSize = IndexTruth.size(0)
    IndexTruth = IndexTruth.view(BatchSize, 1).expand(BatchSize, KMax) # (BatchSize, KMax)

    Value, TopKIndex = ScorePredicted.topk(KMax, 1, largest=True, sorted=True) # TopKIndex: (BatchSize, KMax)
    IndexCorrect = TopKIndex.eq(IndexTruth)
    NumTotal = BatchSize
    NumCorrectList = []
    for K in Ks:
        NumCorrectK = torch.sum(IndexCorrect[:, :K]).item()
        NumCorrectList.append(NumCorrectK)
        # NumTotalList.append(BatchSize)
    return NumTotal, NumCorrectList

import functools
AccTop5 = functools.partial(AccTopK, K=5)

class AnalysisAfterTrain(EpochBatchTrainComponent):
    def __init__(self, SaveDir=None):
        super().__init__()
        Param = self.Param
        #Param._CLASS = "DLUtils.train.EpochBatchTrain.AfterTrainAnalysis"
        if SaveDir is not None:
            self.SaveDir = SaveDir
    def AfterTrain(self, Dict):
        Param = self.Param
        ItemList = Param.Item.List
        TrainSession = Dict.TrainSession
        for Item in ItemList:
            if Item in ["RateCorrect", "Acc"]:
                PlotRateCorrect(TrainSession, Dict.SaveDir)
            elif Item in ["Loss", "LossEpoch"]:
                PlotLoss(TrainSession, Dict.SaveDir)
            elif Item in ["LossBatch"]:
                PlotLossBatch(TrainSession, Dict.SaveDir)
            else:
                raise Exception()
        return self
    def AddItem(self, ItemName):
        Param = self.Param
        Param.Item.setdefault("List", [])
        Param.Item.List.append(ItemName)
        return self
    
def PlotRateCorrect(TrainSession, SaveDir=None):
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    EvaluationLog = TrainSession.EvaluationLog
    # assert isinstance(EvaluationLog, DLUtils.train.Select1FromN.EvaluationLog)
    XsTrain = EvaluationLog.EpochIndexList(IsTrain=True)
    XsVal = EvaluationLog.EpochIndexList(IsTrain=False)
    YsTrain = EvaluationLog.CorrectRateEpoch(IsTrain=True)
    YsVal = EvaluationLog.CorrectRateEpoch(IsTrain=False)
    DLUtils.plot.PlotMultiLineChart(
        XsList = [XsTrain, XsVal],
        YsList = [YsTrain, YsVal],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Correct Rate",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "Val"],
        Title="Epoch - CorrectRate Relationship",
        SavePath=SaveDir + "Epoch ~ CorrectRate.svg"
    )

def PlotLoss(TrainSession, SaveDir=None):
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    EvaluationLog = TrainSession.EvaluationLog
    # assert isinstance(EvaluationLog, DLUtils.train.Select1FromN.EvaluationLog)
    XsTrain = EvaluationLog.EpochIndexList(IsTrain=True)
    XsVal = EvaluationLog.EpochIndexList(IsTrain=False)
    YsTrain = EvaluationLog.LossEpoch(IsTrain=True)
    YsVal = EvaluationLog.LossEpoch(IsTrain=False)
    DLUtils.plot.PlotMultiLineChart(
        XsList = [XsTrain, XsVal],
        YsList = [YsTrain, YsVal],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Loss",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "Val"],
        Title="Epoch ~ Loss Relationship",
        SavePath = SaveDir + "train-curve/" + "Epoch ~ Loss.svg"
    )

def PlotLossBatch(TrainSession, SaveDir=None):
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    EvaluationLog = TrainSession.EvaluationLog
    assert isinstance(EvaluationLog, DLUtils.train.Select1FromN.EvaluationLog)
    XsTrain, YsTrain = EvaluationLog.LossListBatch(IsTrain=True)
    XsVal, YsVal = EvaluationLog.LossListBatch(IsTrain=False)     

    DLUtils.plot.PlotMultiLineChart(
        XsList = [XsTrain, XsVal],
        YsList = [YsTrain, YsVal],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Loss",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "Val"],
        Title="Epoch ~ Loss Relationship",
        SavePath = SaveDir + "train-curve/" + "Epoch ~ Loss.svg"
    )

class DataFetcherForEpochBatchTrain(torch.utils.data.Dataset):
    def __init__(self):
        self.Device = "cpu"
        super().__init__()
    def __len__(self):
        # must be overwritten
        raise Exception()
    def __getitem__(self, Index):
        # must be overwritten
        raise Exception()
    def SetDevice(self, Device):
        self.Device = Device

class DataLoaderForEpochBatchTrain(torch.utils.data.DataLoader, DLUtils.module.AbstractModule):
    ParamMap = DLUtils.IterableKeyToElement({
        ("BatchSize"): "Batch.Size",
        ("BatchNum", "NumBatch"): "Batch.Num",
        ("DropLast"): "Batch.DropLast",
        ("Shuffle"): "Batch.Shuffle",
        ("ThreadNum"): "Thread.Num"
    })
    def __init__(self, DataFetcher=None, **Dict):
        self.DataFetcher = DataFetcher
        DLUtils.module.AbstractModule.__init__(self, **Dict)
        # DataFetcher: get sample input and output according to index.
        if DataFetcher is not None:
            self.SetDataFetcher(DataFetcher)

        torch.utils.data.DataLoader.__init__(
            self,
            dataset=self.DataFetcher,
            **self.GetTorchDataLoaderParam()
            # Setting num_workers > 1 might severely slow down speed.
        )

    def SetDataFetcher(self, DataFetcher):
        self.DataFetcher = DataFetcher
    def BeforeEpoch(self, Dict):
        self.Reset()
    def AfterEpoch(self, Dict):
        self.Reset()
    def GetBatch(self, BatchIndex=None):
        In, OutTarget = next(self.Iter)
        return In.to(self.Device), OutTarget.to(self.Device)
    Get = GetNextBatch = GetBatch
    # single device situation
    def SetDevice(self, Device, IsRoot=True):
        self.DataFetcher.SetDevice(Device)
        self.Device = Device
        return self
    def GetBatchNum(self):
        return self.BatchNum
    def Reset(self):
        self.Iter = iter(self)
        return self
    TorchDataLoaderParamMap = DLUtils.IterableKeyToElement({
        ("Thread.Num"): "num_workers",
        ("PinMemory"): "pin_memory",
        ("Batch.DropLast"): "drop_last",
        ("Batch.Size"): "batch_size",
        ("Batch.Shuffle"): "shuffle"
    })
    def GetTorchDataLoaderParam(self):
        Dict = {}
        Param = self.Param
        for NameInParam, NameForTorch in self.TorchDataLoaderParamMap.items():
            if Param.hasattr(NameInParam):
                Dict[NameForTorch] = Param.getattr(NameInParam)
            else:
                pass
        return Dict
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Param.Thread.setdefault("Num", 1)
        assert isinstance(Param.Thread.Num, int)
        assert hasattr(self, "DataFetcher")

        assert Param.Batch.hasattr("Size")
        self.BatchSize = Param.Batch.Size

        if not Param.Batch.hasattr("Num"):
            DataNum = self.DataFetcher.DataNum
            Param.Batch.Num = DataNum // self.BatchSize
            if DataNum % self.BatchSize > 0:
                Param.Batch.Num += 1 
        self.BatchNum = Param.Batch.Num

        # device setting
        if hasattr(self.DataFetcher, "Device"):
            self.Device = self.DataFetcher.Device

        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.Reset()
        self.Device = "cpu" # default
        return self