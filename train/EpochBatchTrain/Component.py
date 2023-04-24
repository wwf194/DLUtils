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
    def BindTestSession(self, TestSession):
        Param = self.Param
        # if Param.OnlineMonitor.Enable:
        #     OnlineMonitor = DLUtils.train.EpochBatchTrain.EventAfterFixedBatch(
        #         BatchInterval=self.OnlineMonitorBatchNum,
        #         Event=self.OnlineReport
        #     ).Init()
        #     # OnlineMonitor.AfterEpoch = lambda self, Dict: Dict.TestSession.RemoveSubModule(self)
        #     OnlineMonitor.AfterEpoch = lambda Dict: Dict.TestSession.RemoveSubModule("OnlineMonitor")
        #     TestSession.Bind(OnlineMonitor=OnlineMonitor)
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
        Param.Epochs.Test.setdefault("IndexList", [])
        Param.Epochs.Train.setdefault("IndexList", [])
        return self
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
        self.BatchNum = 0
        EpochLog.Epoch.Index = Dict.EpochIndex
        self.EpochLog = EpochLog
        return self
    def BeforeTestEpoch(self, Dict):
        self.BeforeEpoch(Dict, EpochLogName="TestEpoch%d"%Dict.EpochIndex, IsTrain=False)
        return self
    def AfterTestBatch(self, Dict):
        Dict.IsTest = True
        self.AfterBatch(Dict)
        return self
    def AfterTestEpoch(self, Dict):
        self.AfterEpoch(Dict)
        return self
    def AfterEpoch(self, Dict):
        self.EpochLog.Batch.Num = Dict.BatchSize
    def GetEpochLog(self, EpochIndex, IsTrain=True):
        Param = self.Param
        if IsTrain:
            EpochLog = Param.getattr("Epoch%d"%EpochIndex)
        else:
            EpochLog = Param.getattr("TestEpoch%d"%EpochIndex)
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
            EpochIndexList = Param.Epochs.Test.IndexList
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
            if Param.Epochs.Test.hasattr("LossList"):
                return Param.Epochs.Test.LossList
            else:
                LossList = Param.Epochs.Test.LossList = []
                for EpochIndex in Param.Epochs.Test.IndexList:
                    LossList.append(Param.getattr("TestEpoch%d"%EpochIndex).Loss)
            return LossList
    def LossListBatch(self, IsTrain=True):
        Param = self.Param
        BatchIndexListAllEpoch = []
        LossListAllEpoch = []
        if IsTrain:
            EpochIndexList = Param.Epochs.Train.IndexList
        else:
            EpochIndexList = Param.Epochs.Test.IndexList
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
    
class Test(EventAfterEpoch):
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
        self.Event = self.Test # to be called
    def Test(self, Dict):
        TestData = Dict.TestData
        Evaluator = Dict.Evaluator
        Model = Dict.Model
        BatchNum = TestData.GetBatchNum()
        DictTest = DLUtils.param({
            "Model": Model,
            "BatchNum": BatchNum,
            "EpochIndex": Dict.EpochIndex,
            "TestSession": self
        })
        EvaluationLog = Dict.EvaluationLog
        # assert isinstance(EvaluationLog, DLUtils.train.Select1FromN.EvaluationLog)
        EvaluationLog.BindTestSession(self)
        self.Bind(EvaluationLog=EvaluationLog)
        self.BeforeTest(DictTest)
        self.BeforeTestEpoch(DictTest)
        for TestBatchIndex in range(BatchNum):
            DictTest.BatchIndex = TestBatchIndex
            self.BeforeTestBatch(DictTest)
            In, OutTarget = TestData.Get(TestBatchIndex)
            DLUtils.NpArray2Str
            Out = Model(In)
            DictTest.In = In
            DictTest.Out = Out
            DictTest.OutTarget = OutTarget
            Evaluation = Evaluator.Evaluate(DictTest)
            DictTest.Evaluation = Evaluation
            self.AfterTestBatch(DictTest)
        self.AfterTestEpoch(DictTest)
        self.UnBind("EvaluationLog")
        return self
    def Bind(self, **Dict):
        for Name, SubModule in Dict.items():
            super().Bind(Name, SubModule)
            if hasattr(SubModule, "BeforeTest"):
                self.AddBeforeTestEvent(SubModule.BeforeTest)
            elif hasattr(SubModule, "BeforeTrain"):
                self.AddBeforeTestEvent(SubModule.BeforeTrain)

            if hasattr(SubModule, "BeforeTestEpoch"):
                self.AddBeforeTestEpochEvent(SubModule.BeforeTestEpoch)
            elif hasattr(SubModule, "BeforeEpoch"):
                self.AddBeforeTestEpochEvent(SubModule.BeforeEpoch)

            if hasattr(SubModule, "BeforeTestBatch"):
                self.AddBeforeTestBatchEvent(SubModule.BeforeTestBatch)
            elif hasattr(SubModule, "BeforeBatch"):
                self.AddBeforeTestBatchEvent(SubModule.BeforeBatch)
                
            if hasattr(SubModule, "AfterTestBatch"):
                self.AddAfterTestBatchEvent(SubModule.AfterTestBatch)
            elif hasattr(SubModule, "AfterBatch"):
                self.AddAfterTestBatchEvent(SubModule.AfterBatch)
            
            if hasattr(SubModule, "AfterTestEpoch"):
                self.AddAfterTestEpochEvent(SubModule.AfterTestEpoch)
            elif hasattr(SubModule, "AfterEpoch"):
                self.AddAfterTestEpochEvent(SubModule.AfterEpoch)

            if hasattr(SubModule, "AfterTest"):
                self.AddAfterTestEvent(SubModule.AfterTest)
            elif hasattr(SubModule, "AfterTrain"):
                self.AddAfterTestEvent(SubModule.AfterTrain)
        return self
    def AddBeforeTestEvent(self, Event):
        self.BeforeTestEventList.append(Event)
        return self
    def RemoveBeforeTestEvent(self, Event):
        self.BeforeTestEventList.remove(Event)
        return self
    def AddBeforeTestEpochEvent(self, Event):
        self.BeforeTestEpochEventList.append(Event)
        return self
    def RemoveBeforeTestEpochEvent(self, Event):
        self.BeforeTestEpochEventList.remove(Event)
        return self
    def AddBeforeTestBatchEvent(self, Event):
        self.BeforeTestBatchEventList.append(Event)
        return self
    def RemoveBeforeTestBatchEvent(self, Event):
        self.BeforeTestBatchEventList.remove(Event)
        return self
    def AddAfterTestBatchEvent(self, Event):
        self.AfterTestBatchEventList.append(Event)
        return self
    def RemoveAfterTestBatchEvent(self, Event):
        self.AfterTestBatchEventList.remove(Event)
        return self
    def AddAfterTestEpochEvent(self, Event):
        self.AfterTestEpochEventList.append(Event)
        return self
    def RemoveAfterTestEpochEvent(self, Event):
        self.AfterTestEpochEventList.remove(Event)
        return self
    def AddAfterTestEvent(self, Event):
        self.AfterTestEventList.append(Event)
        return self
    def RemoveAfterTestEvent(self, Event):
        self.AfterTestEventList.remove(Event)
        return self
    def BeforeTest(self, Dict):
        for Event in list(self.BeforeTestEventList):
            Event(Dict)
        return self
    def BeforeTestEpoch(self, Dict):
        for Event in list(self.BeforeTestEpochEventList):
            Event(Dict)
        return self
    def BeforeTestBatch(self, Dict):
        for Event in list(self.BeforeTestBatchEventList):
            Event(Dict)
        return self
    def AfterTestBatch(self, Dict):
        for Event in list(self.AfterTestBatchEventList):
            Event(Dict)
        return self
    def AfterTestEpoch(self, Dict):
        for Event in list(self.AfterTestEpochEventList):
            Event(Dict)
        return self
    def AfterTest(self, Dict):
        for Event in list(self.AfterTestEventList):
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
    def RemoveSubModule(self, Name=None, SubModule=None):
        if SubModule is None:
            SubModule = self.GetSubModule(Name)

        if hasattr(SubModule, "BeforeTest"):
            self.RemoveBeforeTestEvent(SubModule.BeforeTest)
        elif hasattr(SubModule, "BeforeTrain"):
            self.RemoveBeforeTestEvent(SubModule.BeforeTrain)

        if hasattr(SubModule, "BeforeTestEpoch"):
            self.RemoveBeforeTestEpochEvent(SubModule.BeforeTestEpoch)
        elif hasattr(SubModule, "BeforeEpoch"):
            self.RemoveBeforeTestEpochEvent(SubModule.BeforeEpoch)

        if hasattr(SubModule, "BeforeTestBatch"):
            self.RemoveBeforeTestBatchEvent(SubModule.BeforeTestBatch)
        elif hasattr(SubModule, "BeforeBatch"):
            self.RemoveBeforeTestBatchEvent(SubModule.BeforeBatch)

        if hasattr(SubModule, "AfterTestBatch"):
            self.RemoveAfterTestBatchEvent(SubModule.AfterTestBatch)
        elif hasattr(SubModule, "AfterBatch"):
            self.RemoveAfterTestBatchEvent(SubModule.AfterBatch)

        if hasattr(SubModule, "AfterTestEpoch"):
            self.RemoveAfterTestEpochEvent(SubModule.AfterTestEpoch)
        elif hasattr(SubModule, "AfterEpoch"):
            self.RemoveAfterTestEpochEvent(SubModule.AfterEpoch)

        if hasattr(SubModule, "AfterTest"):
            self.RemoveAfterTestEvent(SubModule.AfterTestEpoch)
        elif hasattr(SubModule, "AfterTrain"):
            self.RemoveAfterTestEvent(SubModule.AfterEpoch)

        return super().RemoveSubModule(Name=Name, SubModule=SubModule)

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
        self.SaveDir = Param.setdefault("SaveDir", "./test/")
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
    def SetLoss(self, LossModule, *List, **Dict):
        if isinstance(LossModule, str):
            LossModule = DLUtils.Loss(LossModule, *List, **Dict)
        self.LossModule = LossModule
        return self
    def _Evaluate1Loss(self, Dict):
        Loss = self.LossModule(Out=Dict.Out, OutTarget=Dict.OutTarget)
        self.Loss = Loss
        Evaluation = Loss
        return Evaluation
    def _EvaluateNLoss(self, Dict):
        Loss = self.LossModule(Out=Dict.Out, OutTarget=Dict.OutTarget)
        self.Loss = Loss
        Evaluation = Loss
        return Evaluation
    def AfterTrain(self, Dict):
        self.Log(f"After Train.")
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        Type = Param.setdefault("Type", "1Loss")
        if Type in ["1Loss"]:
            self.Evaluate = self._Evaluate1Loss
        elif Type in ["NLoss"]:
            self.Evaluate = self._EvaluateNLoss
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
            "NumCorrect": [],
            "RateCorrect": [] # AccuracyRate
        })
        Param.Log.append(
            TrainLog
        )
        self.TrainLog = TrainLog
    def Evaluate(self, Dict):
        Loss = self.LossModule(Out=Dict.Output, OutTarget=Dict.OutputTarget)
        NumTotal, NumCorrect = RateCorrectSingelClassPrediction(Dict.Output, Dict.OutputTarget)
        self.Loss = Loss
        self.NumTotal = NumTotal
        self.NumCorrect = NumCorrect
        Evaluation = DLUtils.param({
            "Loss": Loss, "NumTotal": NumTotal, "NumCorrect": NumCorrect
        })
        return Evaluation



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
    XsTest = EvaluationLog.EpochIndexList(IsTrain=False)
    YsTrain = EvaluationLog.CorrectRateEpoch(IsTrain=True)
    YsTest = EvaluationLog.CorrectRateEpoch(IsTrain=False)
    DLUtils.plot.PlotMultiLineChart(
        XsList = [XsTrain, XsTest],
        YsList = [YsTrain, YsTest],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Correct Rate",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "test"],
        Title="Epoch - CorrectRate Relationship",
        SavePath=SaveDir + "Epoch ~ CorrectRate.svg"
    )

def PlotLoss(TrainSession, SaveDir=None):
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    EvaluationLog = TrainSession.EvaluationLog
    # assert isinstance(EvaluationLog, DLUtils.train.Select1FromN.EvaluationLog)
    XsTrain = EvaluationLog.EpochIndexList(IsTrain=True)
    XsTest = EvaluationLog.EpochIndexList(IsTrain=False)
    YsTrain = EvaluationLog.LossEpoch(IsTrain=True)
    YsTest = EvaluationLog.LossEpoch(IsTrain=False)
    DLUtils.plot.PlotMultiLineChart(
        XsList = [XsTrain, XsTest],
        YsList = [YsTrain, YsTest],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Loss",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "test"],
        Title="Epoch ~ Loss Relationship",
        SavePath = SaveDir + "train-curve/" + "Epoch ~ Loss.svg"
    )

def PlotLossBatch(TrainSession, SaveDir=None):
    SaveDir = DLUtils.file.StandardizePath(SaveDir)
    EvaluationLog = TrainSession.EvaluationLog
    assert isinstance(EvaluationLog, DLUtils.train.Select1FromN.EvaluationLog)
    XsTrain, YsTrain = EvaluationLog.LossListBatch(IsTrain=True)
    XsTest, YsTest = EvaluationLog.LossListBatch(IsTrain=False)     

    DLUtils.plot.PlotMultiLineChart(
        XsList = [XsTrain, XsTest],
        YsList = [YsTrain, YsTest],
        ColorList = ["Red", "Blue"],
        XLabel="Epoch", YLabel="Loss",
        XTicks="Int", YTicks="Float",
        XRange = [min(XsTrain), max(XsTrain)], YRange=[0.0, 1.0],
        Labels = ["train", "test"],
        Title="Epoch ~ Loss Relationship",
        SavePath = SaveDir + "train-curve/" + "Epoch ~ Loss.svg"
    )

class DataLoaderForEpochBatchTrain(torch.utils.data.DataLoader, DLUtils.module.AbstractModule):
    SetParamMap = DLUtils.IterableKeyToElement({
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
    def GetNextBatch(self, BatchIndex):
        In, OutTarget = next(self.Iter)
        return In, OutTarget
    Get = GetNextBatch
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

        self.BatchSize = Param.Batch.Size
        self.BatchNum = Param.Batch.Num

        # device setting
        if hasattr(self.DataFetcher, "Device"):
            self.Device = self.DataFetcher.Device
        
        super().Init(IsSuper=True, IsRoot=IsRoot)
        self.Reset()
        return self