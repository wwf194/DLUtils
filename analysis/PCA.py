import numpy as np
import collections

import DLUtils

class LogForPCA(DLUtils.log.AbstractLogAlongBatch):
    def __init__(self, EpochIndex=None, BatchIndex=None, **kw):
        super().__init__(**kw)
        self.cache = DLUtils.EmptyPyObj()
        self.data = DLUtils.EmptyPyObj()
        cache = self.cache
        data = self.data
        data.log = DLUtils.GetDefaultDict(
            lambda:DLUtils.PyObj({
                "data":[]
            })
        )
        if EpochIndex is not None:
            self.data.EpochIndex = EpochIndex
        if BatchIndex is not None:
            self.data.BatchIndex = BatchIndex
    # def FromFile(self, FilePath):
    #     self.data = DLUtils.json.DataFile2PyObj(FilePath)
    #     return self
    # def ToFile(self, FilePath):
    #     DLUtils.json.PyObj2DataFile(self.data, FilePath)
    #     return self
    def LogBatch(self, Name, data):
        data = DLUtils.ToNpArray(data)
        data = data.reshape(-1, data.shape[-1]) # [SampleNum, FeatureNum]
        self.data.log[Name.replace(".", "(dot)")].data.append(data)
    def CalculatePCA(self):
        data = self.data
        for name, log in data.log.items():
            log.data = np.concatenate(log.data, axis=0)
            log.PCATransform = DLUtils.math.PCA(log.data)
        return
#DLUtils.transform.SetEpochBatchMethodForModule(LogForPCA, MountLocation="data")

class LogForPCAAlongTrain(DLUtils.log.AbstractLogAlongEpochBatchTrain):
    def __init__(self, EpochNum, BatchNum, **kw):
        super().__init__(**kw)
        #ConnectivityPattern = DLUtils.EmptyPyObj()
        data = self.data = DLUtils.EmptyPyObj()
        data.EpochNum = EpochNum
        data.BatchNum = BatchNum
        data.log = DLUtils.GetDefaultDict(lambda:[])
    def FromLogForPCA(self, logPCA, EpochIndex=None, BatchIndex=None):
        if EpochIndex is None:
            EpochIndex = logPCA.GetEpochIndex()
        if BatchIndex is None:
            BatchIndex = logPCA.GetBatchIndex()
        for Name, Log in logPCA.data.log.Items():
            self.Log(
                Name, EpochIndex, BatchIndex, Log.PCATransform
            )
    def Log(self, Name, EpochIndex, BatchIndex, PCATransform):
        _data = DLUtils.PyObj({
            "EpochIndex": EpochIndex, 
            "BatchIndex": BatchIndex,
        }).FromPyObj(PCATransform)
        self.data.log[Name].append(_data)
        return self
    def CalculateEffectiveDimNum(self, Data, RatioThres=0.5):
        # if not hasattr(Data, "VarianceExplainedRatioAccumulated"):
        #     Data.VarianceExplainedRatioAccumulated = np.cumsum(Data.VarianceExplainedRatio)
        DimCount = 0
        for RatioAccumulated in Data.VarianceExplainedRatioAccumulated:
            DimCount += 1
            if RatioAccumulated >= RatioThres:
                return DimCount
    def Plot(self, SaveDir):
        for Name, Data in self.data.log.items():
            _Name = Name.replace("(dot)", ".")
            self._Plot(
                Data,
                SaveDir + _Name + "/",
                _Name
            )
    def _Plot(self, Data, SaveDir, SaveName):
        BatchNum = self.data.BatchNum
        Data.sort(key=lambda Item:Item.EpochIndex + Item.BatchIndex * 1.0 / BatchNum)
        #SCacheSavePath = SaveDir + "Data/" + "EffectiveDimNums.data"
        # if DLUtils.file.ExistsFile(CacheSavePath):
        #     EffectiveDimNums = DLUtils.json.DataFile2PyObj(CacheSavePath)
        # else:
        for _Data in Data:
            _Data.VarianceExplainedRatioAccumulated = np.cumsum(_Data.VarianceExplainedRatio)
            _Data.from_dict({
                "EffectiveDimNums":{
                    "P100": len(_Data.VarianceExplainedRatio),
                    "P099": self.CalculateEffectiveDimNum(_Data, 0.99),
                    "P095": self.CalculateEffectiveDimNum(_Data, 0.95),
                    "P080": self.CalculateEffectiveDimNum(_Data, 0.80),
                    "P050": self.CalculateEffectiveDimNum(_Data, 0.50),
                }
            })
            DLUtils.json.PyObj2DataFile(
                _Data, SaveDir + "cache/" + "Epoch%d-Batch%d.data"%(_Data.EpochIndex, _Data.BatchIndex)
            )
        EpochIndices, BatchIndices, EpochFloats = [], [], []
        EffectiveDimNums = DLUtils.GetDefaultDict(lambda:[])
        for _Data in Data:
            EpochIndices.append(_Data.EpochIndex)
            BatchIndices.append(_Data.BatchIndex)
            EpochFloats.append(_Data.EpochIndex + _Data.BatchIndex * 1.0 / BatchNum)
            EffectiveDimNums["100"].append(_Data.EffectiveDimNums.P100)
            EffectiveDimNums["099"].append(_Data.EffectiveDimNums.P099)
            EffectiveDimNums["095"].append(_Data.EffectiveDimNums.P095)
            EffectiveDimNums["080"].append(_Data.EffectiveDimNums.P080)
            EffectiveDimNums["050"].append(_Data.EffectiveDimNums.P050)

        fig, axes = DLUtils.plot.CreateFigurePlt(1)
        ax = DLUtils.plot.GetAx(axes, 0)
        LineNum = len(EffectiveDimNums)
        DLUtils.plot.SetMatplotlibParamToDefault()
        DLUtils.plot.PlotMultiLineChart(
            ax, 
            [EpochFloats for _ in range(LineNum)],
            EffectiveDimNums.values(),
            XLabel="Epochs", YLabel="DimNum required to explain proportion of total variance",
            Labels = ["$100\%$", "$99\%$", "$95\%$", "$80\%$", "$50\%$"],
            Title = "Effective Dimension Num - Training Process"
        )
        DLUtils.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + ".svg")

def AnalyzePCAForEpochBatchTrain(ContextObj):
    TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)
    # Do supplementary analysis for all saved models under main save directory.
    GlobalParam = DLUtils.GetGlobalParam()
    ContextObj.setdefault("ObjRoot", GlobalParam)
    
    AnalysisSaveDir = ContextObj.setdefault("SaveDir", DLUtils.GetMainSaveDir() + "PCA-Analysis-Along-Training-Test/")

    DLUtils.DoTasks( # Dataset can be reused.
        "&^param.task.BuildDataset", **ContextObj.ToDict()
    )

    SaveDirs = DLUtils.GetAllSubSaveDirsEpochBatch(Name="SavedModel")
    
    Trainer = ContextObj.Trainer
    EpochNum = Trainer.GetEpochNum()
    
    BatchSize = Trainer.GetBatchSize()
    #BatchNum = GlobalParam.object.image.EstimateBatchNum(BatchSize, Type="Train")
    BatchNum = Trainer.GetBatchNum()

    logPCA = LogForPCAAlongTrain(EpochNum, BatchNum)
    for SaveDir in SaveDirs:
        EpochIndex, BatchIndex = DLUtils.train.ParseEpochBatchFromStr(SaveDir)
        CacheSavePath = AnalysisSaveDir + "cache/" + "Epoch%d-Batch%d.data"%(EpochIndex, BatchIndex)
        if DLUtils.ExistsFile(CacheSavePath): # Using cached data
            Data = DLUtils.json.DataFile2PyObj(CacheSavePath)
            if hasattr(Data, "PCATransform"):
                logPCA.Log(
                    EpochIndex, BatchIndex, Data.PCATransform
                )
            else:
                logPCA.Data.append(Data)
            continue
        DLUtils.AddLog("Testing Model at Epoch%d-Batch%d"%(EpochIndex, BatchIndex))

        DLUtils.DoTasks(
            "&^param.task.Load",
            In={"SaveDir": SaveDir}, 
            **ContextObj.ToDict()
        )
        DLUtils.DoTasks(
            "&^param.task.BuildTrainer", **ContextObj.ToDict()
        )
        _logPCA = RunBatchesAndCalculatePCA( # Run test batches and do PCA.
            EpochIndex=EpochIndex, BatchIndex=BatchIndex, TestBatchNum=TestBatchNum
        )
        DLUtils.json.PyObj2DataFile(
            DLUtils.PyObj({
                "PCATransform": _logPCA.PCATransform,
            }),
            CacheSavePath
        )
        logPCA.Log(
            EpochIndex, BatchIndex, _logPCA.PCATransform
        )
    logPCA.Plot(
        SaveDir=AnalysisSaveDir, SaveName="agent.model.FiringRates"
    )

def PlotPCAAlongTrain(LogsPCA, DataDir=None, SaveDir=None, ContextObj=None):
    LogPCAAlongTrain = LogForPCAAlongTrain(ContextObj.EpochNum, ContextObj.BatchNum)
    for Log in LogsPCA:
        LogPCAAlongTrain.FromLogForPCA(Log)
    LogPCAAlongTrain.Plot(SaveDir)

def ScanLogPCA(ScanDir=None):
    if ScanDir is None:
        ScanDir = DLUtils.GetMainSaveDir() + "PCA-Analysis-Along-Train-Test/" + "cache/"
    FileList = DLUtils.file.ListFiles(ScanDir)
    Logs = []
    LoadedFileName = []
    for FileName in FileList:
        FileName = DLUtils.RStrip(FileName, ".data")
        FileName = DLUtils.RStrip(FileName, ".jsonc")
        if FileName in LoadedFileName:
            continue
        LoadedFileName.append(FileName)
        Logs.append(LogForPCA().FromFile(ScanDir, FileName).Build(IsLoad=True))
        # EpochIndex, BatchIndex = DLUtils.train.ParseEpochBatchFromStr(FileName)
        # Logs.append({
        #     "Epoch": EpochIndex,
        #     "Batch": BatchIndex,
        #     "Value": DLUtils.analysis.LogForPCA().FromFile(ScanDir + FileName)
        # })
    # Logs.sort(cmp=DLUtils.train.CmpEpochBatchDict)
    DLUtils.SortListByCmpMethod(Logs, DLUtils.train.CmpEpochBatchObj)
    return Logs

# def RunBatchesAndCalculatePCA(ContextObj):
#     GlobalParam = DLUtils.GetGlobalParam()
#     Trainer = ContextObj.Trainer
#     agent = Trainer.agent
#     dataset = Trainer.world
#     BatchParam = GlobalParam.param.task.Train.BatchParam
#     Dataset.CreateFlow(BatchParam, "Test")
#     log = DLUtils.log.LogForEpochBatchTrain()
#     log.SetEpochIndex(0)
#     TestBatchNum = ContextObj.setdefault("TestBatchNum", 10)
#     logPCA = LogForPCA()
#     for TestBatchIndex in range(TestBatchNum):
#         DLUtils.AddLog("Epoch%d-Index%d-TestBatchIndex-%d"%(ContextObj.EpochIndex, ContextObj.BatchIndex, TestBatchIndex))
#         log.SetBatchIndex(TestBatchIndex)
#         InList = [
#             Trainer.GetBatchParam(), Trainer.GetOptimizeParam(), log
#         ]
#         # InList = DLUtils.parse.ParsePyObjDynamic(
#         #     DLUtils.PyObj([
#         #         "&^param.task.Train.BatchParam",
#         #         "&^param.task.Train.OptimizeParam",
#         #         #"&^param.task.Train.SetEpochBatchList"
#         #         log,
#         #     ]),
#         #     ObjRoot=GlobalParam
#         # )
#         DLUtils.CallGraph(agent.Dynamics.RunTestBatch, InList=InList)
#         logPCA.Log(
#             log.GetLogValueByName("agent.model.FiringRates")[:, -1, :],
#         )
#     logPCA.ApplyPCA()
#     return logPCA
