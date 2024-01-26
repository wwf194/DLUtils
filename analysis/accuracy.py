
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
else:
    mpl = DLUtils.GetLazyMatplotlib()
    plt = DLUtils.GetLazyPlt()

class LogForAccuracy:
    def __init__(self):
        self.param = DLUtils.EmptyPyObj()
        self.cache = DLUtils.EmptyPyObj()


class LogForAccuracyAlongTrain:
    def __init__(self):
        self.param = DLUtils.EmptyPyObj()
        self.cache = DLUtils.EmptyPyObj()
        param = self.param
        cache = self.cache
        
        EnsureAttrs(param, "LogBatchNum", default=5)
        cache.LogBatchNum = param.LogBatchNum
        
        cache.CorrectNumList = [0 for _ in range(cache.LogBatchNum)]
        cache.TotalNumList = [0 for _ in range(cache.LogBatchNum)]
        cache.ListIndex = 0
        return
    def Update(self, CorrectNum, TotalNum):
        cache = self.cache
        cache.CorrectNumList[cache.ListIndex] = CorrectNum
        cache.TotalNumList[cache.ListIndex] = TotalNum
        cache.ListIndex = (cache.ListIndex + 1) / cache.LogBatchNum
    def GetAccuracy(self):
        cache = self.cache
        return 1.0 * sum(cache.CorrectNumList) / sum(cache.TotalNumList)


def PlotAccuracyEpochBatch(LogTrain, LogTest=None, SaveDir=None, SaveName=None, ContextObj=None):
    XsData, YsData = [], []
    
    EpochsFloatTrain = DLUtils.log.LogDict2EpochsFloat(LogTrain, BatchNum=ContextObj["BatchNum"])
    CorrectRateTrain = LogTrain["CorrectRate"]
    fig, ax = DLUtils.plot.CreateFigurePlt()
    DLUtils.plot.PlotLineChart(
        ax, Xs=EpochsFloatTrain, Ys=CorrectRateTrain,
        PlotTicks=False, Label="Train", Color="Red",
        Title="Accuracy - Epoch", XLabel="Epoch", YLabel="Accuracy",
    )
    XsData.append(EpochsFloatTrain)
    YsData.append(CorrectRateTrain)

    if LogTest is not None:
        EpochsFloatTest = DLUtils.log.LogDict2EpochsFloat(LogTest, BatchNum=ContextObj["BatchNum"])
        CorrectRateTest = LogTest["CorrectRate"]
        DLUtils.plot.PlotLineChart(
            ax, Xs=EpochsFloatTest, Ys=CorrectRateTest,
            PlotTicks=False, Label="Test", Color="Blue",
            Title="Accuracy - Epoch", XLabel="Epoch", YLabel="Accuracy",
        )
        XsData.append(EpochsFloatTest)
        YsData.append(CorrectRateTest)

    DLUtils.plot.SetXTicksFloatFromData(ax, XsData)
    DLUtils.plot.SetYTicksFloatFromData(ax, YsData)
    ax.legend()

    if SaveName is None:
        SaveName = "Accuracy~Epoch"

    DLUtils.plot.SaveFigForPlt(SavePath=SaveDir + SaveName + ".svg")
    DLUtils.file.Table2TextFileDict(LogTrain, SavePath=SaveDir + SaveName + ".txt")
    return