import torch
from collections import defaultdict
import DLUtils
#Operators = DLUtils.PyObj()

ModuleList = []

def BuildModuleIfIsLegalType(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type

    if IsLegalModuleType(Type):
        return BuildModule(param, **kw)
    else:
        return None

def BuildModule(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
        
    if Type in ["FunctionsOutputs"]:
        return FunctionsOutputs()
    else:
        raise Exception(Type)

def IsLegalModuleType(Type):
    if Type in ModuleList:
        return True
    else:
        return False

def Add(*Args):
    return sum(Args)
ModuleList.append(["Add"])

def Filterfrom_dict(Dict, Name):
    return Dict[Name]
ModuleList.append(["Filterfrom_dict"])

def Split(Args):
    if isinstance(Args, list):
        return Args
    elif isinstance(Args, DLUtils.PyObj) and Args.IsListLike():
        return Args
    else:
        raise Exception
# Operators.Split = Split
ModuleList.append(["Split"])

def Merge(*Args):
    return Args
ModuleList.append(["Merge"])

def FunctionsOutputs2List(Functions):
    Outputs = []
    for Function in Functions:
        Output = DLUtils.CallFunction(Function)
        Outputs.append(Output)
    return Outputs
# Operators.FunctionsOutputs2List = FunctionsOutputs2List

from DLUtils.transform import AbstractTransform
class FunctionsOutputs(AbstractTransform):
    # def __init__(self, param=None, data=None, **kw):
    #     self.InitModule(self, param, data, 
    #         ClassPath="DLUtils.transform.operator.FunctionsOutputs", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        cache = self.cache
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        DLUtils.ParseFunctionParamsStatic(param.Functions)
        cache.Functions = DLUtils.parse.ParsePyObjDynamic(
            param.Functions,
            ObjCurrent=param.Functions.cache.__ResolveRef__
            # to be implemented: ObjRoot = ?
        )
        return
    def __call__(self):
        return self.forward()
    def forward(self):
        return FunctionsOutputs2List(self.cache.Functions)
#DLUtils.transform.SetMethodForTransformModule(FunctionsOutputs)
ModuleList.append("FunctionsOutputs")

def CalculateGradient(loss):
    loss.backward()
    return
# Operators.CalculateGradient = CalculateGradient
ModuleList.append(["CalculateGradient"])



def CreateDataLogger():
    return DLUtils.log.DataLogger()
ModuleList.append("CreateDataLogger")

def PlotDistribution(Activity, Name="UnNamed"):
    activity = DLUtils.ToNpArray(Activity)
    DLUtils.plot.PlotDistribution1D(activity, Name=Name)

def LogStat(data, Name):
    data = DLUtils.ToNpArray(data)
    statistics = DLUtils.math.NpStatistics(data, ReturnType="Dict")
    DLUtils.GetDataLogger().LogDict({statistics})

def Tensor2Statistics2File(data, Name, FilePath=None):
    #Name, FilePath = DLUtils.ParseTextFilePathFromName(Name, FilePath)
    if FilePath is None:
        FilePath = DLUtils.GetMainSaveDir() + Name + "-statistics" + ".txt"
        FilePath = DLUtils.RenameIfFileExists(FilePath)
    statistics = DLUtils.math.TorchTrainParamtat(data)
    DLUtils.Data2TextFile(statistics, FilePath=FilePath)

ModuleList.append("Data2TextFile")

from DLUtils.plot import CompareDensityCurve
ModuleList.append("CompareDensityCurve")

# from DLUtils.train import ClearGrad
# ModuleList.append("ClearGrad")

# from DLUtils.train import Probability2MostProbableIndex
# ModuleList.append("Probability2MostProbableIndex")

# from DLUtils.transform import LogAccuracyForSingleClassPrediction
# ModuleList.append("LogAccuracyForSingleClassPrediction")