import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
# from DLUtils.attr import *
import DLUtils

class AbstractModule:
    def __init__(self, Log=None):
        self.Name = "NullName"
        self.SubModules = DLUtils.param()
        Param = self.Param = DLUtils.Param()
        Param._CLASS = "DLUtils.NN.AbstractModule"
        Param._PATH = "Root"
        if Log is not None:
            self.Log = Log
    def __call__(self, *List, **Dict):
        return self.Receive(*List, **Dict)
    def SetLog(self, Log, SetForSubModules=True):
        self.Log = Log
        Param = self.Param
        if hasattr(self, "LogCache"):
            for log in self.LogCache:
                self.Log.Add(log)
        if SetForSubModules:
            self.SetLogRecur()
        return self
    def SetLogRecur(self, Log=None):
        if Log is None:
            Log = self.Log
        for Name, SubModule in self.SubModules.items():
            SubModule.SetLog(Log)
        return self
    def AddLog(self, Content, Type="Unknown"):
        Param = self.Param
        log = DLUtils.param({
                "Logger": Param._PATH,
                "Type": Type,
                "Content": Content
            })
        if not hasattr(self, "Log"):
            if not hasattr(self, "LogCache"):
                self.LogCache = []
            self.LogCache.append(log)
        else:
            self.Log.Add(log)
        return self
    def ExtractParam(self, RetainSelf=True):
        Param = self.Param
        self.ExtractParamRecur(self, Param, RetainSelf)
        return Param
    def ExtractParamRecur(self, Param, RetainSelf):
        for Name, SubModule in self.SubModules.items():
            setattr(Param, Name, SubModule.ExtractParam(RetainSelf))
        return self.Param
    def LoadParam(self, Param):
        self.Param = Param
        if Param.hasattr("Tensors"):
            self.UpdateTensorfrom_dict()
        else:
            Param.Tensors = []
        self.LoadParamRecur(Param)
        return self
    def LoadParamRecur(self, Param):
        for Name, SubModule in Param.SubModules.items():
            ModuleClass = DLUtils.python.ParseClass(SubModule._CLASS)
            SubModule._PATH = Param._PATH + "." + Name
            self.SubModules[Name] = ModuleClass().LoadParam(SubModule)
            #setattr(self, Name, SubModule)
        return self
    def AddSubModule(self, Name, Module):
        Param = self.Param
        if hasattr(Module, "Param"):
            Param.SubModules.setattr(Name, Module.Param)
        else:
            Param.SubModules.setattr(Name, "_THIS_MODULE_HAS_NO_PARAM_")
        self.SubModules[Name] = Module
        setattr(self, Name, Module)
        return self
    def DelSubModule(self, Name):
        Param = self.Param
        Param.SubModules.delattr(Name)
    def SetAsRoot(self):
        self.Param._IS_ROOT = True
        return self
    def UpdateTensorFromDict(self):
        Param = self.Param
        if isinstance(self, DLUtils.NN.NonLinearLayer):
            a = 1
        for Name in Param.TrainableParam:
            setattr(self, Name, DLUtils.ToTorchTensorOrNum(getattr(Param.Data, Name)))
    def UpdateDictFromTensor(self):
        Param = self.Param
        for Name in Param.TrainableParam:
            setattr(Param.Data, Name, DLUtils.ToNpArrayOrNum(getattr(self, Name)))
    def ToFile(self, FilePath):
        Param = self.ExtractParam(RetainSelf=False)
        DLUtils.file.Obj2File(Param, FilePath)
        return self
    def FromFile(self, FilePath):
        self.SubModules = {}
        Param = DLUtils.file.File2Obj(FilePath)
        self.LoadParam(Param)
        return self
    def ToJsonFile(self, FilePath):
        self.ExtractParam(RetainSelf=True).ToJsonFile(FilePath)
        return self
    def PathStr(self):
        Param =self.Param
        if not hasattr(self, "_PathStr"):
            if isinstance(Param._PATH, str):
                self._PathStr = Param._PATH
            elif isinstance(Param._PATH, list):
                self._PathStr = ".".join(Param._PATH)
            else:
                raise Exception()
        return self._PathStr
    def ClassStr(self):
        if hasattr(self, "ClassStr"):    
            return self.ClassStr
        else:
            return str(self.__class__)
    def Init(self, IsSuper=False, IsRoot=True):
        # Check whether this object is correctly configured.
        for Name, SubModule in self.SubModules.items():
            SubModule.Init(IsSuper=False, IsRoot=False)
        if hasattr(self, "Log"):
            self.SetLogRecur()
        return self
    def AddLogWithSelfInfo(self, Content, Type="Unknown"):
        self.AddLog(f"{self.PathStr()}({self.ClassStr()}): {Content}", Type=Type)
        return self

class AbstractOperator(AbstractModule):
    # operation module without trainable parameter
    def __init__(self, Log=None):
        super().__init__(Log=Log)
    def AddSubModule(self, Name, Module):
        raise Exception("AbstractOperator module.")

class AbstractNetwork(AbstractModule):
    # network with trainable weights.
    def __init__(self, Log=None):
        super().__init__(Log=Log)
        Param = self.Param
        Param.Tensors = []
        Param.TrainableParam = []
    def ExtractTrainableParam(self, ParamDict={}, PathStrPrefix=True, Recur=True):
        self.UpdateDictFromTensor()
        Param = self.Param
        TrainableParamName = self.Param.get("TrainableParam")
        if PathStrPrefix:
            Prefix = self.PathStr() + "."
        else:
            Prefix = ""
        if TrainableParamName is not None:
            for Name in TrainableParamName:
                ParamDict[Prefix + Name] = Param.Data.getattr(Name)
        if Recur:
            self.ExtractTrainableParamRecur(ParamDict=ParamDict, PathStrPrefix=PathStrPrefix)
        return ParamDict
    def ExtractTrainableParamRecur(self, ParamDict={}, PathStrPrefix=True):
        for Name, SubModule in self.SubModules.items():
            SubModule.ExtractTrainableParam(ParamDict=ParamDict, PathStrPrefix=PathStrPrefix, Recur=True) 
        return ParamDict
    def PlotWeight(self, SaveDir=None, SaveName=None):
        Param = self.Param
        Param = self.ExtractParam()
        SavePath = DLUtils.ParseSavePath(SaveDir, SaveName, SaveNameDefault=Param._PATH)
        for WeightName in Param.TrainableParam:
            Data = Param.Data.getattr(WeightName)
            if hasattr(Data, "shape"):
                DimNum = len(Data.shape)
            else:
                DimNum = 0
            if DimNum == 1 or DimNum == 0:
                DLUtils.plot.PlotData1D(
                    Name=WeightName,
                    Data=Data,
                    SavePath=SavePath + "." + WeightName + ".svg",
                    #XLabel="Dimension 0", YLabel="Dimension 0"
                )
                DLUtils.plot.PlotDataAndDistribution1D(
                    Name=WeightName,
                    Data=Data,
                    SavePath=SavePath + "." + WeightName + " - Distribution.svg",
                    XLabel="Dimension 0", YLabel="Dimension 0"
                )
            elif DimNum == 2:
                DLUtils.plot.PlotData2D(
                    Name=WeightName,
                    Data=Data,
                    SavePath=SavePath + "." + WeightName + ".svg",
                    XLabel="Output Dimension", YLabel="Input Dimension"
                )
                DLUtils.plot.PlotDataAndDistribution2D(
                    Name=WeightName,
                    Data=Data,
                    SavePath=SavePath + "." + WeightName + " - Distribution.svg",
                    XLabel="Output Dimension", YLabel="Input Dimension"
                )
        self.PlotWeightRecur(SaveDir, SaveName)
        return self
    def ExtractParam(self, RetainSelf=True):
        Param = self.Param
        if not RetainSelf:
            # prune some empty nodes for readability and storage space saving.
            if Param.hasattr("TrainableParam") and len(Param.TrainableParam) == 0:
                Param.delattr("TrainableParam")
                if isinstance(self, DLUtils.NN.LinearLayer):
                    a = 1
        self.UpdateDictFromTensor()
        self.ExtractParamRecur(Param, RetainSelf)
        return Param
    def LoadParam(self, Param):
        self.Param = Param
        if Param.hasattr("TrainableParam"):
            self.UpdateTensorFromDict()
        else:
            Param.TrainableParam = []
        self.LoadParamRecur(Param)
        return self
    def PlotWeightRecur(self, SaveDir, SaveName):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "PlotWeight"):
                SubModule.PlotWeight(SaveDir, SaveName)
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        super().Init(True, IsRoot=IsRoot)
        self.UpdateTensorFromDict()
        assert hasattr(self, "Receive")
        if IsRoot:
            self.AddLog(f"{Param._CLASS}: initialization finished.", Type="initialization")
        return self