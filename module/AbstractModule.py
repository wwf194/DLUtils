import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
# from DLUtils.attr import *
import DLUtils
class AbstractModule:
    def __init__(self):
        self.Name = "NullName"
        self.SubModules = {}
        Param = self.Param = DLUtils.Param()
        Param.Tensors = []
        Param._CLASS = "DLUtils.NN.AbstractModule"
        Param._PATH = "Root"
    def __call__(self, *List, **Dict):
        return self.Receive(*List, **Dict)
    def ExtractParam(self, RetainSelf=True):
        Param = self.Param
        if not RetainSelf:
            # prune some empty nodes for readability and storage space saving.
            if len(Param.Tensors) == 0:
                Param.delattr("Tensors")
        self.UpdateDictFromTensor()
        return Param
    def ExtractParamRecur(self, Param):
        for Name, SubModule in self.SubModules:
            setattr(Param, Name, SubModule.ExtractParam())
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
        Param.SubModules.setattr(Name, Module.Param)
        self.SubModules[Name] = Module
        #setattr(self, Name, Module)
        return self
    def DelSubModule(self, Name):
        Param = self.Param
        Param.SubModules.delattr(Name)
    def SetAsRoot(self):
        self.Param._IS_ROOT = True
        return self
    def UpdateTensorfrom_dict(self):
        Param = self.Param
        for Name in Param.Tensors:
            setattr(self, Name, DLUtils.ToTorchTensorOrNum(getattr(Param.Data, Name)))
    def UpdateDictFromTensor(self):
        Param = self.Param
        for Name in Param.Tensors:
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
    def Init(self, IsSuper=False):
        # Check whether this object is correctly configured.
        for Name, SubModule in self.SubModules.items():
            SubModule.Init()
        return self

class AbstractNetwork(AbstractModule):
    # network with trainable weights.
    def __init__(self):
        super().__init__()
    def PlotWeight(self, SaveDir=None, SaveName=None):
        Param = self.Param
        Param = self.ExtractParam()
        SavePath = DLUtils.ParseSavePath(SaveDir, SaveName, SaveNameDefault=Param._PATH)
        for WeightName in Param.Tensors:
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
    def PlotWeightRecur(self, SaveDir, SaveName):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "PlotWeight"):
                SubModule.PlotWeight(SaveDir, SaveName)
        return self
    def Init(self, IsSuper=False):
        super().Init(True)
        assert hasattr(self, "Receive")
        return self