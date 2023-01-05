import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
# from DLUtils.attr import *
import DLUtils

import functools
import warnings

class LogComponent:
    # method for class
    def SetLog(self, Logger, SetForSubModules=True):
        self.Logger = Logger
        if hasattr(self, "LogCache"):
            for Log in self.LogCache:
                self.Logger.Add(Log)
        if SetForSubModules:
            self.SetLogRecur()
        return self
    def SetLogRecur(self, Logger=None):
        if Logger is None:
            Logger = self.Logger
        for Name, SubModule in self.SubModules.items():
            SubModule.SetLog(Logger)
        return self
    def Log(self, Content, Type="Unknown"):
        Param = self.Param
        log = DLUtils.param({
                "Logger": Param._PATH,
                "Type": Type,
                "Content": Content
            })
        if not hasattr(self, "Logger"):
            if not hasattr(self, "LogCache"):
                self.LogCache = []
            self.LogCache.append(log)
        else:
            self.Logger.Add(log)
        return self

class AbstractModule(LogComponent):
    def __init__(self, Logger=None):
        self.Name = "NullName"
        self.SubModules = DLUtils.param()
        Param = self.Param = DLUtils.Param()
        #Param._CLASS = "DLUtils.module.AbstractModule"
        Param._CLASS = DLUtils.system.ClassPathStr(self)
        Param._PATH = "Root"
        if Logger is not None:
            self.Logger = Logger
    def __call__(self, *List, **Dict):
        return self.Receive(*List, **Dict)
    def ExtractParam(self, RetainSelf=True):
        if hasattr(self, "Param"):
            Param = self.Param
            self.ExtractParamRecur(Param, RetainSelf)
            return Param
        else:
            return "MODULE_WITHOUT_PARAM"
    def ExtractParamRecur(self, Param, RetainSelf):
        for Name, SubModule in self.SubModules.items():
            setattr(Param, Name, SubModule.ExtractParam(RetainSelf))
        return self.Param
    def LoadParam(self, Param):
        self.Param = Param
        self.SubModules = DLUtils.param()
        self.SetEventDict()
        self.LoadParamRecur(Param)
        return self
    def LoadParamRecur(self, Param):
        for Name, SubModule in Param.SubModules.items():
            ModuleClass = DLUtils.python.ParseClass(SubModule._CLASS)
            SubModule._PATH = Param._PATH + "." + Name
            self.SubModules[Name] = ModuleClass().LoadParam(SubModule)
        return self
    def SetParam(self, **Dict):
        Param = self.Param
        for Key, Value in Dict.items():
            Param.setattr(Key, Value)
        return self
    def SetAttr(self, **Dict):
        for Key, Value in Dict.items():
            setattr(self, Key, Value)
        return self
    def AddSubModule(self, Name, SubModule):
        Param = self.Param
        if hasattr(SubModule, "Param"):
            Param.SubModules.setattr(Name, SubModule.Param)
            SubModule.Param._PATH = Param._PATH + "." + Name
        else:
            Param.SubModules.setattr(Name, "MODULE_WITHOUT_PARAM")
        self.SubModules[Name] = SubModule
        setattr(self, Name, SubModule)
        return self
    def GetSubModule(self, Name):
        return self.SubModules.getattr(Name)
    def RemoveSubModule(self, SubModule):
        Param = self.Param
        if isinstance(SubModule, str):
            Name = SubModule
            if Name in self.SubModules.keys():
                self.SubModules.pop(Name)
                Param.SubModules.delattr(Name)
            else:
                warnings.warn("{1}.RemoveSubModule: No such SubModule: {2}".format(Param._PATH, Name))
        else:
            HasRemoved = False
            for Name, _SubModule in self.SubModules.items():
                if SubModule == _SubModule:
                    self.SubModules.pop(Name)
                    Param.SubModules.delattr(Name)
                    HasRemoved = True
                    break
            if not HasRemoved:
                warnings.warn("{1}.RemoveSubModule: No such SubModule: {2}".format(Param._PATH, Name))
        return self

    def DelSubModule(self, Name):
        Param = self.Param
        Param.SubModules.delattr(Name)
    def SetAsRoot(self):
        self.Param._IS_ROOT = True
        return self
    def ToFile(self, FilePath, RetainSelf=False):
        Param = self.ExtractParam(RetainSelf=RetainSelf)
        DLUtils.file.Obj2File(Param, FilePath)
        return self
    def FromFile(self, FilePath):
        Param = DLUtils.file.File2Obj(FilePath)
        self.LoadParam(Param)
        return self
    def ToJsonFile(self, FilePath):
        self.ExtractParam(RetainSelf=True).ToJsonFile(FilePath)
        return self
    def PathStr(self):
        Param = self.Param
        if isinstance(self, DLUtils.network.NonLinearLayer):
            a = 1
        #if not hasattr(self, "_PathStr"):
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
    def SetEventDict(self):
        if not hasattr(self, "EventDict"):
            self.EventDict = DLUtils.param({
                "TensorMovement":[]
            })
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        for Name, SubModule in self.SubModules.items():
            SubModule.Init(IsSuper=False, IsRoot=False)
            setattr(self, Name, SubModule)
        self.SetEventDict()
        if hasattr(self, "Logger"):
            self.SetLogRecur(self.Logger)
        self.SetConnectEvents()
        return self
    def LogWithSelfInfo(self, Content, Type="Unknown"):
        self.Log(f"{self.PathStr()}({self.ClassStr()}): {Content}", Type=Type)
        return self
    def SetDevice(self, Device=None, IsRoot=True):
        if Device is None:
            Device = self.Device
        else:
            self.Device = Device
        Param = self.Param
        if Param.hasattr("Tensor"):
            for TensorName in Param.Tensor:
                if hasattr(self, TensorName):
                    Tensor = getattr(self, TensorName)
                    TensorNew = Tensor.to(Device).detach() # requires_grad remains same.
                    TensorNew.requires_grad = Tensor.requires_grad
                    setattr(self, TensorName, TensorNew)
                else:
                    Tensor = Param.Data.getattr(TensorName)
                    TensorNew = DLUtils.ToTorchTensor(Tensor).to(Device).detach()
                    TensorNew.requires_grad = Tensor.requires_grad
                    setattr(self, TensorName, TensorNew)
        self.SetDeviceRecur(Device, IsRoot=False)
        if IsRoot:
            self.OnTensorMovement()
            self.Log(
                f"Set device to {Device}", "SystemConfigChange"
            )
        return self
    def SetDeviceRecur(self, Device, IsRoot=True):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "SetDevice"):
                SubModule.SetDevice(Device, IsRoot=False)
        return self
    def ParamNumDict(self, Recur=True, Dict=None, Prefix=None):
        Param = self.Param
        if Dict is None:
            Dict = {}
        if Param.hasattr("Tensor"):
            for Name in Param.Tensor:
                Tensor = Param.Data.getattr(Name)
                Dict[Name] = Tensor.size
        if Prefix is None:
            _PATH = Param.get("_PATH")
            if _PATH is not None:
                Prefix = _PATH + "."
        if Recur:
            Dict = self.ParamNumDictRecur(Recur=Recur, Dict=Dict)
        return Dict
    def ParamNumDictRecur(self, Recur=True, Dict=None):
        if Dict is None:
            Dict = {}
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "ParamNum"):
                SubModule.ParamNum(Recur=Recur, Dict=Dict)
        return Dict
    def Clear(self):
        Attrs = list(self.__dict__.keys())
        for Attr in Attrs:
            if Attr in ["EventDict"]:
                continue
            delattr(self, Attr)
        return self
    def On(self, EventName, Function):
        if not hasattr(self, EventName):
            EventName = EventName.lstrip("On")
        assert hasattr(self, "On" + EventName)
        self.EventDict[EventName].append(Function)
        return self
    def OnTensorMovement(self):
        for Event in self.EventDict.TensorMovement:
            Event(Model=self)
    def SetConnectEvents(self):
        Param = self.Param
        if not Param.Event.hasattr("Connect"):
            return self
        for Connect in Param.Event.Connect:
           self.SetConnectEvent(Connect) 
    def AddConnectEvent(self, SubModule1, Event1, SubModule2, Event2):
        Param = self.Param
        Param.Event.setdefault("Connect", [])
        Param.Event.Connect.append(DLUtils.Param({
            "SubModule1": SubModule1, "Event1": Event1,
            "SubModule2": SubModule2, "Event2": Event2
        }))
        return self
    def SetConnectEvent(self, Connect):
        SubModule1 = self.GetSubModule(Connect.SubModule1)
        SubModule2 = self.GetSubModule(Connect.SubModule2)
        Event2 = getattr(SubModule2, Connect.Event2)
        # def _ConnectEvent(Event, *List, **Dict):
        #     return Event(*List, **Dict)
        SubModule1.On(Connect.Event1, Event2)
        return self
EmptyModule = AbstractModule

class AbstractOperator(AbstractModule):
    # operation module without trainable parameter
    def __init__(self, Logger=None):
        super().__init__(Logger=Logger)
    def AddSubModule(self, Name, SubModule):
        raise Exception("AbstractOperator module.")

class AbstractNetwork(AbstractModule):
    # network with trainable weights.
    def __init__(self, Logger=None):
        super().__init__(Logger=Logger)
        Param = self.Param
        Param.Tensor = set()
        Param.TrainParam = set()
    def AddTrainParam(self, Name, TrainParam):
        Param = self.Param
        Param.Data.setattr(Name, TrainParam)
        Param.TrainParam.add(Name)
        Param.Tensor.add(Name)
        return self
    def AddTensor(self, Name, Tensor):
        Param = self.Param
        Param.Data.setattr(Name, Tensor)
        Param.Tensor.add(Name)
        return self
    def UpdateTensorFromDict(self, Recur=False):
        Param = self.Param
        if Param.hasattr("Tensor"):
            for Name in Param.Tensor:
                Tensor = DLUtils.ToTorchTensorOrNum(getattr(Param.Data, Name))
                if hasattr(self, "Device"):
                    TensorNew = Tensor.to(self.Device).detach()
                    TensorNew.requires_grad = Tensor.requires_grad
                else:
                    TensorNew = Tensor
                setattr(self, Name, TensorNew)
        if Param.hasattr("TrainParam"):
            for Name in Param.TrainParam:
                Tensor = getattr(self, Name)
                Tensor.requires_grad = True
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateTensorFromDict"):
                    SubModule.UpdateTensorFromDict()
        self.OnTensorMovement()
        return self
    def UpdateDictFromTensor(self, Recur=False):
        Param = self.Param
        if  Param.hasattr("TrainParam"):
            for Name in Param.TrainParam:
                if hasattr(self, Name):
                    Tensor = getattr(self, Name)
                    setattr(Param.Data, Name, DLUtils.ToNpArrayOrNum(Tensor))
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateDictFromTensor"):
                    SubModule.UpdateDictFromTensor()
        return self
    def ExtractTrainParam(self, ParamDict={}, PathStrPrefix=True, Recur=True):
        self.UpdateDictFromTensor()
        # self.UpdateTensorFromDict()
        Param = self.Param
        if PathStrPrefix:
            Prefix = self.PathStr() + "."
        else:
            Prefix = ""
        TrainParam = Param.get("TrainParam")
        if TrainParam is not None:
            for Name in TrainParam:
                ParamDict[Prefix + Name] = getattr(self, Name)
        if Recur:
            self.ExtractTrainParamRecur(ParamDict=ParamDict, PathStrPrefix=PathStrPrefix)
        return ParamDict
    def ExtractTrainParamRecur(self, ParamDict={}, PathStrPrefix=True):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "ExtractTrainParam"):
                SubModule.ExtractTrainParam(ParamDict=ParamDict, PathStrPrefix=PathStrPrefix, Recur=True) 
        return ParamDict
    def PlotWeight(self, SaveDir=None, SaveName=None):
        Param = self.Param
        Param = self.ExtractParam()
        SavePath = DLUtils.ParseSavePath(SaveDir, SaveName, SaveNameDefault=Param._PATH)
        if Param.hasattr("TrainParam"):
            for WeightName in Param.TrainParam:
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
        self.UpdateDictFromTensor()
        if not RetainSelf:
            # prune some empty nodes for readability and storage space saving.
            if Param.hasattr("TrainParam") and len(Param.TrainParam) == 0:
                Param.delattr("TrainParam")
        self.ExtractParamRecur(Param, RetainSelf)
        return Param
    def LoadParam(self, Param):

        super().LoadParam(Param)
        # if Param.hasattr("TrainParam"):
        #     self.UpdateTensorFromDict()
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
        if Param.hasattr("TrainParam"):
            for Name in Param.TrainParam:
                setattr(self, Name, Param.Data.getattr(Name))
        if Param.hasattr("Tensor"):
            for Name in Param.Tensor:
                setattr(self, Name, Param.Data.getattr(Name))
        self.UpdateTensorFromDict()
        assert hasattr(self, "Receive")
        if IsRoot:
            self.Log(f"{Param._CLASS}: initialization finished.", Type="initialization")
        return self
    def OnTensorMovement(self):
        for Event in self.EventDict.TensorMovement:
            Event(Model=self, Param=self.ExtractTrainParam())