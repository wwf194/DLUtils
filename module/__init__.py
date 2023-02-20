import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import DLUtils
import functools
import warnings


def BuildModule(param, **Dict):
    if hasattr(param, "ClassPath"):
        try:
            Class = DLUtils.parse.ParseClass(param.ClassPath)
            return Class(**Dict)
        except Exception:
            DLUtils.AddWarning("Cannot parse ClassPath: %s"%param.ClassPath)

    module = BuildExternalModule(param, **Dict)
    if module is not None:
        return module
    raise Exception()

ExternalModules = {}

def RegisterExternalModule(Type, Class):
    ExternalModules[Type] = Class

def BuildExternalModule(param, **kw):
    if isinstance(param, str):
        Type = param
    else:
        Type = param.Type
    if Type in ExternalModules:
        return ExternalModules[Type](**kw)
    else:
        return None

class LogComponent:
    # method for class
    def SetLog(self, Log, SetForSubModules=True):
        self._Log = Log
        if hasattr(self, "LogCache"):
            for Log in self.LogCache:
                self._Log.Add(Log)
        if SetForSubModules:
            self.SetLogRecur()
        return self
    def SetLogRecur(self, Log=None):
        if Log is None:
            _Log = self._Log
        else:
            _Log = Log
        for Name, SubModule in self.SubModules.items():
            SubModule.SetLog(_Log)
        return self
    def Log(self, Content, Type="Unknown"):
        Param = self.Param
        log = DLUtils.param({
                "Subject": Param._PATH,
                "Type": Type,
                "Content": Content
            })
        if not hasattr(self, "_Log"):
            if not hasattr(self, "LogCache"):
                self.LogCache = []
            self.LogCache.append(log)
        else:
            self._Log.Add(log)
        return self

class AbstractModule(LogComponent):
    def __init__(self, Log=None, **Dict):
        self.Name = "_ABSTRACT_MODULE"
        self.SubModules = DLUtils.param()
        self.BindModules = DLUtils.param()
        Param = self.Param = DLUtils.Param()
        Param._CLASS = DLUtils.system.ClassPathStr(self)
        Param._PATH = "Root"
        if Log is not None:
            self._Log = Log
        self.SetParam(**Dict)
    def __call__(self, *List, **Dict):
        return self.CallMethod(*List, **Dict)
    def ExtractParam(self, RetainSelf=True):
        if hasattr(self, "Param"):
            Param = DLUtils.Param(self.Param)
            self.ExtractParamRecur(Param, RetainSelf)
            Attr = Param.delattrifexists("BindModules")
            return Param
        else:
            return "MODULE_WITHOUT_PARAM"
    def ExtractParamRecur(self, Param, RetainSelf):
        for Name, SubModule in self.SubModules.items():
            setattr(Param.SubModules, Name, SubModule.ExtractParam(RetainSelf))
        return self.Param
    def LoadParam(self, Param):
        self.Param = Param
        self._IsLoad = True
        self.SubModules = DLUtils.param()
        self.BindModules = DLUtils.param()
        self.SetEventDict()
        self.LoadParamRecur(Param)
        return self
    def LoadParamRecur(self, Param):
        for Name, SubModule in Param.SubModules.items():
            ModuleClass = DLUtils.python.ParseClass(SubModule._CLASS)
            SubModule._PATH = Param._PATH + "." + Name
            self.SubModules[Name] = ModuleClass().LoadParam(SubModule)
        return self
    def SetParam(self, Key=None, Value=None, **Dict):
        Param = self.Param
        UseSetParamMapDefault = Dict.setdefault("UseSetParamMapDefault", True)
        Map = {}
        if UseSetParamMapDefault:
            Map = SetParamMapDefault
        else:
            Map = {}

        if hasattr(self, "SetParamMap"):
            Map.update(self.SetParamMap)

        # for Key, Value in Dict.items():
        #     if Key in Map:
        #         Param.setattr(Map[Key], Value)
        #     else:
        #         Param.setattr(Key, Value)
        # return self
        if Key is not None or Value is not None:
            assert Value is not None
            assert Key is not None
            self._SetParam(Key, Value, Map)
        for _Key, _Value in Dict.items():
            self._SetParam(_Key, _Value, Map)
        return self
    def _SetParam(self, Key, Value, Map):
        Param = self.Param
        if Key in Map:
            Param.setattr(Map[Key], Value)
        else:
            Param.setattr(Key, Value)
        return self
    def SetAttr(self, **Dict):
        for Key, Value in Dict.items():
            setattr(self, Key, Value)
        return self
    def AddSubModule(self, Name=None, SubModule=None, **Dict):
        if Name is not None:
            assert SubModule is not None
            
            assert len(Dict) == 0
            self._AddSubModule(Name, SubModule)
        for _Name, _SubModule in Dict.items():
            self._AddSubModule(Name=_Name, SubModule=_SubModule)
        return self
    def _AddSubModule(self, Name, SubModule):
        Param = self.Param
        # detect overwrite module
        assert not self.SubModules.hasattr(Name)
        if hasattr(SubModule, "Param"):
            Param.SubModules.setattr(Name, SubModule.Param)
            SubModule.Param._PATH = Param._PATH + "." + Name
        else:
            Param.SubModules.setattr(Name, "MODULE_WITHOUT_PARAM")
        self.SubModules[Name] = SubModule
        setattr(self, Name, SubModule)
        return self
    def BindModule(self, Name=None, Module=None, **Dict):
        if Name is not None:
            assert Module is not None
            assert len(Dict) == 0
            self._BindModule(Name, Module)
        for _Name, _Module in Dict.items():
            self._BindModule(Name=_Name, Module=_Module)
        return self
    Bind = BindModule
    def _BindModule(self, Name, Module):
        Param = self.Param
        if hasattr(Module, "Param"):
            Param.BindModules.setattr(Name, Module.Param)
            #Module.Param._PATH = Param._PATH + "." + Name
        else:
            Param.BindModules.setattr(Name, "MODULE_WITHOUT_PARAM")
        self.BindModules[Name] = Module
        setattr(self, Name, Module)
        return self
    def UnBindModule(self, Name=None, *List):
        if Name is not None:
            # assert Module is not None
            # assert len(List) == 0
            self._UnBindModule(Name)
        for _Name, _Module in List:
            self._UnBindModule(Name=_Name)
        return self
    UnBind = UnBindModule
    Unbind = UnBindModule
    def _UnBindModule(self, Name):
        Param = self.Param
        Module = self.BindModules.getattr(Name)
        if Module is None:
            return self
        if hasattr(Module, "Param"):
            Param.BindModules.delattrifexists(Module.Param)
            #Module.Param._PATH = Param._PATH + "." + Name
        else:
            #Param.BindModules.delattrifexists(Name, "MODULE_WITHOUT_PARAM")
            pass
        if Name in self.BindModules:
            self.BindModules.pop(Name)
        if hasattr(self, Name):
            delattr(self, Name)
        return self
    _UnBind = _UnBindModule
    _Unbind = _UnBindModule
    _UnbindModule = _UnBindModule
    def GetSubModule(self, Name):
        return self.SubModules.getattr(Name)
    def RemoveSubModule(self, Name=None, SubModule=None):
        Param = self.Param
        if Name is not None:
            if Name in self.SubModules.keys():
                self.SubModules.pop(Name)
                Param.SubModules.delattr(Name)
                if hasattr(self, Name):
                    delattr(self, Name)
            else:
                warnings.warn("{0}.RemoveSubModule: No such SubModule: {1}".format(Param._PATH, Name))
        elif SubModule is not None:
            HasRemoved = False
            for Name, _SubModule in self.SubModules.items():
                if SubModule == _SubModule:
                    self.SubModules.pop(Name)
                    Param.SubModules.delattr(Name)
                    HasRemoved = True
                    break
                if hasattr(self, Name):
                    delattr(self, Name)
            if not HasRemoved:
                warnings.warn("{0}.RemoveSubModule: No such SubModule: {1}".format(Param._PATH, Name))
        else:
            raise Exception()
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
        DLUtils.file.EnsureFileDir(FilePath)
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
        if hasattr(self, "_ClassStr"):    
            return self._ClassStr
        else:
            return DLUtils.system.ClassPathStr(self)
    def SetName(self, Name, Recur=True):
        _PATH = self.Param._PATH
        PathList = _PATH.split(".")
        PathList[0] = Name
        self.Param._PATH = ".".join(PathList)
        if Recur:
            for SubModule in self.SubModules.values():
                SubModule.SetName(Name)
    def Rename(self, Name):
        self.SetName(Name)
    def ReName(self, Name):
        self.SetName(Name)
    def IsFixedRoot(self):
        if hasattr(self, "Param"):
            if hasattr(self.Param, "IsRoot"):
                if self.Param.IsRoot:
                    return True
        return False
    def SetRoot(self):
        self.Param.setattr("IsRoot", True)
        return self
    def SetEventDict(self):
        if not hasattr(self, "EventDict"):
            self.EventDict = DLUtils.param({
                "TensorMovement":[]
            })
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        if self.InitFinished(): # avoid double init
            return self
        if self.IsLoad():
            for Name, SubModule in self.SubModules.items():
                SubModule.Init(IsSuper=False, IsRoot=False)
                setattr(self, Name, SubModule)
        else:
            if IsRoot:
                self.Param._PATH = "Root"
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "Param"):
                    if not SubModule.IsFixedRoot():
                        SubModule.Param._PATH = self.Param._PATH + "." + Name
                SubModule.Init(IsSuper=False, IsRoot=False)
                setattr(self, Name, SubModule)
        self.SetEventDict()
        if hasattr(self, "_Log"):
            self.SetLogRecur(self._Log)
        self.SetConnectEvents()

        self.SetTrain()

        if hasattr(self, "Receive"):
            self.CallMethod = self.Receive
        elif hasattr(self, "forward"):
            self.CallMethod = self.forward
        else:
            # raise Exception()
            # pass
            pass
        if not hasattr(self, "Receive"):
            if hasattr(self, "__call__"):
                self.Receive = self.__call__
            elif hasattr(self, "forward"):
                self.Receive = self.forward
            else:
                # raise Exception()
                pass
        if IsRoot:
            self.SetTest()
        # assert hasattr(self, "CallMethod")
        self._InitFinished = True
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
        if not self.HandleTensorBySelf():
            if Param.hasattr("Tensor"):
                for TensorName in Param.Tensor:
                    if hasattr(self, TensorName):
                        TensorData = getattr(self, TensorName)
                        TensorNew = TensorData.to(Device).detach() # requires_grad remains same.
                        TensorNew.requires_grad = TensorData.requires_grad
                        setattr(self, TensorName, TensorNew)
                    else:
                        # Tensor = Param.Data.getattr(TensorName)
                        TensorData = Param.getattr(TensorName)
                        TensorNew = DLUtils.ToTorchTensor(TensorData).to(Device).detach()
                        TensorNew.requires_grad = TensorData.requires_grad
                        setattr(self, TensorName, TensorNew)
        self.SetDeviceRecur(Device, IsRoot=False)
        if IsRoot:
            self.OnTensorMovement()
            self.Log(
                f"Set device to {Device}", "SystemConfigChange"
            )
        return self
    def SetDeviceRecur(self, Device, IsRoot=True):
        for Name, BindModule in self.BindModules.items():
            if hasattr(BindModule, "SetDevice"):
                BindModule.SetDevice(Device, IsRoot=False)
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
                # Tensor = Param.Data.getattr(Name)
                Tensor = Param.getattr(Name)
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
    def SetTrain(self):
        self.IsTrain = True
        if hasattr(self, "ReceiveTrain"):
            self.Receive = self.ReceiveTrain
        return self
    def SetTest(self):
        self.IsTrain = False
        if hasattr(self, "ReceiveTest"):
            self.Receive = self.ReceiveTest
        return self
    def IsInit(self):
        return not self.IsLoad()
    def IsLoad(self):
        return hasattr(self, "_IsLoad") and self._IsLoad
    def InitFinished(self):
        return hasattr(self, "_InitFinished") and self._InitFinished
    def HandleTensorBySelf(self):
        return hasattr(self, "_HandleTensorBySelf") and self._HandleTensorBySelf
    def SetTrain(self, Recur=True):
        if Recur:
            for SubModule in self.SubModules.values():
                if hasattr(SubModule, "SetTrain"):
                    SubModule.SetTrain(Recur=True)
        return self
    def SetTest(self, Recur=True):
        if Recur:
            for SubModule in self.SubModules.values():
                if hasattr(SubModule, "SetTrain"):
                    SubModule.SetTest(Recur=True)
        return self

EmptyModule = AbstractModule
class AbstractOperator(AbstractModule):
    # operation module without trainable parameter
    def __init__(self, *List, **Dict):
        super().__init__(*List, **Dict)
    def AddSubModule(self, Name, SubModule):
        raise Exception("AbstractOperator module.")

class AbstractNetwork(AbstractModule):
    # network with trainable weights.
    def __init__(self, **Dict):
        super().__init__(**Dict)
        Param = self.Param
        Param.setemptyattr("TrainParam")
        Param.setemptyattr("Tensor")
    def SetTrainable(self, Name):
        Param = self.Param
        assert Param.Tensor.hasattr(Name)
        Path = Param.Tensor.getattr(Name)
        Param.TrainParam.setattr(Name, Path)
        return self
    def SetTrainParam(self, Name=None, Path=None, Value=None, **Dict):
        if Name is not None:
            assert Value is not None
            self._SetTrainParam(Name, Path, Value)
        for _Name, _Value in Dict.items():
            self._SetTrainParam(_Name, None, _Value)
        return self
    SetTrainableParam = SetTrainParam
    def _SetTrainParam(self, Name, Path=None, Value=None):
        Param = self.Param
        if Path is None:
            Path = "Data." + Name
        assert Value is not None
        if hasattr(Value, "requires_grad"):
            Value.requires_grad = True
        Param.TrainParam.setattr(Name, Path)
        Param.setattr(Path, Value)
        Param.Tensor.setattr(Name, Path)
        setattr(self, Name, Value)
        return self
    def SetUnTrainableParam(self, Name=None, Path=None, Value=None, Trainable=False, **Dict):
        if Trainable:
            self.SetTrainParam(Name=Name, Path=Path, Value=Value, **Dict)
            return self
        if Name is not None:
            assert Value is not None
            self._SetTensor(Name, Path, Value)
        for _Name, _Value in Dict.items():
            self._SetTensor(_Name, None, _Value)
        return self
    SetTensor = SetUntrainableParam = SetUnTrainableParam
    def _SetTensor(self, Name, Path=None, Value=None):
        if Path is None:
            Path = Name
        assert Value is not None
        Param = self.Param
        Param.Tensor.setattr(Name, Path)
        Param.setattr(Path, Value)
        return self
    def AddTensor(self, Name=None, Path=None, **Dict):
        if Name is not None:
            assert Path is not None
        for _Name, _Path in Dict.items():
            self._AddTensor(_Name, _Path)
        return self
    def _AddTensor(self, Name, Path=None):
        if Path is None:
            Path = Name
        Param = self.Param
        Param.Tensor.setattr(Name, Path)
        return self
    AddTensorName = AddTensor
    def AddTrainParam(self, Name, Path=None):
        if Path is None:
            Path = Name
        Param = self.Param
        Param.TrainParma.setattr(Name, Path)
        self.AddTensor(Name, Path)
        return self
    AddTrainParamName = AddTrainParam
    def GetTensor(self, Name):
        assert not self.HandleTensorBySelf()
        Param = self.Param
        # return Param.Data.getattr(Name)
        assert Param.Tensor.hasattr(Name)
        if hasattr(self, Name):
            return getattr(self, Name)
        else:
            Path = Param.Tensor.getattr(Name)
            return Param.getattr(Path)
    def UpdateTensorFromDict(self, Recur=False):
        Param = self.Param        
        if  self.HandleTensorBySelf():
            return
        if Param.hasattr("Tensor"):
            for Name, Path in Param.Tensor.items():
                # assert Param.Data.hasattr(Name)
                # Data = Param.Data.getattr(Name)
                assert Param.hasattr(Path)
                TensorData = Param.getattr(Path)
                Tensor = DLUtils.ToTorchTensorOrNum(TensorData)
                if hasattr(self, "Device"):
                    TensorNew = Tensor.to(self.Device).detach()
                    TensorNew.requires_grad = Tensor.requires_grad
                else:
                    TensorNew = Tensor
                setattr(self, Name, TensorNew)
        if Param.hasattr("TrainParam"):
            for Name, Path in Param.TrainParam.items():
                Tensor = getattr(self, Name)
                Tensor.requires_grad = True
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateTensorFromDict"):
                    SubModule.UpdateTensorFromDict(Recur=True)
        self.OnTensorMovement()
        return self
    def UpdateDictFromTensor(self, Recur=False):
        Param = self.Param
        if not self.HandleTensorBySelf():
            if Param.hasattr("TrainParam"):
                for Name, Path in Param.TrainParam.items():
                    if hasattr(self, Name):
                        TrainParamData = getattr(self, Name)
                        Param.setattr(Path, DLUtils.ToNpArray(TrainParamData))
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateDictFromTensor"):
                    SubModule.UpdateDictFromTensor(Recur=True)
        return self
    def ExtractTrainParam(self, TrainParam={}, PathStrPrefix=True, Recur=True):
        self.UpdateDictFromTensor(Recur=False)
        # self.UpdateTensorFromDict()
        Param = self.Param
        if PathStrPrefix:
            Prefix = self.PathStr() + "."
        else:
            Prefix = ""
        if Param.get("TrainParam") is not None:
            for Name, Path in Param.TrainParam.items():
                TrainParam[Prefix + Name] = getattr(self, Name)
        if Recur:
            self.ExtractTrainParamRecur(TrainParam=TrainParam, PathStrPrefix=PathStrPrefix)
        return TrainParam
    def ExtractTrainParamRecur(self, TrainParam={}, PathStrPrefix=True):
        for Name, SubModule in self.SubModules.items():
            if hasattr(SubModule, "ExtractTrainParam"):
                SubModule.ExtractTrainParam(
                    TrainParam=TrainParam,
                    PathStrPrefix=PathStrPrefix,
                    Recur=True
                ) 
        return TrainParam
    def PlotWeight(self, SaveDir=None, SaveName=None):
        Param = self.Param
        Param = self.ExtractParam()
        SavePath = DLUtils.ParseSavePath(SaveDir, SaveName, SaveNameDefault=Param._PATH)
        if Param.hasattr("TrainParam"):
            for WeightName in Param.TrainParam:
                # Data = Param.Data.getattr(WeightName)
                Data = Param.getattr(WeightName)
                if hasattr(Data, "shape"):
                    DimNum = len(Data.shape)
                else:
                    DimNum = 0
                if DimNum == 1 or DimNum == 0:
                    # DLUtils.plot.PlotData1D(
                    #     Name=WeightName,
                    #     Data=Data,
                    #     SavePath=SavePath + "." + WeightName + ".svg",
                    #     #XLabel="Dimension 0", YLabel="Dimension 0"
                    # )
                    DLUtils.plot.PlotDataAndDistribution1D(
                        Name=WeightName,
                        Data=Data,
                        SavePath=SavePath + "." + WeightName + " - Distribution.svg",
                        XLabel="Dimension 0", YLabel="Dimension 0",
                        Title=f"Shape {Data.shape[0]}"
                    )
                elif DimNum == 2:
                    # DLUtils.plot.PlotData2D(
                    #     Name=WeightName,
                    #     Data=Data,
                    #     SavePath=SavePath + "." + WeightName + ".svg",
                    #     XLabel="Output Dimension", YLabel="Input Dimension"
                    # )
                    DLUtils.plot.PlotDataAndDistribution2D(
                        Name=WeightName,
                        Data=Data,
                        SavePath=SavePath + "." + WeightName + " - Distribution.svg",
                        XLabel="Output Dimension", YLabel="Input Dimension",
                        TitlePlot=f"Shape {Data.shape[0], Data.shape[1]}"
                    )
            self.PlotWeightRecur(SaveDir, SaveName)
        return self
    def ExtractParam(self, RetainSelf=True):
        Param = DLUtils.Param(self.Param)
        self.UpdateDictFromTensor(Recur=False)
        if not RetainSelf:
            # prune some empty nodes for readability and storage space saving.
            if Param.hasattr("TrainParam") and len(Param.TrainParam) == 0:
                Param.delattr("TrainParam")
        Attr = Param.delattrifexists("BindModules")
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

        self.UpdateTensorFromDict()
        if IsRoot:
            self.Log(f"{Param._CLASS}: initialization finished.", Type="initialization")
        return self
    def OnTensorMovement(self):
        for Event in self.EventDict.TensorMovement:
            Event(Model=self, Param=self.ExtractTrainParam())
    def parameters(self):
        return

class TorchModuleWrapper(AbstractNetwork):
    def __init__(self, *List, **Dict):
        self._HandleTensorBySelf = True
        super().__init__(*List, **Dict)
    def SetTest(self, Recur=True):
        self.module.eval()
        return super().SetTest(Recur=Recur)
    def SetTrain(self, Recur=True):
        self.module.train()
        return super().SetTrain(Recur=Recur)
    def SetDevice(self, Device=None, IsRoot=True):
        self.module = self.module.to(Device)
        return super().SetDevice(Device, IsRoot)
    def UpdateModuleFromParam(self):
        #raise Exception() # must be implemented by child class
        Param = self.Param
        Dict = self.module.state_dict()
        if hasattr(self, "StateDictMap"):
            for Key, Value in self.StateDictMap.items():
                assert Value in Dict
                if Param.hasattr(Key):
                    Dict[Value] = DLUtils.ToRunFormat(Param.getattr(Key)) 
        self.module.load_state_dict(Dict)
    def UpdateDictFromTensor(self, Recur=False):
        self.UpdateModuleFromParam()
        return super().UpdateDictFromTensor(Recur)
    def UpdateParamFromModule(self):
        Param = self.Param
        Dict = self.module.state_dict()
        if hasattr(self, "StateDictMap"):
            for Key, Value in self.StateDictMap.items():
                if Value in Dict:
                    Param.setattr(Key, DLUtils.ToSaveFormat(Dict[Value]))
        self.module.load_state_dict(Dict)
    def UpdateTensorFromDict(self, Recur=False):
        self.UpdateParamFromModule()
        return super().UpdateTensorFromDict(Recur)

SetParamMapDefault = DLUtils.IterableKeyToElement({
    ("InNum", "InputNum"): "In.Num",
    ("InType", "InputType"): "In.Type",
    ("OutNum", "OutputNum"): "Out.Num",
    ("OutType", "OutputType"): "Out.Type",
    ("EpochNum"): "Epoch.Num",
    ("BatchNum"): "Batch.Num",
    ("BatchSize"): "Batch.Size",
    ("AfterOperation"): "Operation.After"
})

class AbstractModuleGroup(AbstractNetwork):
    def __init__(self, *List, **Dict):
        super().__init__(**Dict)
        if len(List) == 0:
            ModuleList = Dict.get("ModuleList")
        else:
            assert Dict.get("ModuleList") is None
            if len(List) == 1 and DLUtils.IsIterable(List[0]):
                if isinstance(List[0], dict):
                    ModuleList = List[0]
                else:
                    ModuleList = List[0]
            else:
                ModuleList = List
        self.ModuleList = []
        if ModuleList is not None:
            if isinstance(ModuleList, tuple):
                ModuleList = list(ModuleList)
            assert isinstance(ModuleList, list) or isinstance(ModuleList, dict)
            self.SetModuleList(ModuleList)
    def SetModuleList(self, ModuleList):
        Param = self.Param
        if isinstance(ModuleList, list):
            for Index, SubModule in enumerate(ModuleList):
                self.AddSubModule(f"L{Index}", SubModule)
            self.ModuleList = ModuleList
        if isinstance(ModuleList, dict):
            for Name, SubModule in ModuleList.items():
                self.AddSubModule(
                    Name, SubModule
                )
            self.ModuleList = list(ModuleList.values())
        return self
    def Init(self, IsSuper=False, IsRoot=True):
        Param = self.Param
        if self.IsLoad():
            self.ModuleList = list(self.SubModules.values())
        self.ModuleNum = Param.Module.Num = len(self.ModuleList)
        return super().Init(IsSuper=True, IsRoot=IsRoot)