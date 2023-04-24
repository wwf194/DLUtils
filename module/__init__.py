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

from .abstract_module import AbstractModule

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
        self._RegisterTensor(Name, Path)
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
    def RegisterTensor(self, Name=None, Path=None, **Dict):
        if Name is not None:
            assert Path is not None
            self._RegisterTensor(Name, Path)
        for _Name, _Path in Dict.items():
            self._RegisterTensor(_Name, _Path)
        return self
    def _RegisterTensor(self, Name, Path=None):
        if Path is None:
            Path = Name
        Param = self.Param
        Param.Tensor.setattr(Name, Path)
        return self
    RegisterTensorName = RegisterTensor
    def RegisterTrainParam(self, Name, Path=None):
        if Path is None:
            Path = Name
        Param = self.Param
        Param.TrainParam.setattr(Name, Path)
        self.RegisterTensor(Name, Path)
        return self
    RegisterTrainParamName = RegisterTrainParam
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
        super().Init(IsSuper=True, IsRoot=IsRoot)
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


from .module_graph import ModuleGraph
from .module_series import ModuleList, ModuleSeries, _ModuleList, _ModuleSeries
