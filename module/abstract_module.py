import warnings
import DLUtils
from ..module import LogComponent

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
    def GetParamMap(self, UseParamMapDefault=True):
        if UseParamMapDefault or \
            hasattr(self, "UseParamMapDefault") and self.UseParamMapDefault:
            ParamMap = GetParamMapDefault()
        else:
            ParamMap = {}
        if hasattr(self, "ParamMap"):
            ParamMap.update(self.ParamMap)
        return ParamMap
    def SetParam(self, Key=None, Value=None, **Dict):
        Param = self.Param
        # UseParamMapDefault = Dict.setdefault("UseParamMapDefault", True)
        
        # Map = {}
        # if UseParamMapDefault:
        #     Map = dict(ParamMapDefault)
        # else:
        #     Map = {}

        ParamMap = self.GetParamMap()

        # if hasattr(self, "ParamMap"):
        #     ParamMap.update(self.ParamMap)

        # for Key, Value in Dict.items():
        #     if Key in Map:
        #         Param.setattr(Map[Key], Value)
        #     else:
        #         Param.setattr(Key, Value)
        # return self
        if Key is not None or Value is not None:
            assert Value is not None
            assert Key is not None
            self._SetParam(Key, Value, ParamMap)
        for _Key, _Value in Dict.items():
            self._SetParam(_Key, _Value, ParamMap)
        return self
    def _SetParam(self, Key, Value, ParamMap):
        Param = self.Param
        if Key in ParamMap:
            Param.setattr(ParamMap[Key], Value)
        else:
            Param.setattr(Key, Value)
        return self
    def UpdateParam(self, Param):
        if isinstance(Param, dict):
            self.Param.absorb_dict(Dict)
            return self
        elif isinstance(Param, DLUtils.param):
            self.Param.Absorb(Param)
        return self
    def UpdateTensorFromDict(self, Recur=False):
        Param = self.Param        
        if self.HandleTensorBySelf():
            pass
        else:
            if Param.hasattr("Tensor"):
                for Name, Path in Param.Tensor.items():
                    assert Param.hasattr(Path)
                    TensorData = Param.getattr(Path)
                    Tensor = DLUtils.ToTorchTensorOrNum(TensorData)
                    if hasattr(self, "Device"):
                        TensorDevice = Tensor.to(self.Device).detach()
                        TensorDevice.requires_grad = Tensor.requires_grad
                    else:
                        TensorDevice = Tensor
                    setattr(self, Name, TensorDevice)
            if Param.hasattr("TrainParam"):
                for Name, Path in Param.TrainParam.items():
                    assert Param.Tensor.hasattr(Name)
                    Tensor = getattr(self, Name)
                    Tensor.requires_grad = True
                    Tensor = torch.nn.Parameter(Tensor, requires_grad=True)
                    setattr(self, Name, Tensor)
        if Recur:
            for Name, SubModule in self.SubModules.items():
                if hasattr(SubModule, "UpdateTensorFromDict"):
                    SubModule.UpdateTensorFromDict(Recur=True)
        self.OnTensorMovement()
        return self
    def UpdateDictFromTensor(self, Recur=False):
        Param = self.Param
        if self.HandleTensorBySelf():
            pass
        else:
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
    def AbsorbParam(self, Param, Prefix=None, Remove=True, PrefixOnly=False):
        ParamMap = self.GetParamMap()

        if Prefix is not None:
            PrefixLength = len(Prefix)
            if PrefixOnly:
                assert isinstance(Prefix, str)
                for Key, Value in Param.iteritems():
                    if Key.startswith(Prefix) and len(Key) > PrefixLength:
                        self.Param.setattr(Key[PrefixLength:], Value)
                        if Remove:
                            Param.deleteattr(Key)
                return self
            else:
                for Key, Value in Param.iteritems():
                    Sig = False
                    if Key.startswith(Prefix) and len(Key) > PrefixLength:
                        self.Param.setattr(Key[PrefixLength], Value)
                        Sig
                    elif Key in ParamMap:
                        self.Param.setattr(ParamMap.getattr(Key), Value)
                        Sig = True

                    if Sig:
                        if Remove:
                            Param.deleteattr(Key)
                return self

        for Key, Value in Param.iteritems():
            if Key in ParamMap:
                self.Param.setattr(ParamMap.getattr(Key), Value)
                if Remove:
                    Param.deleteattr(Key)
            else:
                self.Param.setattr(Key, Value)
                if Remove:
                    Param.deleteattr(Key)
        return self
    def SetAttr(self, **Dict):
        for Key, Value in Dict.items():
            setattr(self, Key, Value)
        return self
    def HasSubModule(self, Name):
        return self.Param.SubModules.hasattr(Name)
    def HasBindModule(self, Name):
        return self.BindModules.hasattr(Name)
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
    # 1D dropout
    def _DropOut(self, X):
        import torch
        import torch.nn.functional as F
        return F.dropout(X, self.DropOutProbability, training=self.IsTrain(), inplace=self.DropOutInPlace)
    def _DropOutNull(self, X):
        return X
    def SetDropOut(self):
        Param = self.Param
        # dropout setting
        if Param.DropOut.Enable:
            self.DropOut = self._DropOut
            self.DropOutInPlace = Param.DropOut.InPlace
        else:
            self.DropOut = self._DropOutNull
        return self
    def SetDropOutInit(self):
        Param = self.Param
        if Param.DropOut.hasattr("Probability"):
            assert isinstance(Param.DropOut.Probability, float)
            Param.DropOut.Enable = True
            if Param.DropOut.Probability == 0.0:
                Param.DropOut.Enable = False
        Param.DropOut.setdefault("Enable", False)
        if Param.DropOut.Enable:
            Param.DropOut.setdefault("Probability", 0.1)
        Param.DropOut.setdefault("InPlace", True) # could help save some memory
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
        self.SetTrain() # default : train mode. might influence dropout etc.

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
        self.UpdateTensorFromDict()
        self._InitFinished = True
        return self
    def LogWithSelfInfo(self, Content, Type="Unknown"):
        self.Log(f"{self.PathStr()}({self.ClassStr()}): {Content}", Type=Type)
        return self
    def SetDevice(self, Device=None, IsRoot=True):
        import torch
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
                        if Param.TrainParam.hasattr(TensorName):
                            TensorNew = torch.nn.Parameter(TensorNew, requires_grad=True)
                        setattr(self, TensorName, TensorNew)
                    else:
                        # Tensor = Param.Data.getattr(TensorName)
                        TensorData = Param.getattr(TensorName)
                        TensorNew = DLUtils.ToTorchTensor(TensorData).to(Device).detach()
                        TensorNew.requires_grad = TensorData.requires_grad
                        if Param.TrainParam.hasattr(TensorName):
                            TensorNew = torch.nn.Parameter(TensorNew, requires_grad=True)
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
        self._IsTrain = True
        if hasattr(self, "ReceiveTrain"):
            self.Receive = self.ReceiveTrain
        return self
    def SetTest(self):
        self._IsTrain = False
        if hasattr(self, "ReceiveTest"):
            self.Receive = self.ReceiveTest
        return self
    def IsTrain(self):
        return self._IsTrain
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
    def TorchModel(self):
        return DLUtils.module.TorchModule().FromAbstractModule(self)

EmptyModule = AbstractModule

from ..utils._dict import IterableKeyToElement, ExpandIterableKey

def GetParamMapDefault():
    return ExpandIterableKey({
        ("InNum", "InputNum"): "In.Num",
        ("InType", "InputType"): "In.Type",
        ("OutNum", "OutputNum"): "Out.Num",
        ("OutType", "OutputType"): "Out.Type",
        ("EpochNum"): "Epoch.Num",
        ("BatchNum"): "Batch.Num",
        ("BatchSize"): "Batch.Size",
        ("AfterOperation"): "Operation.After"
    })
ParamMapDefault = GetParamMapDefault()