import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
else:
    torch = DLUtils.GetLazyTorch()

class TorchModule(torch.nn.Module):
    def FromAbstractModule(self, Module):
        self.Module = Module
        for Name, Param in Module.ExtractTrainParam().items():
            # Param: torch.nn.Parameter
            Name = Name.replace(".", "|")
            # Param = torch.nn.Parameter(Param)
            self.register_parameter(name=Name, param=Param)
        self.forward = Module.Receive
        self.Parent = Module
        return self
    def forward(self, *List, **Dict):
        return self.Parent.Receive(*List, **Dict)
    def ExtractTrainParam(self):
        return dict(self.named_parameters())
    def ToFile(self, FilePath, RetainSelf=True):
        self.Parent.ToFile(FilePath, RetainSelf=RetainSelf)
        return self
    def SetTrain(self):
        self.Parent.SetTrain()
        self.train()
    def SetTest(self):
        self.Parent.SetTest()
        self.eval()

class TorchModelWrapper():
    def __init__(self, Class:torch.nn.Module=None, *List, **Dict):
        param = self.param = DLUtils.Param()
        param.ModuleInitArgList = list(List)
        param.ModuleInitArgDict = dict(Dict)
        param.ModuleClassStr = DLUtils.python.ClassPathStr(Class)
        self.module = Class(*List, **Dict)
    def ToFile(self, FilePath, RetainSelf=False):
        DLUtils.file.Obj2BinaryFile(
            {
                "state_dict": TorchModelStateDict2CPU(self.module.state_dict()),
                "param": self.param
            },
            FilePath
        )
    def FromFile(self, FilePath):
        ModelData = DLUtils.file.BinaryFile2Obj(FilePath)
        Class = DLUtils.python.ClassPathStr2Class()
        assert isinstance(Class, torch.nn.Module)
        self.module = Class(
            *self.param.ModuleInitArgList, **self.param.ModuleInitArgDict
        )
        self.param = ModelData["param"]
        self.module.load_state_dict(
            ModelData["state_dict"]
        )
        return self.module
    def LoadModel(self, state_dict):
        if hasattr(self, "module"):
            module = self.module
            self.module.load_state_dict(state_dict)
            return self.module
        else:
            param = self.param
            Class = DLUtils.python.ClassPathStr2Class(param.ModuleClassStr)
            self.module = Class(
                *self.param.ModuleInitArgList, **self.param.ModuleInitArgDict
            )
            self.module.load_state_dict(state_dict)
            return self.module

class TorchModuleParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, TorchModule, *List, **Dict):
        super().__init__(TorchModule, *List, **Dict)
        self.Parent = TorchModule.Parent
        self.TorchModule = TorchModule
    def ToFile(self, FilePath, RetainSelf=True):
        self.Parent.ToFile(FilePath, RetainSelf=RetainSelf)
        return self
    def Clear(self):
        self.Parent.Clear()
    def FromFile(self, FilePath):
        raise Exception() # not supported
        return self
    def FromFileAndInit(self, FilePath, Device=None):
        self.Parent.Clear()
        self.Parent.FromFile(FilePath)
        self.Parent.Init()
        TorchModel = self.Parent.TorchModel()
        if Device is not None:
            TorchModel.to(Device)
        return TorchModuleParallel(TorchModel)
    def SetTrain(self):
        self.Parent.SetTrain()
        self.train()
    def SetTest(self):
        self.Parent.SetTest()
        self.eval()
    def forward(self, *List, **Dict):
        return self.Parent.Receive(*List, **Dict)


from ...module.abstract_network import AbstractNetwork
class TorchModuleWrapper(AbstractNetwork):
    def __init__(self, *List, **Dict):
        self._HandleTensorBySelf = True
        super().__init__(*List, **Dict)
    def ExtractParam(self, RetainSelf=True):
        if hasattr(self, "Param"):
            Param = DLUtils.Param(self.Param)
            module_dict = self.module.state_dict()
            module_dict = StateDict2CPU(module_dict)
            Param.Module.Data = module_dict
            self.ExtractParamRecur(Param, RetainSelf)
            Param.delattrifexists("BindModules")
            return Param
        else:
            return "MODULE_WITHOUT_PARAM"
    def ExtractTrainParam(self, TrainParamDict={}, PathStrPrefix=True, Recur=True):
        assert isinstance(self.module, torch.nn.Module)
        self.UpdateDictFromTensor(Recur=False)
        Param = self.Param
        if PathStrPrefix:
            Prefix = self.PathStr() + "."
        else:
            Prefix = ""
        if Param.get("TrainParam") is not None:
            for Name, Param in self.module.named_parameters():
                TrainParamDict[Prefix + Name] = Param
        if Recur:
            self.ExtractTrainParamRecur(TrainParamDict=TrainParamDict, PathStrPrefix=PathStrPrefix)
        return TrainParamDict
    def SetTest(self, Recur=True):
        self.module.eval()
        return super().SetTest(Recur=Recur)
    def SetTrain(self, Recur=True):
        self.module.train()
        return super().SetTrain(Recur=Recur)
    def SetDevice(self, Device=None, IsRoot=True):
        self.module.to(Device)
        return super().SetDevice(Device, IsRoot)
    def UpdateModuleFromParam(self):
        Param = self.Param
        module_dict = self.module.state_dict()
        if hasattr(self, "ModuleParamMap"):
            for Key, Value in self.ModuleParamMap.items():
                assert Value in module_dict
                if Param.hasattr(Key):
                    module_dict[Value] = DLUtils.ToRunFormat(Param.getattr(Key))
                else:
                    raise Exception()
        else:
            raise Exception()
        self.module.load_state_dict(module_dict)
        return self
    def UpdateParamFromModule(self):
        Param = self.Param
        module_dict = self.module.state_dict()
        if hasattr(self, "ModuleParamMap"):
            for Key, Value in self.ModuleParamMap.items():
                if Value in module_dict:
                    Param.setattr(Key, DLUtils.ToSaveFormat(module_dict[Value]))
                else:
                    raise Exception()
        self.module.load_state_dict(module_dict)
        return self
    def UpdateTensorFromDict(self, Recur=False):
        self.UpdateParamFromModule()
        return super().UpdateTensorFromDict(Recur=Recur)
    def UpdateDictFromTensor(self, Recur=False):
        self.UpdateModuleFromParam()
        return super().UpdateDictFromTensor(Recur=Recur)
    def Build(self, IsSuper=False, IsRoot=True):
        assert hasattr(self, "module")
        Param = self.Param
        if self.IsInit():
            Param.setdefault("Mode", "Wrap")
            # wrap: save and load model state_dict
        Mode = self.Mode = Param.Mode
        if Mode in ["Wrap"]:
            if self.IsInit():
                # Param.Module.Data = self.module.state_dict()
                pass
            else:
                self.module.load_state_dict(Param.Module.Data)
    
        if self.IsInit():
            assert hasattr(self, "module")
        else:
            assert hasattr(self, "module")

        return super().Init(IsSuper=True, IsRoot=IsRoot)