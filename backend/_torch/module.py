import torch
import numpy as np
import DLUtils

def ChangeTorchModuleParameter(module:torch.nn.Module):
    ParamDict = dict(module.named_parameters())
    ParamDict

def ReportTorchInfo(): # print info about training environment, global variables, etc.
    return torch.pytorch_info()

def PrintOptimizerStateDict(Optimizer, Out="Std"):
    StateDict = Optimizer.state_dict()
    """
    Adam.state_dict():
        "state": {
            0 : { // key is int, starting from 0.
                "step": tensor(391.),
                "exp_avg": historical average. tensor, same shape as weight tensor
                "exp_avg_sq": historical sq. tensor, same shape as weight tensor
            },
            1 : {
            },
            ...
        },
        "param_groups": [
                {
                    "lr": float,
                    "betas" ...,
                    "eps": ...,
                    "weight_decay": ...,
                    "amsgrad": ...,
                    "maximize": ...,
                    "foreach": ...,
                    "capturable": ...,
                    "params": [
                        0, 1, 2, ... //与Adam.state_dict()["state"].keys()一致
                        //为与Adam.param_groups()[0]列表的index
                    ]
                }, //param_group_1
                {
                }, //param_group_2
                {
                }
            ]
    Adam.param_groups():
        [
            {
                "lr": float,
                "betas" ...,
                "eps": ...,
                "weight_decay": ...,
                "amsgrad": ...,
                "maximize": ...,
                "foreach": ...,
                "capturable": ...,
                "params": [ //Adam.state_dict()["state"].keys()就是这个列表的index
                    tensor0, 
                    tensor1,
                    tensor2,
                    ... 
                ]
            }, //param_group_1
            {
            }, //param_group_2
            {
            }
        ]
    """
    DLUtils.PrintDict(StateDict, Out=Out)
    return

def GetLR(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def TorchModelStateDict2CPU(state_dict):
    return StateDict2CPURecur(state_dict)
StateDict2CPU = TorchModelStateDict2CPU

def StateDict2CPURecur(Obj):
    if isinstance(Obj, dict):
        ObjCPU = {}
        for Key, Value in Obj.items():
            ObjCPU[Key] = StateDict2CPURecur()
    elif isinstance(Obj, list):
        ObjCPU = []
        for Index, Item in enumerate(Obj):
            ObjCPU.append(StateDict2CPURecur(Item))
    elif isinstance(Value, torch.Tensor) or hasattr(Value, "cpu"):
        ObjCPU = Obj.cpu()
    else:
        ObjCPU = Obj
    return ObjCPU

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
    def Init(self, IsSuper=False, IsRoot=True):
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


import DLUtils
def TorchModel2File(Model, FilePath):
    DLUtils.file.EnsureFileDir(FilePath)
    torch.save(Model.state_dict(), FilePath)

def File2TorchModel(Class, FilePath, *List, **Dict):
    FilePath = DLUtils.file.CheckFileExists(FilePath)
    state_dict = torch.load(FilePath)
    Model = Class(*List, **Dict)
    Model.load_state_dict(state_dict)
    return Model
    
def TorchModelWithAdditionalParam2File(Model, FilePath):
    if hasattr(Model, "ExtractParam"):
        Param = Model.ExtractParam()
    else:
        Param = Model.Param
    ModelData = {
        "state_dict": TorchModelStateDict2CPU(Model.state_dict()),
        "param": Param
    }
    DLUtils.Obj2BinaryFile(ModelData, FilePath)

def File2TorchModelWithAdditionalParam(Class, FilePath, *List, **Dict):
    ModelData = DLUtils.BinaryFile2Obj(FilePath)
    state_dict = ModelData["state_dict"]
    Param = ModelData["param"]
    if hasattr(Model, "LoadParam"):
        Model.LoadParam(Param)
        Model = Class(*List, **Dict)
        Model.load_state_dict(state_dict)
    else:
        Model = Class(param=Param, *List, **Dict)
        Model.load_state_dict(state_dict)
    return Model