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

def StateDict2CPU(state_dict):
    return StateDict2CPURecur(state_dict)

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

from ..module.abstract_network import AbstractNetwork
class TorchModuleWrapper(AbstractNetwork):
    def __init__(self, *List, **Dict):
        self._HandleTensorBySelf = True
        super().__init__(*List, **Dict)
    def ExtractParam(self, RetainSelf=True):
        if hasattr(self, "Param"):
            Param = DLUtils.Param(self.Param)
            module_dict = self.module.state_dict()
            module_dict = StateDict2CPU(module_dict)
            Param.Module.Data = self.module.state_dict()
            self.ExtractParamRecur(Param, RetainSelf)
            Attr = Param.delattrifexists("BindModules")
            return Param
        else:
            return "MODULE_WITHOUT_PARAM"
    def ExtractTrainParam(self, TrainParam={}, PathStrPrefix=True, Recur=True):
        assert isinstance(self.module, torch.nn.Module)
        self.UpdateDictFromTensor(Recur=False)
        Param = self.Param
        if PathStrPrefix:
            Prefix = self.PathStr() + "."
        else:
            Prefix = ""
        if Param.get("TrainParam") is not None:
            for Name, Param in self.module.named_parameters():
                TrainParam[Prefix + Name] = Param
        if Recur:
            self.ExtractTrainParamRecur(TrainParam=TrainParam, PathStrPrefix=PathStrPrefix)
        return TrainParam
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
            assert hasattr()
            if self.IsLoad():
                self.module.load_state_dict(Param.Module.Data)
            else:
                Param.Module.Data = self.module.state_dict()
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
        return self.Receive(*List, **Dict)
    def ExtractTrainParam(self):
        return self
    def ToFile(self, FilePath, RetainSelf=True):
        self.Parent.ToFile(FilePath, RetainSelf=RetainSelf)
        return self


def ToTorchTensor(Data, Device=None):
    if isinstance(Data, np.ndarray):
        _Data = NpArray2Tensor(Data)
    elif isinstance(Data, list):
        _Data = NpArray2Tensor(List2NpArray(Data))
    elif isinstance(Data, torch.Tensor):
        _Data = Data
    else:
        raise Exception(type(Data))
    if Device is not None:
        _Data = _Data.to(Device)
    return _Data

def ToTorchTensorOrNum(data):
    if isinstance(data, float):
        return data
    elif isinstance(data, int):
        return data
    else:
        return ToTorchTensor(data)

def NpArray2TorchTensor(data, Location="cpu", DataType=torch.float32, RequiresGrad=False):
    data = torch.from_numpy(data)
    data = Tensor2GivenDataType(data, DataType)
    data = data.to(Location)
    data.requires_grad = RequiresGrad
    return data

NpArray2Tensor = NpArray2TorchTensor