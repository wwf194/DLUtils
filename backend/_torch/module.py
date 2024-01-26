from __future__ import annotations
import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn

else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")

import typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._module import TorchModule, TorchModuleParallel

def __getattr__(Name):
    if Name in ["TorchModule"]:
        from .module import TorchModule as _TorchModule
        global TorchModule
        TorchModule = _TorchModule
        return TorchModule
    elif Name in ["TorchModuleWrapper"]:
        from .module import TorchModuleWrapper as _TorchModuleWrapper
        global TorchModuleWrapper
        TorchModuleWrapper = _TorchModuleWrapper
        return TorchModuleWrapper
    elif Name in ["_TorchModule"]:
        from .module import TorchModuleParallel as _TorchModuleParallel
        global TorchModuleParallel
        TorchModuleParallel = _TorchModuleParallel
        return TorchModuleParallel
    else:
        raise Exception(Name)

def ChangeTorchModuleParameter(module:torch.nn.Module):
    ParamDict = dict(module.named_parameters())
    ParamDict

def ReportTorchInfo(): # print info about training environment, global variables, etc.
    print("Torch version: ",torch.__version__)
    print("Is CUDA enabled: ",torch.cuda.is_available())

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

def GetLearningRate(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']
GetLR = GetLearningRate

def TorchModelStateDict2CPU(state_dict):
    return StateDict2CPURecur(state_dict)
StateDict2CPU = TorchModelStateDict2CPU

def StateDictToCPURecur(Obj):
    if isinstance(Obj, dict):
        ObjCPU = {}
        for Key, Value in Obj.items():
            ObjCPU[Key] = StateDictToCPURecur()
    elif isinstance(Obj, list):
        ObjCPU = []
        for Index, Item in enumerate(Obj):
            ObjCPU.append(StateDictToCPURecur(Item))
    elif isinstance(Value, torch.Tensor) or hasattr(Value, "cpu"):
        ObjCPU = Obj.cpu()
    else:
        ObjCPU = Obj
    return ObjCPU
StateDict2CPURecur = StateDictToCPURecur

def TorchModel2File(Model, FilePath):
    DLUtils.file.EnsureFileDir(FilePath)
    torch.save(Model.state_dict(), FilePath)

def File2TorchModel(Class, FilePath, *List, **Dict):
    FilePath = DLUtils.file.CheckFileExists(FilePath)
    state_dict = torch.load(FilePath)
    Model = Class(*List, **Dict)
    Model.load_state_dict(state_dict)
    return Model
    
def TorchModelWithAdditionalParamToFile(Model:TorchModule, FilePath):
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