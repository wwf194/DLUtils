

import DLUtils
# import DLUtils.module.AbstractModule as AbstractModule
from .AbstractModule import AbstractModule
from .AbstractModule import AbstractNetwork

import DLUtils.module.loss as loss

def BuildModuleFromType(Type):
    module = DLUtils.transform.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module
    
    module = DLUtils.loss.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module
    
    module = DLUtils.dataset.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module

    module = DLUtils.optimize.BuildModuleIfIsLegalType(Type)
    if module is not None:
        return module

    raise Exception()

def BuildModule(param, **kw):
    if hasattr(param, "ClassPath"):
        try:
            Class = DLUtils.parse.ParseClass(param.ClassPath)
            return Class(**kw)
        except Exception:
            DLUtils.AddWarning("Cannot parse ClassPath: %s"%param.ClassPath)
    # if param.Type in ['transform.RNNLIF']:
    #     print("aaa")
    module = DLUtils.transform.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module
    
    module = DLUtils.loss.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module
    
    module = DLUtils.dataset.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module

    module = DLUtils.optimize.BuildModuleIfIsLegalType(param, **kw)
    if module is not None:
        return module

    module = BuildExternalModule(param, **kw)
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
    
import torch

