import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_torch
from utils_torch.attrs import *

DefaultRoutings = [
    "&GetBias |--> bias",
    "&GetNoise |--> noise",
    "hiddenState, input, noise, bias |--> &Add |--> inputTotal",
    "inputTotal, cellState |--> &ProcessInputTotalAndcellState |--> cellState",
    "cellState |--> &NonLinear |--> hiddenState",
    "hiddenState |--> &HiddenStateTransform |--> hiddenState",
    "cellState |--> &CellStateDecay |--> cellState"
]

class RecurrentLIFLayer(nn.Module):
    def __init__(self, param=None, data=None):
        super(RecurrentLIFLayer, self).__init__()
        utils_torch.model.InitForModel(self, param, data, ClassPath="utils_torch.Models.RecurrentLIFLayer")
    def InitFromParam(self, param=None, IsLoad=False):
        if param is None:
            param = self.param
            data = self.data
            cache = self.cache
        else:
            self.param = param
            self.data = utils_torch.EmptyPyObj()
            self.cache = utils_torch.EmptyPyObj()
        
        cache.IsLoad = IsLoad
        cache.IsInit = not IsLoad
        cache.Modules = utils_torch.EmptyPyObj()
        
        EnsureAttrs(param, "IsExciInhi", default=False)
        cache.Tensors = []
        self.BuildModules()
        self.InitModules()
        self.SetInternalMethods()
        self.ParseRouters()

    def BuildModules(self):
        param = self.param
        cache = self.cache
        for Name, ModuleParam in ListAttrsAndValues(param.Modules, Exceptions=["__ResolveRef__"]):
            # if Name in ["GetBias"]:
            #     print("aaa")
            if hasattr(ModuleParam, "Type") and ModuleParam.Type in ["Internal"]:
                continue
            setattr(ModuleParam, "Name", Name)
            setattr(ModuleParam, "FullName", param.FullName + "." + Name)
            Module = utils_torch.model.BuildModule(ModuleParam)
            setattr(cache.Modules, Name, Module)
            if isinstance(Module, nn.Module):
                self.add_module(Name, Module)
    def InitModules(self):
        param = self.param
        cache = self.cache
        for Module in ListValues(cache.Modules):
            if hasattr(Module, "InitFromParam"):
                Module.InitFromParam()
            else:
                utils_torch.AddWarning("Module %s has not implemented InitFromParam method."%Module)
    def SetInternalMethods(self):
        param = self.param
        cache = self.cache
        Modules = cache.Modules
        if param.IsExciInhi:
            if cache.IsInit:
                if not (HasAttrs(param, "TimeConst.Excitatory") and HasAttrs(param, "TimeConst.Inhibitory")):
                    EnsureAttrs(param, "TimeConst", default=0.1)
                    SetAttrs(param, "TimeConst.Excitatory", GetAttrs(param.TimeConst))
                    SetAttrs(param, "TimeConst.Inhibitory", GetAttrs(param.TimeConst))
                    utils_torch.model.ParseExciInhiNum(param.Neurons)
            ExciNeuronsNum = param.Neurons.Excitatory.Num
            InhiNeuronsNum = param.Neurons.Inhibitory.Num
            #ExciNeuronsNum = 80

            if param.TimeConst.Excitatory==param.TimeConst.Inhibitory:
                TimeConst = param.TimeConst.Excitatory
                Modules.CellStateDecay = lambda CellState: TimeConst * CellState
                Modules.ProcessTotalInput = \
                    lambda TotalInput: (1.0 - TimeConst) * TotalInput
                Modules.ProcessCellStateAndTotalInput = \
                    lambda CellState, TotalInput: \
                    CellState + Modules.ProcessTotalInput(TotalInput)
            else:
                TimeConstExci = param.TimeConst.Excitatory
                TimeConstInhi = param.TimeConst.Inhibitory
                if not (0.0 <= TimeConstExci <= 1.0 and 0.0 <= TimeConstExci <= 1.0):
                    raise Exception()
                Modules.CellStateDecay = lambda CellState: \
                    torch.concatenate([
                            CellState[:, :ExciNeuronsNum] * TimeConstExci, 
                            CellState[:, ExciNeuronsNum:] * TimeConstInhi
                        ], 
                        axis=1
                    )
                Modules.ProcessTotalInput = lambda TotalInput: \
                    torch.concatenate([
                            TotalInput[:, :ExciNeuronsNum] * (1.0 - TimeConstExci),
                            TotalInput[:, ExciNeuronsNum:] * (1.0 - TimeConstInhi),
                        ], 
                        axis=1
                    )
                Modules.ProcessCellStateAndTotalInput = lambda CellState, TotalInput: \
                    CellState + Modules.ProcessTotalInput(TotalInput)
        else:
            if cache.IsInit:
                EnsureAttrs(param, "TimeConst", default=0.1)
            TimeConst = GetAttrs(param.TimeConst)
            if not 0.0 <= TimeConst <= 1.0:
                raise Exception()
            Modules.CellStateDecay = lambda CellState: TimeConst * CellState
            Modules.ProcessTotalInput = lambda TotalInput: (1.0 - TimeConst) * TotalInput
            Modules.ProcessCellStateAndTotalInput = lambda CellState, TotalInput: \
                CellState + Modules.ProcessTotalInput(TotalInput)
    def forward(self, CellState, RecurrentInput, Input):
        cache = self.cache
        return utils_torch.CallGraph(cache.Dynamics.Main, [CellState, RecurrentInput, Input])  
    # def SetLogger(self, logger):
    #     return utils_torch.model.SetLoggerForModel(self, logger)
    # def GetLogger(self):
    #     return utils_torch.model.GetLoggerForModel(self)
    # def Log(self, data, Name="Undefined"):
    #     return utils_torch.model.LogForModel(self, data, Name)
__MainClass__ = RecurrentLIFLayer
utils_torch.model.SetMethodForModelClass(__MainClass__)