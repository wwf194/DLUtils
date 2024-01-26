import DLUtils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    np = DLUtils.GetLazyNumpy()
    torch = DLUtils.GetLazyTorch()
    nn = DLUtils.LazyImport("torch.nn")
    F = DLUtils.LazyImport("torch.nn.functional")

DefaultRoutings = [
    "&GetBias |--> bias",
    "&GetNoise |--> noise",
    "hiddenState, input, noise, bias |--> &Add |--> inputTotal",
    "inputTotal, membranePotential |--> &ProcessInputTotalAndmembranePotential |--> membranePotential",
    "membranePotential |--> &NonLinear |--> hiddenState",
    "hiddenState |--> &HiddenStateTransform |--> hiddenState",
    "membranePotential |--> &MembranePotentialDecay |--> membranePotential"
]

from DLUtils.transform import AbstractTransformWithTensor
class RecurrentLIFLayer(AbstractTransformWithTensor):
    # def __init__(self, param=None, data=None, **kw):
    #     super(RecurrentLIFLayer, self).__init__()
    #     self.InitModule(self, param, data, ClassPath="DLUtils.transform.RecurrentLIFLayer", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        data = self.data
        cache = self.cache
        EnsureAttrs(param, "IsExciInhi", default=False)
        
        self.BuildModules(IsLoad=IsLoad)
        self.InitModules(IsLoad=IsLoad)
        self.SetInternalMethods()
        self.ParseRouters()

        return self
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
                    DLUtils.transform.ParseExciInhiNum(param.Neurons)
            ExciNeuronsNum = param.Neurons.Excitatory.Num
            InhiNeuronsNum = param.Neurons.Inhibitory.Num
            #ExciNeuronsNum = 80

            if param.TimeConst.Excitatory==param.TimeConst.Inhibitory:
                TimeConst = param.TimeConst.Excitatory
                Modules.MembranePotentialDecay = lambda MembranePotential: (1.0 - TimeConst) * MembranePotential
                Modules.ProcessTotalInput = \
                    lambda TotalInput: TimeConst * TotalInput
                Modules.ProcessMembranePotentialAndTotalInput = \
                    lambda MembranePotential, TotalInput: \
                    MembranePotential + Modules.ProcessTotalInput(TotalInput)
            else:
                TimeConstExci = param.TimeConst.Excitatory
                TimeConstInhi = param.TimeConst.Inhibitory
                if not (0.0 <= TimeConstExci <= 1.0 and 0.0 <= TimeConstExci <= 1.0):
                    raise Exception()
                Modules.MembranePotentialDecay = lambda MembranePotential: \
                    torch.concatenate([
                            MembranePotential[:, :ExciNeuronsNum] * (1.0 - TimeConstExci), 
                            MembranePotential[:, ExciNeuronsNum:] * (1.0 - TimeConstInhi),
                        ], 
                        axis=1
                    )
                Modules.ProcessTotalInput = lambda TotalInput: \
                    torch.concatenate([
                            TotalInput[:, :ExciNeuronsNum] * TimeConstExci,
                            TotalInput[:, ExciNeuronsNum:] * TimeConstInhi,
                        ], 
                        axis=1
                    )
                Modules.ProcessMembranePotentialAndTotalInput = lambda MembranePotential, TotalInput: \
                    MembranePotential + Modules.ProcessTotalInput(TotalInput)
        else:
            if cache.IsInit:
                EnsureAttrs(param, "TimeConst", default=0.1)
            TimeConst = GetAttrs(param.TimeConst)
            if not 0.0 <= TimeConst <= 1.0:
                raise Exception()
            Modules.MembranePotentialDecay = lambda MembranePotential: (1.0 - TimeConst) * MembranePotential
            Modules.ProcessTotalInput = lambda TotalInput: TimeConst * TotalInput
            Modules.ProcessMembranePotentialAndTotalInput = lambda MembranePotential, TotalInput: \
                MembranePotential + Modules.ProcessTotalInput(TotalInput)
    # def forward(self, MembranePotential, RecurrentInput, Input):
    #     cache = self.cache
    #     return DLUtils.CallGraph(cache.Dynamics.Main, [MembranePotential, RecurrentInput, Input])
    def Run(self, recurrentInput, membranePotential, input, log=None):
        Modules = self.Modules
        bias = Modules.GetBias()
        noise = Modules.GenerateNoise(input)
        inputTotal = recurrentInput + input + bias + noise
        membranePotential = Modules.ProcessMembranePotentialAndTotalInput(inputTotal, membranePotential)
        firingRate = Modules.NonLinear(membranePotential)
        output = Modules.FiringRate2Output(firingRate)
        recurrentInputNext = Modules.FiringRate2RecurrentInput(firingRate)
        membranePotentialNext = Modules.MembranePotentialDecay(membranePotential)
        return recurrentInputNext, membranePotentialNext, output, firingRate
        # "Main":{
        #     "In": ["recurrentInput", "membranePotential", "input"],
        #     "Out": ["recurrentInputNext", "membranePotentialNext", "firingRate", "output"],
        #     "Routings":[
        #         "                       &GetBias                   |--> bias",
        #         "recurrentInput, input, bias |--> &Add             |--> inputTotal",
        #         "inputTotal        |--> &NoiseGenerator            |--> noise",
        #         "inputTotal, noise |--> &Add                       |--> inputTotal",
        #         "inputTotal, membranePotential |--> &ProcessMembranePotentialAndTotalInput |--> membranePotential",
        #         "membranePotential |--> &NonLinear                 |--> firingRate",
        #         "firingRate        |--> &FiringRate2RecurrentInput |--> recurrentInputNext",
        #         "firingRate        |--> &FiringRate2Output         |--> output",
        #         "membranePotential |--> &MembranePotentialDecay    |--> membranePotentialNext",
        #     ]
        # },
    def __call__(self, *Args, **Kw):
        return self.Run(*Args, **Kw)
__MainClass__ = RecurrentLIFLayer
# DLUtils.transform.SetMethodForTransformModule(__MainClass__)