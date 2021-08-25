import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_torch.model import *
from utils_torch.utils import *
from utils_torch.utils import HasAttrs, EnsureAttrs, MatchAttrs, compose_function, SetAttrs
from utils_torch.model import GetNonLinearFunction, GetConstraintFunction, CreateSelfConnectionMask, CreateExcitatoryInhibitoryMask, Create2DWeight

def InitFromParams(param):
    model = SingleLayer()
    model.InitFromParams(param)
    return model

def load_model(args):
    return 

class SingleLayer(nn.Module):
    def __init__(self, param=None):
        super(SingleLayer, self).__init__()
        if param is not None:
            self.InitFromParams(param)
    def InitFromParams(self, param):
        super(SingleLayer, self).__init__()
        SetAttrs(param, "Type", value="SingleLayer")
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")
        self.param = param
        EnsureAttrs(param, "Subtype", default="f(Wx+b)")

        EnsureAttrs(param, "Weight", default=utils_torch.PyObj(
            {"Initialize":{"Method":"kaiming", "Coefficient":1.0}}))

        if not HasAttrs(param.Weight, "Size"):
            SetAttrs(param.Weight, "Size", value=[param.Input.Num, param.Output.Num])
        if param.Subtype in ["f(Wx+b)"]:
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = GetNonLinearFunction(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.get_Weight()) + self.Bias)
        elif param.Subtype in ["f(Wx)+b"]:
            self.CreateWeight()
            self.CreateBias()
            self.NonLinear = GetNonLinearFunction(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.get_Weight())) + self.Bias
        elif param.Subtype in ["Wx"]:
            self.CreateWeight()
            self.forward = lambda x:torch.mm(x, self.get_Weight())
        elif param.Subtype in ["Wx+b"]:
            self.CreateWeight()
            self.CreateBias()
            self.forward = lambda x:torch.mm(x, self.get_Weight()) + self.Bias         
        else:
            raise Exception("SingleLayer: Invalid Subtype: %s"%param.Subtype)
    def CreateBias(self, Size=None):
        param = self.param
        EnsureAttrs(param, "Bias", default=False)
        if Size is None:
            Size = param.Weight.Size[1]
        if MatchAttrs(param.Bias, value=False):
            self.Bias = 0.0
        elif MatchAttrs(param.Bias, value=True):
            self.Bias = torch.nn.Parameter(torch.zeros(Size))
        else:
            # to be implemented 
            raise Exception()
    def CreateWeight(self):
        param = self.param
        sig = HasAttrs(param.Weight, "Size")
        if not HasAttrs(param.Weight, "Size"):
            SetAttrs(param.Weight, "Size", value=[param.Input.Num, param.Output.Num])
        self.Weight = torch.nn.Parameter(Create2DWeight(param.Weight))
        get_Weight_function = [lambda :self.Weight]
        if MatchAttrs(param.Weight, "isExciInhi", value=True):
            self.ExciInhiMask = CreateExcitatoryInhibitoryMask(*param.Weight.Size, param.Weight.excitatory.Num, param.Weight.inhibitory.Num)
            get_Weight_function.append(lambda Weight:Weight * self.ExciInhiMask)
            EnsureAttrs(param.Weight, "ConstraintMethod", value="AbsoluteValue")
            self.WeightConstraintMethod = GetConstraintFunction(param.Weight.ConstraintMethod)
            get_Weight_function.append(self.WeightConstraintMethod)
        if MatchAttrs(param.Weight, "NoSelfConnection", value=True):
            if param.Weight.Size[0] != param.Weight.Size[1]:
                raise Exception("NoSelfConnection requires Weight to be square matrix.")
            self.SelfConnectionMask = CreateSelfConnectionMask(param.Weight.Size[0])            
            get_Weight_function.append(lambda Weight:Weight * self.SelfConnectionMask)
        self.get_Weight = compose_function(get_Weight_function)