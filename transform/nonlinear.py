import torch
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
from DLUtils.attr import *

def Build(param):
    model = NonLinearLayer()
    model.Build(param)
    return model

def ParseNonLinearMethod(param):
    if isinstance(param, str):
        param = DLUtils.PyObj({
            "Type": param,
            "Coefficient": 1.0
        })
    elif isinstance(param, list):
        if len(param)==2:
            param = DLUtils.PyObj({
                "Type": param[0],
                "Coefficient": param[1]
            })
        else:
            # to be implemented
            pass
    elif isinstance(param, DLUtils.PyObj):
        if not hasattr(param, "Coefficient"):
            param.Coefficient = 1.0
    else:
        raise Exception("ParseNonLinearMethod: invalid param Type: %s"%type(param))
    return param

def GetNonLinearMethod(param, **kw):
    param = ParseNonLinearMethod(param)
    if param.Type in ["NonLinear"]:
        if hasattr(param, "Subtype"):
            Type = param.Subtype
    else:
        Type = param.Type

    if Type in ["relu", "ReLU"]:
        if param.Coefficient==1.0:
            return F.relu
        else:
            return lambda x:param.Coefficient * F.relu(x)
    elif Type in ["tanh", "Tanh"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)       
    elif Type in ["sigmoid", "Sigmoid"]:
        if param.Coefficient==1.0:
            return F.tanh
        else:
            return lambda x:param.Coefficient * F.tanh(x)         
    else:
        raise Exception("GetNonLinearMethod: Invalid nonlinear function Type: %s"%Type)
GetActivationFunction = GetNonLinearMethod

from DLUtils.transform.SingleLayer import SingleLayer
class NonLinearLayer(SingleLayer):
    DataIsNotEmpty = True
    # def __init__(self, param=None, data=None, **kw):
    #     super().__init__()
    #     self.InitModule(self, param, data, ClassPath="DLUtils.transform.NonLinearLayer", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        super().Build(IsLoad)
        param = self.param        
        data = self.data
        cache = self.cache

        SetAttrs(param, "Type", value="NonLinearLayer")
        EnsureAttr(param.Subtype, "f(Wx+b)")     

        if param.Subtype in ["f(Wx+b)"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", True)
                SetAttrs(param, "Bias.Size", param.Output.Num)
            self.SetWeight()
            self.SetBias()
            self.NonLinear = DLUtils.transform.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + self.GetBias())
        elif param.Subtype in ["f(Wx)+b"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", True)
                SetAttrs(param, "Bias.Size", param.Output.Num)
            self.SetWeight()
            self.SetBias()
            self.NonLinear = DLUtils.transform.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight())) + data.Bias
        elif param.Subtype in ["f(Wx)"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", False)
            self.SetWeight()
            self.NonLinear = DLUtils.transform.GetNonLinearMethod(param.NonLinear)
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()))
        elif param.Subtype in ["f(W(x+b))"]:
            if cache.IsInit:
                SetAttrs(param, "Bias", True)
                SetAttrs(param, "Bias.Size", param.Input.Num)
            self.SetWeight()
            self.SetBias()
            self.forward = lambda x:self.NonLinear(torch.mm(x, self.GetWeight()) + data.Bias)       
        else:
            if param.Subtype in ["Wx", "Wx+b"]:
                raise Exception("NonLinearLayer: Invalid Subtype. Try using LinearLayer.: %s"%param.Subtype)
            else:
                raise Exception("NonLinearLayer: Invalid Subtype: %s"%param.Subtype)
        return self
    
__MainClass__ = NonLinearLayer
#DLUtils.transform.SetMethodForTransformModule(__MainClass__)