import torch
import torch.nn as nn
import torch.nn.functional as F

from DLUtils.attr import *

from DLUtils.transform import AbstractTransform
class LambdaLayer(AbstractTransform):
    # def __init__(self, param=None, data=None, **kw):
    #     super(LambdaLayer, self).__init__()
    #     #self.InitModule(self, param, data, ClassPath="DLUtils.transform.LambdaLayer", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        self.forward = DLUtils.parse.ResolveStr(param.Lambda, ObjCurrent=param.cache.__ResolveRef__)
        return self
    def SetFullName(self, FullName):
        DLUtils.transform.SetFullNameForModule(self, FullName)

__MainClass__ = LambdaLayer
#DLUtils.transform.SetMethodForTransformModule(__MainClass__)