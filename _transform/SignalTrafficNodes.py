import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import DLUtils
from DLUtils.attr import *

class SignalHolder(DLUtils.module.AbstractModule):
    # def __init__(self, param=None, data=None, **kw):
    #     kw.setdefault("HasTensor", False)
    #     self.InitModule(self, param, data, ClassPath="DLUtils.transform.SignalHolder", **kw)
    HasTensor = False
    def __init__(self, **kw):
        super().__init__(**kw)
        return
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        return
    def Receive(self, Obj):
        self.cache.Content = Obj
    def Send(self):
        return self.cache.Content
    def Clear(self):
        DLUtils.attr.RemoveAttrIfExists(self.cache, "Content")
#DLUtils.transform.SetMethodForTransformModule(SignalHolder, HasTensor=False)

from DLUtils.transform import AbstractTransform
class SerialSender(AbstractTransform):
    # def __init__(self, param=None, data=None, **kw):
    #     #super(SerialSender, self).__init__()
    #     self.InitModule(self, param, data,  ClassPath="DLUtils.transform.SerialSender", **kw)
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        param = self.param
        cache = self.cache
        cache.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
    def SetSendMethod(self):
        param = self.param
        EnsureAttrs(param, "Send.Method", default="Default")
        method = GetAttrs(param.Send.Method)
        if method in ["Default"]:
            self._Send = self.SendDefault
        elif method in ["Lambda", "eval"]:
            self._Send = eval(GetAttrs(param.Send.Args))
        else:
            raise Exception(method)
        return
    def SetReceiveMethod(self):
        param = self.param
        EnsureAttrs(param, "Receive.Method", default="Default")
        method = GetAttrs(param.Receive.Method)
        if method in ["Default"]:
            self.Receive = self.ReceiveDefault
        elif method in ["Lambda", "eval"]:
            self.Receive = eval(GetAttrs(param.Receive.Args))
        else:
            raise Exception(method)
        self.append = self.Receive
        return
    def ReceiveDefault(self, content):
        cache = self.cache
        cache.ContentList = content
        cache.NextSendIndex = 0
    def _SendDefault(self, ContentList, Index):
        return ContentList[Index]
    def RegisterExtractMethod(self, method):
        self.ExtractMethod = method
    def Send(self):
        cache = self.cache
        Content = self._Send(cache.ContentList, Index=cache.NextSendIndex)
        cache.NextSendIndex += 1
        return Content
#DLUtils.transform.SetMethodForTransformModule(SerialSender, HasTensor=False)

class SerialReceiver(AbstractTransform):
    # def __init__(self, param=None, data=None, **kw):
    #     self.InitModule(self, param, data, ClassPath="DLUtils.transform.SerialReceiver", **kw)
    def GenerateParam(self, Type):
        if Type in ["ActivityAlongTime"]:
            return DLUtils.PyObj({
                "Type": "SerialReceiver",
                "Send": {
                    "Method": "Lambda", "Args": "lambda List:torch.stack(List, axis=1)"
                }
            })
        else:
            raise Exception(Type)
    def LoadParam(self, param=None, Type=None):
        if Type is not None:
            super().LoadParam(self.GenerateParam(Type))
        else:
            super().LoadParam(param)
        return self
    def __init__(self, **kw):
        super().__init__(**kw)
    def Build(self, IsLoad=False):
        self.BeforeBuild(IsLoad)
        cache = self.cache
        self.ContentList = []
        self.SetSendMethod()
        self.SetReceiveMethod()
        return self
    def SetSendMethod(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
            EnsureAttrs(param, "Send.Method", default="Default")
        method = GetAttrs(param.Send.Method)
        if method in ["Default"]:
            self._Send = self._SendDefault
        elif method in ["Lambda", "eval"]:
            self._Send = eval(GetAttrs(param.Send.Args))
        else:
            raise Exception(method)
        return
    def SetReceiveMethod(self):
        param = self.param
        cache = self.cache
        if cache.IsInit:
            EnsureAttrs(param, "Receive.Method", default="Default")
        method = GetAttrs(param.Receive.Method)
        if method in ["Default"]:
            self.Receive = self.ReceiveDefault
        elif method in ["Lambda", "eval"]:
            self.Receive = eval(GetAttrs(param.Receive.Args))
        else:
            raise Exception(method)
        self.append = self.Receive
        return
    def ReceiveDefault(self, content):
        self.ContentList.append(content)
    def _SendDefault(self, List):
        return List
    def Send(self):
        result = self._Send(self.ContentList)
        self.ContentList = []
        return result
    def SendWithoutFlush(self):
        return self.ContentList
#DLUtils.transform.SetMethodForTransformModule(SerialReceiver, HasTensor=False)